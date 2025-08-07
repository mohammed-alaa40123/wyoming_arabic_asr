"""Event handler for Arabic ASR clients."""
import argparse
import asyncio
import logging
from typing import Any
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import kaldi_native_fbank as knf
import librosa
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop, AudioStart
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)

class ArabicAsrEventHandler(AsyncEventHandler):
    """Event handler for clients using Arabic ONNX ASR model."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model_path: str,
        tokens_path: str,
        model_lock: asyncio.Lock,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model_path = model_path
        self.tokens_path = tokens_path
        self.model_lock = model_lock
        self.audio_buffer: bytearray | None = None
        self.sample_rate = 16000
        self.sample_width = 2
        self.channels = 1
        
        # Load ONNX session
        providers = []
        if cli_args.device == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        _LOGGER.info(f"Arabic ASR model loaded from {model_path}")
        
        # Load tokens
        self.tokens = self._load_tokens()
        _LOGGER.info(f"Loaded {len(self.tokens)} tokens")

    def _load_tokens(self):
        """Load tokens from file"""
        tokens = {}
        with open(self.tokens_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    token, idx = parts[0], int(parts[1])
                    tokens[idx] = token
        return tokens

    def _compute_features(self, audio_samples):
        """Compute fbank features from audio samples"""
        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = self.sample_rate
        opts.mel_opts.num_bins = 80

        online_fbank = knf.OnlineFbank(opts)
        online_fbank.accept_waveform(self.sample_rate, (audio_samples * 32768).tolist())
        online_fbank.input_finished()

        features = np.stack(
            [online_fbank.get_frame(i) for i in range(online_fbank.num_frames_ready)]
        )
        
        # Normalize features
        mean = features.mean(axis=0, keepdims=True)
        stddev = features.std(axis=0, keepdims=True)
        features = (features - mean) / (stddev + 1e-5)
        
        return features

    def _decode_ctc_greedy(self, logits, blank_id=1024):
        """Simple greedy CTC decoding"""
        # Get most likely token at each timestep
        predictions = np.argmax(logits, axis=-1)
        
        # Remove consecutive duplicates and blanks
        unique_preds = []
        prev_token = None
        
        for pred in predictions:
            if pred != blank_id and pred != prev_token:
                unique_preds.append(pred)
            prev_token = pred
        
        # Convert to text
        text_tokens = []
        for pred in unique_preds:
            if pred in self.tokens:
                text_tokens.append(self.tokens[pred])
        
        # Join tokens (handle subword tokens)
        text = "".join(text_tokens)
        text = text.replace("▁", " ")  # Replace subword separator
        return text.strip()

    async def handle_event(self, event: Event) -> bool:
        if AudioStart.is_type(event.type):
            audio_start = AudioStart.from_event(event)
            self.sample_rate = audio_start.rate
            self.sample_width = audio_start.width
            self.channels = audio_start.channels
            self.audio_buffer = bytearray()
            _LOGGER.debug(f"Audio started: {self.sample_rate} Hz, {self.sample_width*8}-bit, {self.channels} channel(s)")
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)

            if self.audio_buffer is None:
                self.audio_buffer = bytearray()

            self.audio_buffer.extend(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped. Transcribing Arabic...")
            assert self.audio_buffer is not None

            # Convert audio buffer to numpy array
            audio_s16 = np.frombuffer(self.audio_buffer, dtype=np.int16)
            audio_f32 = audio_s16.astype(np.float32) / 32768.0

            async with self.model_lock:
                try:
                    # Compute features
                    features = self._compute_features(audio_f32)  # (T, C)
                    features = np.expand_dims(features, axis=0)  # (N, T, C)
                    features = features.transpose(0, 2, 1)  # (N, C, T)
                    features_length = np.array([features.shape[2]], dtype=np.int64)
                    
                    # Prepare inputs
                    inputs = {
                        self.session.get_inputs()[0].name: features,
                        self.session.get_inputs()[1].name: features_length,
                    }
                    
                    # Run inference
                    outputs = self.session.run([self.session.get_outputs()[0].name], inputs)
                    logits = outputs[0]  # Shape: [batch, time, vocab]
                    
                    # Decode
                    if len(logits.shape) == 3:
                        logits = logits[0]  # Remove batch dimension
                    
                    transcription = self._decode_ctc_greedy(logits)
                    
                except Exception as e:
                    _LOGGER.error(f"Transcription error: {e}")
                    transcription = ""

            self.audio_buffer = None
            _LOGGER.info(f"Arabic transcription: {transcription}")

            await self.write_event(Transcript(text=transcription).event())
            _LOGGER.debug("Completed Arabic ASR request")

            return True

        if Transcribe.is_type(event.type):
            self.audio_buffer = None
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True