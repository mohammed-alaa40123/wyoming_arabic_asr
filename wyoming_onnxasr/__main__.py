#!/usr/bin/env python3
"""Arabic ASR Wyoming Server for Home Assistant Add-on with auto-download"""

import argparse
import asyncio
import logging
from functools import partial
from pathlib import Path
import sys
import os
import json
from urllib.parse import urlparse

import numpy as np
import onnxruntime as ort
import kaldi_native_fbank as knf
import librosa
import requests
from huggingface_hub import hf_hub_download, list_repo_files

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop, AudioStart
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Info, Describe
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger(__name__)

def download_model_files(repo_id: str, cache_dir: Path) -> dict:
    """Download model files from Hugging Face Hub"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    _LOGGER.info(f"Downloading model files from {repo_id}...")
    
    try:
        # List all files in the repository
        files = list_repo_files(repo_id)
        _LOGGER.info(f"Found files in repository: {files}")
        
        downloaded_files = {}
        
        # Download required files
        required_files = {
            "arabic_ctc_model.onnx": "model",
            "tokens.txt": "tokens",
            "vocab.txt": "vocab",  # Alternative name
            "config.json": "config"
        }
        
        for filename in files:
            if filename in required_files:
                _LOGGER.info(f"Downloading {filename}...")
                try:
                    file_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        cache_dir=cache_dir,
                        local_dir=cache_dir,
                        local_dir_use_symlinks=False
                    )
                    downloaded_files[required_files[filename]] = file_path
                    _LOGGER.info(f"Downloaded {filename} to {file_path}")
                except Exception as e:
                    _LOGGER.warning(f"Failed to download {filename}: {e}")
        
        # Check if we have the essential files
        if "model" not in downloaded_files:
            # Try alternative model names
            alternative_models = [f for f in files if f.endswith('.onnx')]
            if alternative_models:
                model_file = alternative_models[0]
                _LOGGER.info(f"Using alternative model file: {model_file}")
                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=model_file,
                    cache_dir=cache_dir,
                    local_dir=cache_dir,
                    local_dir_use_symlinks=False
                )
                downloaded_files["model"] = file_path
        
        if "tokens" not in downloaded_files and "vocab" in downloaded_files:
            # Use vocab.txt as tokens if tokens.txt is not available
            downloaded_files["tokens"] = downloaded_files["vocab"]
        
        if "tokens" not in downloaded_files:
            # Try to find any txt file that might be the vocabulary
            txt_files = [f for f in files if f.endswith('.txt')]
            if txt_files:
                tokens_file = txt_files[0]
                _LOGGER.info(f"Using {tokens_file} as tokens file")
                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=tokens_file,
                    cache_dir=cache_dir,
                    local_dir=cache_dir,
                    local_dir_use_symlinks=False
                )
                downloaded_files["tokens"] = file_path
        
        # Validate essential files
        if "model" not in downloaded_files:
            raise ValueError("No ONNX model file found in repository")
        
        if "tokens" not in downloaded_files:
            raise ValueError("No tokens/vocabulary file found in repository")
        
        _LOGGER.info("Model files downloaded successfully!")
        return downloaded_files
        
    except Exception as e:
        _LOGGER.error(f"Failed to download model files: {e}")
        raise

class ArabicAsrEventHandler(AsyncEventHandler):
    """Event handler for Arabic ASR clients."""

    def __init__(
        self,
        wyoming_info: Info,
        model_path: str,
        tokens_path: str,
        device: str,
        model_lock: asyncio.Lock,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

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
        if device == "cuda":
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
                elif len(parts) == 1:
                    # Handle vocab.txt format (one token per line)
                    token = parts[0]
                    idx = len(tokens)
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

async def main() -> None:
    """Main entry point for Arabic ASR Wyoming server."""
    parser = argparse.ArgumentParser(description="Wyoming Arabic ASR Server")
    parser.add_argument(
        "--model-repo",
        default="Mo-alaa/arabic-fastconformer-ctc-onnx",
        help="Hugging Face model repository",
    )
    parser.add_argument(
        "--cache-dir",
        default="/share/arabic_asr_cache",
        help="Directory to cache downloaded models",
    )
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for inference",
    )
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s"
    )
    _LOGGER.debug(args)

    # Download model files
    cache_dir = Path(args.cache_dir)
    try:
        downloaded_files = download_model_files(args.model_repo, cache_dir)
        model_path = downloaded_files["model"]
        tokens_path = downloaded_files["tokens"]
    except Exception as e:
        _LOGGER.error(f"Failed to download model: {e}")
        return

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="arabic-asr-onnx",
                description="Arabic speech-to-text with ONNX Runtime",
                attribution=Attribution(
                    name="Mo-alaa/NeMo",
                    url=f"https://huggingface.co/{args.model_repo}",
                ),
                installed=True,
                version="1.0.0",
                models=[
                    AsrModel(
                        name="arabic-fastconformer-ctc",
                        description=f"Arabic FastConformer CTC model from {args.model_repo}",
                        attribution=Attribution(
                            name="NVIDIA NeMo",
                            url="https://catalog.ngc.nvidia.com/orgs/nvidia/collections/nemo_asr",
                        ),
                        installed=True,
                        languages=["ar", "ar-EG", "ar-SA", "ar-AE"],
                        version="1.0.0",
                    )
                ],
            )
        ],
    )

    _LOGGER.info(f"Loading Arabic ASR model from {model_path}")
    _LOGGER.info(f"Using tokens from {tokens_path}")
    _LOGGER.info(f"Device: {args.device}")

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Arabic ASR Wyoming server ready")
    
    model_lock = asyncio.Lock()
    await server.run(
        partial(
            ArabicAsrEventHandler,
            wyoming_info,
            str(model_path),
            str(tokens_path),
            args.device,
            model_lock,
        )
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass