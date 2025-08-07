#!/usr/bin/env python3
"""Test script for Arabic ASR"""

import asyncio
import logging
import wave
import numpy as np
from pathlib import Path
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient

async def test_arabic_asr():
    """Test Arabic ASR with audio file"""
    
    # Load your test audio file
    audio_file = "/home/malaa/Aref/NEMO/inference_output_2.wav"
    
    with wave.open(audio_file, "rb") as wav_file:
        frames = wav_file.readframes(-1)
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
    
    print(f"Test audio: {sample_rate}Hz, {channels} channels, {sample_width} bytes/sample")
    
    # Connect to Wyoming server
    try:
        client = AsyncTcpClient("127.0.0.1", 10305)
        await client.connect()
        
        print("Connected to Wyoming server")
        
        # Start audio
        await client.write_event(
            AudioStart(
                rate=sample_rate,
                width=sample_width,
                channels=channels
            ).event()
        )
        print("Sent AudioStart")
        
        # Send transcribe event
        await client.write_event(Transcribe().event())
        print("Sent Transcribe")
        
        # Send audio in chunks
        chunk_size = 1024 * sample_width * channels
        total_chunks = (len(frames) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(frames), chunk_size):
            chunk = frames[i:i + chunk_size]
            await client.write_event(
                AudioChunk(
                    audio=chunk,
                    rate=sample_rate,
                    width=sample_width,
                    channels=channels
                ).event()
            )
            if (i // chunk_size) % 10 == 0:  # Progress every 10 chunks
                print(f"Sent chunk {i // chunk_size + 1}/{total_chunks}")
        
        print("All audio chunks sent")
        
        # Stop audio
        await client.write_event(AudioStop().event())
        print("Sent AudioStop")
        
        # Read transcript with timeout
        print("Waiting for transcript...")
        try:
            timeout_seconds = 30
            transcript_received = False
            
            async def read_with_timeout():
                nonlocal transcript_received
                while not transcript_received:
                    event = await client.read_event()
                    print(f"Received event type: {event.type}")
                    
                    if Transcript.is_type(event.type):
                        transcript = Transcript.from_event(event)
                        print(f"✅ Arabic transcript: '{transcript.text}'")
                        transcript_received = True
                        return transcript.text
                    
            result = await asyncio.wait_for(read_with_timeout(), timeout=timeout_seconds)
            return result
            
        except asyncio.TimeoutError:
            print(f"❌ Timeout after {timeout_seconds} seconds waiting for transcript")
        
    except ConnectionRefusedError:
        print("❌ Connection refused. Make sure the Wyoming server is running:")
        print("   script/run")
        print("   or")
        print("   python -m wyoming_onnxasr --model-path /path/to/model.onnx --tokens-path /path/to/tokens.txt --uri tcp://0.0.0.0:10305")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'client' in locals():
            await client.disconnect()

# Alternative simple test without Wyoming protocol
async def test_direct_model():
    """Test the model directly without Wyoming protocol"""
    print("\n" + "="*50)
    print("Testing model directly (without Wyoming)")
    print("="*50)
    
    try:
        # Use absolute paths
        model_path = "/home/malaa/Aref/nvidia/nemo-onnx2/arabic_ctc_model.onnx"
        tokens_path = "/home/malaa/Aref/nvidia/nemo-onnx2/tokens.txt"
        audio_file = "/home/malaa/Aref/NEMO/inference_output_2.wav"
        
        # Check if files exist
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            return
        
        if not Path(tokens_path).exists():
            print(f"❌ Tokens not found: {tokens_path}")
            return
            
        if not Path(audio_file).exists():
            print(f"❌ Audio file not found: {audio_file}")
            return
        
        print(f"✅ All files found")
        print(f"  Model: {model_path}")
        print(f"  Tokens: {tokens_path}")
        print(f"  Audio: {audio_file}")
        
        # Import the direct ONNX test function
        import sys
        sys.path.append("/home/malaa/Aref/nvidia")
        
        # Modify the onnx_run.py to use absolute paths for testing
        import onnxruntime as ort
        import kaldi_native_fbank as knf
        import librosa
        import numpy as np
        
        # Load tokens
        def load_tokens(tokens_file):
            tokens = {}
            with open(tokens_file, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        token, idx = parts[0], int(parts[1])
                        tokens[idx] = token
            return tokens
        
        # Compute features
        def compute_feat(filename):
            sample_rate = 16000
            samples, _ = librosa.load(filename, sr=sample_rate)
            opts = knf.FbankOptions()
            opts.frame_opts.dither = 0
            opts.frame_opts.snip_edges = False
            opts.frame_opts.samp_freq = sample_rate
            opts.mel_opts.num_bins = 80

            online_fbank = knf.OnlineFbank(opts)
            online_fbank.accept_waveform(sample_rate, (samples * 32768).tolist())
            online_fbank.input_finished()

            features = np.stack(
                [online_fbank.get_frame(i) for i in range(online_fbank.num_frames_ready)]
            )
            mean = features.mean(axis=0, keepdims=True)
            stddev = features.std(axis=0, keepdims=True)
            features = (features - mean) / (stddev + 1e-5)
            return features
        
        # CTC decode
        def decode_ctc_greedy(logits, tokens, blank_id=1024):
            predictions = np.argmax(logits, axis=-1)
            unique_preds = []
            prev_token = None
            
            for pred in predictions:
                if pred != blank_id and pred != prev_token:
                    unique_preds.append(pred)
                prev_token = pred
            
            text_tokens = []
            for pred in unique_preds:
                if pred in tokens:
                    text_tokens.append(tokens[pred])
            
            text = "".join(text_tokens)
            text = text.replace("▁", " ")
            return text.strip()
        
        # Run inference
        print("🔄 Loading model and running inference...")
        
        # Load model
        session = ort.InferenceSession(model_path)
        
        # Load tokens
        tokens = load_tokens(tokens_path)
        print(f"Loaded {len(tokens)} tokens")
        
        # Compute features
        features = compute_feat(audio_file)
        features = np.expand_dims(features, axis=0)
        features = features.transpose(0, 2, 1)
        features_length = np.array([features.shape[2]], dtype=np.int64)
        
        print(f"Features shape: {features.shape}")
        print(f"Features length: {features_length}")
        
        # Run inference
        inputs = {
            session.get_inputs()[0].name: features,
            session.get_inputs()[1].name: features_length,
        }
        
        outputs = session.run([session.get_outputs()[0].name], inputs)
        logits = outputs[0]
        
        print(f"Output logits shape: {logits.shape}")
        
        # Decode
        if len(logits.shape) == 3:
            logits = logits[0]
        
        result = decode_ctc_greedy(logits, tokens)
        print(f"✅ Direct model test result: '{result}'")
        
        return result
        
    except Exception as e:
        print(f"❌ Direct model test failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run both tests"""
    print("🧪 Testing Arabic ASR")
    print("=" * 50)
    
    # Test 1: Direct model (should work)
    await test_direct_model()
    
    print("\n" + "="*50)
    print("Testing Wyoming protocol")
    print("="*50)
    
    # Test 2: Wyoming protocol (needs server running)
    await test_arabic_asr()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())