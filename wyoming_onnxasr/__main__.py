#!/usr/bin/env python3
import argparse
import asyncio
import logging
from functools import partial
from pathlib import Path

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import ArabicAsrEventHandler

_LOGGER = logging.getLogger(__name__)

async def main() -> None:
    """Main entry point for Arabic ASR Wyoming server."""
    parser = argparse.ArgumentParser(description="Wyoming Arabic ASR Server")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the Arabic ONNX model file",
    )
    parser.add_argument(
        "--tokens-path",
        required=True,
        help="Path to the tokens.txt file",
    )
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for inference",
    )
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    # Validate paths
    model_path = Path(args.model_path)
    tokens_path = Path(args.tokens_path)
    
    if not model_path.exists():
        _LOGGER.error(f"Model file not found: {model_path}")
        return
    
    if not tokens_path.exists():
        _LOGGER.error(f"Tokens file not found: {tokens_path}")
        return

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="arabic-asr-onnx",
                description="Arabic speech-to-text with ONNX Runtime",
                attribution=Attribution(
                    name="Mo-alaa/NeMo",
                    url="https://github.com/NVIDIA/NeMo",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name="arabic-fastconformer-ctc",
                        description="Arabic FastConformer CTC model",
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
            args,
            str(model_path),
            str(tokens_path),
            model_lock,
        )
    )

def run() -> None:
    asyncio.run(main())

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass