"""Standalone TTS smoke — run via `uv run python tests/_smoke_tts.py`.
NOT a pytest. Touches real audio hardware.
"""
import asyncio
import sys
import time
from pathlib import Path

from jarvis.audio_io import RealSpeaker
from jarvis.tts import RealTts


async def main() -> None:
    voice = Path("C:/Users/Erwin/Desktop/projekte/jarvis/models/piper/en_GB-alan-medium.onnx")
    print("loading voice...", flush=True)
    tts = RealTts(voice)
    sp = RealSpeaker(sample_rate=22050)
    text = "Good evening, sir. How may I assist you today?"
    print(f"synthesizing: {text!r}", flush=True)
    t0 = time.monotonic()
    first_chunk_at: float | None = None
    n = 0
    async for chunk in tts.synthesize(text):
        if first_chunk_at is None:
            first_chunk_at = time.monotonic() - t0
            print(f"  first chunk at {first_chunk_at:.2f}s", flush=True)
        n += 1
        await sp.play(chunk)
    elapsed = time.monotonic() - t0
    print(f"total: {n} chunks in {elapsed:.2f}s", flush=True)
    await asyncio.sleep(3.0)
    await sp.aclose()


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)
