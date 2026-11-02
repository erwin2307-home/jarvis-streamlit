"""TTS — Piper (real) + Fake. Streams float32 audio chunks."""
from __future__ import annotations

import asyncio
import math
import queue as _queue
from pathlib import Path
from typing import AsyncIterator, Protocol

import numpy as np
import numpy.typing as npt


class Tts(Protocol):
    def synthesize(self, text: str) -> AsyncIterator[npt.NDArray[np.float32]]: ...


class RealTts:
    """Piper voice via piper-tts package. True streaming: each Piper chunk is
    sub-chunked and yielded to the consumer as soon as it lands on a worker
    thread, so the speaker starts playing well before synthesis finishes.
    """

    _SENTINEL: object = object()

    def __init__(
        self, model_path: Path, chunk_size: int = 2048, length_scale: float = 1.0
    ) -> None:
        from piper import PiperVoice
        from piper.config import SynthesisConfig

        self._voice = PiperVoice.load(str(model_path))
        self._chunk_size = chunk_size
        # length_scale > 1.0 slows speech down (Piper/VITS convention:
        # multiplies phoneme duration) — slower speech reads as clearer
        # for a low-quality-tier voice.
        self._syn_config = SynthesisConfig(length_scale=length_scale)

    async def synthesize(self, text: str) -> AsyncIterator[npt.NDArray[np.float32]]:
        if not text.strip():
            return
        loop = asyncio.get_running_loop()

        # Thread-safe queue with blocking put gives natural backpressure: when
        # the consumer can't keep up, the producer thread blocks on Queue.put
        # rather than dropping chunks (which an asyncio.Queue + put_nowait
        # would do).
        q: _queue.Queue[npt.NDArray[np.float32] | object] = _queue.Queue(maxsize=8)

        def _producer() -> None:
            try:
                for audio_chunk in self._voice.synthesize(text, syn_config=self._syn_config):
                    arr: npt.NDArray[np.float32] = np.asarray(
                        audio_chunk.audio_float_array, dtype=np.float32
                    )
                    for i in range(0, arr.size, self._chunk_size):
                        q.put(arr[i : i + self._chunk_size])
            finally:
                q.put(self._SENTINEL)

        producer_fut = loop.run_in_executor(None, _producer)
        try:
            while True:
                # run blocking get in executor so we don't stall the loop.
                item = await loop.run_in_executor(None, q.get)
                if item is self._SENTINEL:
                    break
                assert isinstance(item, np.ndarray)
                yield item
        finally:
            await producer_fut


class EdgeTts:
    """Microsoft Edge's free neural TTS (edge-tts package, no API key) —
    needs internet, but real neural quality. Used because Piper's only
    German female voices (eva_k, kerstin, ramona) are low/x-low tier and
    were rejected as too low quality.

    edge-tts streams MP3 chunks over the network, but individual chunks
    aren't independently decodable (MP3 frame boundaries don't align with
    network packet boundaries) — so the full response is buffered, then
    decoded once, then re-chunked for playback streaming. This trades a
    bit of time-to-first-audio for correctness; the alternative (decoding
    partial MP3 data) risks garbled or failed decodes.
    """

    # A few MB is ample for a multi-sentence spoken reply; caps the
    # buffer against a runaway/misbehaving response instead of growing
    # unbounded (resource exhaustion). The stream is also time-bounded
    # overall, independent of the byte cap.
    _MAX_MP3_BYTES = 5 * 1024 * 1024
    _STREAM_TIMEOUT_S = 30.0

    _warned_once = False

    def __init__(
        self,
        voice: str = "de-DE-KatjaNeural",
        chunk_size: int = 2048,
        rate: str = "+0%",
    ) -> None:
        self._voice = voice
        self._chunk_size = chunk_size
        self._rate = rate
        if not EdgeTts._warned_once:
            from loguru import logger

            logger.warning(
                "EdgeTts active: every spoken response's text is sent to "
                "Microsoft's edge-tts endpoint over the internet. Set "
                "JARVIS_TTS_BACKEND=piper for a fully offline (lower "
                "quality) voice."
            )
            EdgeTts._warned_once = True

    async def synthesize(self, text: str) -> AsyncIterator[npt.NDArray[np.float32]]:
        if not text.strip():
            return
        import asyncio
        import io

        import edge_tts
        import soundfile as sf  # type: ignore[import-untyped]
        from loguru import logger

        communicate = edge_tts.Communicate(text, self._voice, rate=self._rate)
        mp3_bytes = bytearray()

        async def _collect() -> None:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_bytes.extend(chunk["data"])
                    if len(mp3_bytes) > self._MAX_MP3_BYTES:
                        raise RuntimeError(
                            f"edge-tts response exceeded {self._MAX_MP3_BYTES} bytes"
                        )

        try:
            await asyncio.wait_for(_collect(), timeout=self._STREAM_TIMEOUT_S)
        except (TimeoutError, RuntimeError) as exc:
            logger.warning(f"EdgeTts synthesis aborted (ignored): {exc!r}")
            return

        if not mp3_bytes:
            return
        audio, _sr = sf.read(io.BytesIO(bytes(mp3_bytes)), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # downmix to mono if stereo
        for i in range(0, audio.size, self._chunk_size):
            yield audio[i : i + self._chunk_size]
            await asyncio.sleep(0)  # yield to event loop between chunks


class FakeTts:
    """Emits multiple zeros-chunks per word to simulate streaming TTS.

    Each word produces ``_chunks_per_word`` chunks of ``_n`` samples each so
    that concurrent tasks (e.g. barge-in detection) have enough scheduling
    opportunities to run between chunks.
    """

    _SAMPLES_PER_WORD: int = 1024  # virtual audio per word before chunking

    def __init__(self, samples_per_chunk: int = 1024) -> None:
        self._n = samples_per_chunk
        # At least 1 chunk per word; more chunks when samples_per_chunk is small.
        self._chunks_per_word = max(1, math.ceil(self._SAMPLES_PER_WORD / self._n))

    async def synthesize(self, text: str) -> AsyncIterator[npt.NDArray[np.float32]]:
        for word in text.split():
            if not word:
                continue
            for _ in range(self._chunks_per_word):
                yield np.zeros(self._n, dtype=np.float32)
                await asyncio.sleep(0)  # yield to event loop between chunks
