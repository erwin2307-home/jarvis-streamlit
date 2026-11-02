"""Microphone capture + speaker playback. Real (sounddevice) + Fake (test) implementations."""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import sounddevice as sd  # type: ignore[import-untyped]


# ── Protocols ─────────────────────────────────────────────────────────

@runtime_checkable
class MicStream(Protocol):
    def frames(self) -> AsyncIterator[npt.NDArray[np.int16]]: ...


@runtime_checkable
class Speaker(Protocol):
    async def play(self, samples: npt.NDArray[Any]) -> None: ...


# ── Utility ───────────────────────────────────────────────────────────

def rms_dbfs(int16_samples: npt.NDArray[np.int16]) -> float:
    """Return RMS level in dBFS. Returns a very negative value for digital silence."""
    if int16_samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(int16_samples.astype(np.float64)))))
    if rms <= 0:
        return -120.0
    full_scale = 32767.0
    return float(20.0 * np.log10(rms / full_scale))


# ── Real (sounddevice) ─────────────────────────────────────────────────

class RealMicStream:
    """Captures from default input device. Yields int16 frames of `frame_size` samples."""

    def __init__(self, sample_rate: int = 16000, frame_size: int = 512) -> None:
        # Silero VAD on 16 kHz requires >= 512 samples per chunk (sr/x.shape[1] > 31.25).
        # 512 samples = 32 ms at 16 kHz, still well within real-time budgets.
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self._queue: asyncio.Queue[npt.NDArray[np.int16]] = asyncio.Queue()
        self._stream: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _callback(
        self,
        indata: npt.NDArray[Any],
        frames: int,
        time: Any,
        status: Any,
    ) -> None:
        if status:
            pass  # overruns expected during heavy LLM work — drop silently
        if indata.dtype != np.int16:
            pcm: npt.NDArray[np.int16] = (indata[:, 0] * 32767).astype(np.int16)
        else:
            pcm = indata[:, 0].astype(np.int16)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, pcm.copy())

    async def __aenter__(self) -> "RealMicStream":
        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_size,
            channels=1,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    async def frames(self) -> AsyncIterator[npt.NDArray[np.int16]]:
        while True:
            frame = await self._queue.get()
            yield frame

    def drain(self) -> int:
        """Discard any frames buffered while the orchestrator was busy with
        STT/LLM/TTS. Returns the number of frames dropped. Without this,
        the next IDLE cycle would start by chewing through seconds of stale
        audio (including TTS playback echo), which often blinds the
        wake-word detector to the user's next "Jarvis"."""
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
                dropped += 1
            except asyncio.QueueEmpty:
                break
        return dropped


class RealSpeaker:
    """Streaming speaker. Maintains a persistent sd.OutputStream and writes
    chunks sequentially. Using sd.play() per chunk would cancel any prior
    playback, so streaming TTS output (many short chunks) would be silent.

    Tracks total samples written + first-write timestamp so drain() can wait
    until the queued audio has actually finished playing — otherwise the
    orchestrator would resume mic-listening while the speaker is still
    flushing its buffer, polluting the next wake detection with TTS echo.
    """

    def __init__(self, sample_rate: int = 22050) -> None:
        self.sample_rate = sample_rate
        self._stream: Any = None  # sd.OutputStream | None — sounddevice has no stubs
        self._samples_written: int = 0
        self._first_write_at: float | None = None

    def _ensure_stream(self) -> Any:
        if self._stream is None or not self._stream.active:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )
            self._stream.start()
        return self._stream

    async def play(self, samples: npt.NDArray[Any]) -> None:
        import time as _time
        loop = asyncio.get_running_loop()
        stream = self._ensure_stream()
        buf = samples.astype(np.float32, copy=False)
        if self._first_write_at is None:
            self._first_write_at = _time.monotonic()
        self._samples_written += buf.size
        await loop.run_in_executor(None, stream.write, buf)

    async def drain(self) -> None:
        """Block until everything written so far has actually played."""
        import time as _time
        if self._first_write_at is None:
            return
        audio_secs = self._samples_written / self.sample_rate
        elapsed = _time.monotonic() - self._first_write_at
        # Small safety margin for the OS audio buffer.
        remaining = audio_secs - elapsed + 0.1
        if remaining > 0:
            await asyncio.sleep(remaining)
        self._samples_written = 0
        self._first_write_at = None

    async def aclose(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


# ── Fake (tests) ───────────────────────────────────────────────────────

class FakeMicStream:
    """Replays a fixed PCM array as a frame stream.

    The cursor position persists across multiple calls to ``frames()`` so that
    the IDLE phase and the SPEAKING barge-in phase share the same sequential
    stream — matching the behaviour of a real microphone that continuously
    produces frames regardless of which consumer is reading.
    """

    def __init__(self, pcm: npt.NDArray[np.int16], frame_size: int = 480) -> None:
        self._pcm = pcm
        self._frame_size = frame_size
        self._pos: int = 0  # sample-index cursor; shared across frames() calls

    async def frames(self) -> AsyncIterator[npt.NDArray[np.int16]]:
        n = (self._pcm.size // self._frame_size) * self._frame_size
        while self._pos < n:
            i = self._pos
            self._pos += self._frame_size
            yield self._pcm[i : i + self._frame_size].copy()
            await asyncio.sleep(0)  # let other tasks run

    def drain(self) -> int:
        """No-op for the fake stream — it has no async buffer."""
        return 0


class FakeSpeaker:
    """Collects played audio for assertions."""

    def __init__(self) -> None:
        self._chunks: list[npt.NDArray[Any]] = []

    async def play(self, samples: npt.NDArray[Any]) -> None:
        self._chunks.append(samples.copy())

    async def drain(self) -> None:
        """No-op for the fake speaker — playback is synchronous in tests."""
        return

    def collected(self) -> npt.NDArray[np.float32]:
        if not self._chunks:
            return np.empty(0, dtype=np.float32)
        result: npt.NDArray[np.float32] = np.concatenate(self._chunks)
        return result
