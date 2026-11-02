"""Voice Activity Detection — Silero VAD wrapper + speech-end detector."""
from __future__ import annotations

from typing import Any, Iterable, Protocol

import numpy as np
import numpy.typing as npt


class Vad(Protocol):
    async def is_speech(self, frame_int16: npt.NDArray[np.int16]) -> bool: ...


class RealVad:
    """Silero-VAD wrapper."""

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000) -> None:
        import torch
        from silero_vad import load_silero_vad  # type: ignore[import-untyped]

        self._model: Any = load_silero_vad()
        self._threshold = threshold
        self._sr = sample_rate
        self._torch = torch

    async def is_speech(self, frame_int16: npt.NDArray[np.int16]) -> bool:
        tensor = self._torch.from_numpy(frame_int16.astype(np.float32) / 32768.0)
        score = float(self._model(tensor, self._sr).item())
        return score >= self._threshold


class FakeVad:
    """Returns scripted sequence."""

    def __init__(self, speech_mask: Iterable[bool]) -> None:
        self._mask: list[bool] = list(speech_mask)
        self._i = 0

    async def is_speech(self, frame_int16: npt.NDArray[np.int16]) -> bool:
        if self._i >= len(self._mask):
            return False
        v = self._mask[self._i]
        self._i += 1
        return v


class SpeechEndDetector:
    """
    Tracks transitions: emits ``'speech_start'`` on first speech frame,
    ``'speech_end'`` after ``silence_frames`` consecutive non-speech post-speech.
    """

    def __init__(self, silence_frames: int = 27) -> None:
        # default 27 ≈ 800 ms at 30 ms frames
        self._silence_target = silence_frames
        self._silence_count = 0
        self._in_speech = False

    def update(self, is_speech: bool) -> str | None:
        if is_speech:
            self._silence_count = 0
            if not self._in_speech:
                self._in_speech = True
                return "speech_start"
            return None
        if self._in_speech:
            self._silence_count += 1
            if self._silence_count >= self._silence_target:
                self._in_speech = False
                self._silence_count = 0
                return "speech_end"
        return None
