"""Wake-word detection. Real (OpenWakeWord) + Fake (test)."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt


class WakeWord(Protocol):
    async def detect(self, frame_int16: npt.NDArray[np.int16]) -> bool: ...


class RealWakeWord:
    """OpenWakeWord-based detector."""

    def __init__(self, model_path: Path, threshold: float = 0.5) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Wake-word model not found: {model_path}")
        from openwakeword.model import Model  # type: ignore[import-untyped]

        self._model = Model(wakeword_models=[str(model_path)])
        self._threshold = threshold
        # OpenWakeWord exposes a single trigger key per loaded model
        self._key: str = list(self._model.models.keys())[0]

    async def detect(self, frame_int16: npt.NDArray[np.int16]) -> bool:
        # OpenWakeWord expects 16-bit mono PCM. Internally buffers.
        scores: dict[str, float] = self._model.predict(frame_int16)
        return bool(scores[self._key] >= self._threshold)


class FakeWakeWord:
    """Fires on a frame whose first sample equals `marker_value`.

    `threshold` is accepted for interface compatibility with WakeWord but is
    not used — detection is purely marker-based.
    """

    def __init__(self, threshold: float = 0.5, marker_value: int = 12345) -> None:
        self._threshold = threshold  # unused; kept for WakeWord interface compatibility
        self._marker = marker_value

    async def detect(self, frame_int16: npt.NDArray[np.int16]) -> bool:
        if frame_int16.size == 0:
            return False
        return int(frame_int16[0]) == self._marker
