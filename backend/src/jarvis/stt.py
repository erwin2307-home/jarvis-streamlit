"""Speech-to-text — faster-whisper wrapper + fake."""
from __future__ import annotations

import asyncio
from typing import Protocol

import numpy as np
import numpy.typing as npt


class Stt(Protocol):
    async def transcribe(self, pcm_int16: npt.NDArray[np.int16]) -> str: ...


def strip_lone_surrogates(text: str) -> str:
    """Remove lone UTF-16 surrogate codepoints that faster-whisper can
    occasionally emit from a malformed decode (observed on Windows, on
    ambiguous/noisy audio). These have no valid UTF-8 encoding and crash
    later JSON/HTTP serialization when the transcript is sent to the LLM."""
    return text.encode("utf-8", errors="ignore").decode("utf-8")


class RealStt:
    """faster-whisper wrapper. Runs synchronously in executor."""

    def __init__(
        self,
        model_name: str = "base",
        compute_type: str = "int8",
        language: str = "de",
        initial_prompt: str | None = None,
    ) -> None:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        self._model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
        self._language = language
        # Biases decoding toward specific vocabulary/spelling without
        # forcing it — lets Whisper correctly recognize domain jargon
        # (brand names, loanwords) it would otherwise mangle into
        # phonetically-similar German nonsense.
        self._initial_prompt = initial_prompt

    async def transcribe(self, pcm_int16: npt.NDArray[np.int16]) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, pcm_int16)

    def _transcribe_sync(self, pcm_int16: npt.NDArray[np.int16]) -> str:
        audio_f32 = pcm_int16.astype(np.float32) / 32768.0
        segments, _info = self._model.transcribe(
            audio_f32,
            language=self._language,
            vad_filter=False,
            beam_size=1,
            initial_prompt=self._initial_prompt,
        )
        text = "".join(seg.text for seg in segments).strip()
        return strip_lone_surrogates(text)


class FakeStt:
    """Returns a fixed string regardless of input."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def transcribe(self, pcm_int16: npt.NDArray[np.int16]) -> str:
        return self._text
