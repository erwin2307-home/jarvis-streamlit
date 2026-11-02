"""TTS tests using fake synthesizer."""
import math

import numpy as np

from jarvis.tts import FakeTts


async def test_fake_tts_yields_chunks():
    tts = FakeTts(samples_per_chunk=4)
    chunks = []
    async for chunk in tts.synthesize("hello world"):
        chunks.append(chunk)
    # FakeTts emits ceil(SAMPLES_PER_WORD / samples_per_chunk) chunks per word.
    chunks_per_word = max(1, math.ceil(FakeTts._SAMPLES_PER_WORD / 4))
    assert len(chunks) == 2 * chunks_per_word
    assert chunks[0].shape == (4,)
    assert chunks[0].dtype == np.float32


async def test_fake_tts_empty_text_no_chunks():
    tts = FakeTts(samples_per_chunk=4)
    out = [c async for c in tts.synthesize("")]
    assert out == []
