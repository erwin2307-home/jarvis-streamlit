"""Filler phrase picker tests."""
import random

from jarvis.filler import FILLER_PHRASES, random_filler


def test_phrases_not_empty() -> None:
    assert len(FILLER_PHRASES) >= 10
    for p in FILLER_PHRASES:
        assert p.strip(), "no empty filler phrases allowed"


def test_phrases_short() -> None:
    """Filler must be brief — max 2 sentences each per spec."""
    for p in FILLER_PHRASES:
        # Count terminal sentence punctuation.
        n = sum(p.count(c) for c in ".!?—")
        # Some Shakespeare lines use em-dashes mid-sentence; cap on real
        # sentence terminators is what matters.
        terminators = p.count(".") + p.count("!") + p.count("?")
        assert terminators <= 2, f"too many sentences in: {p!r}"


def test_random_filler_returns_phrase_from_list() -> None:
    chosen = random_filler()
    assert chosen in FILLER_PHRASES


def test_random_filler_deterministic_with_seed() -> None:
    rng = random.Random(42)
    a = random_filler(rng)
    rng = random.Random(42)
    b = random_filler(rng)
    assert a == b
