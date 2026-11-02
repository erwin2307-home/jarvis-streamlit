"""Instant-answer lookup for known questions.

Skips the LLM entirely when the transcript matches a pre-researched
question closely enough — no network call, no model inference, just the
already-known answer. Matching is plain keyword overlap (no embeddings,
no extra dependency): cheap enough to run on every turn.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QaEntry:
    keywords: frozenset[str]
    answer: str


# Common German stopwords — excluded from both entry keywords and the
# query. Without this, a short entry built from a generic phrase like
# "was ist formamid wofür" scores 0.5 (2 of 4 keywords: "was", "ist")
# against ANY "was ist X" question, regardless of topic — real bug
# observed in practice ("ist das gefährlich was man da benutzt" matched
# the formamide entry purely on "ist" + "gefährlich", not on formamide).
_STOPWORDS = frozenset({
    "der", "die", "das", "den", "dem", "des",
    "ein", "eine", "einen", "einem", "eines", "einer",
    "ist", "sind", "war", "waren", "wird", "werden", "wurde", "wurden",
    "und", "oder", "mit", "für", "von", "zu", "im", "am", "auf", "aus",
    "bei", "beim", "über", "um", "nach", "vor",
    "was", "wie", "wo", "wer", "wen", "wem", "wieso", "warum", "weshalb",
    "wofür", "wozu",
    "man", "du", "sie", "er", "es", "ich", "wir", "ihr", "da", "dass",
    "mir", "mich", "dir", "dich",
    "kann", "können", "muss", "müssen", "soll", "sollen",
    "braucht", "brauchen", "gibt", "benutzt", "benutzen",
    "verwendet", "verwenden",
    "nicht", "auch", "noch", "schon",
})


def _normalize_words(text: str) -> set[str]:
    words = set(re.findall(r"[a-zäöüß0-9]+", text.lower()))
    return words - _STOPWORDS


def make_entry(trigger_phrases: str, answer: str) -> QaEntry:
    """Build a QaEntry from a free-form string of representative trigger
    words (German), e.g. "beadchip illumina was ist"."""
    return QaEntry(keywords=frozenset(_normalize_words(trigger_phrases)), answer=answer)


def best_match(
    text: str,
    entries: tuple[QaEntry, ...],
    threshold: float = 0.5,
    min_overlap: int = 2,
) -> str | None:
    """Return the best-matching entry's answer, or None if nothing clears
    `threshold`. Score = (words shared with the entry) / (entry's total
    keyword count) — i.e. what fraction of an entry's trigger words
    actually showed up in what was asked.

    `min_overlap` additionally requires at least that many *absolute*
    shared words, not just a high ratio — otherwise a definition entry
    reduced to a single content keyword by stopword stripping (e.g. just
    {"formamid"}) trivially scores a perfect 1.0 on any query containing
    that one word, outranking a more specific entry that needs several
    matching keywords but only reaches 0.5."""
    words = _normalize_words(text)
    if not words:
        return None
    best_score = 0.0
    best_answer: str | None = None
    for entry in entries:
        if not entry.keywords:
            continue
        overlap = len(words & entry.keywords)
        if overlap < min_overlap:
            continue
        score = overlap / len(entry.keywords)
        if score > best_score:
            best_score = score
            best_answer = entry.answer
    return best_answer if best_score >= threshold else None
