"""Combines every specialized-domain QA dataset (BeadChip, NGS,
Blutanalyse, Pharmakogenomik) into one fast-path matcher and one
topic-context injector, so app.py and remote_api.py each wire a single
pair of callables instead of one per topic.

Order in _TOPIC_CONTEXTS matters: on a marker-word collision (e.g. a
query containing both "cluster" and "cluster density"), the first
matching topic in this list wins. BeadChip is checked last since its
own markers include very generic bare words ("illumina", "genotyp")
that would otherwise shadow the more specific new topics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from jarvis.qa_beadchip import BEADCHIP_QA, beadchip_topic_context, reagent_answer
from jarvis.qa_blood_analysis import BLOOD_ANALYSIS_QA, blood_analysis_topic_context
from jarvis.qa_lookup import QaEntry, best_match
from jarvis.qa_ngs import NGS_QA, ngs_topic_context
from jarvis.qa_pharmacogenomics import PHARMACOGENOMICS_QA, pharmacogenomics_topic_context
from jarvis.unanswered import DEFAULT_LOG_PATH, log_unanswered

ALL_QA: tuple[QaEntry, ...] = (
    NGS_QA + PHARMACOGENOMICS_QA + BLOOD_ANALYSIS_QA + BEADCHIP_QA
)

_TOPIC_CONTEXTS: tuple[tuple[str, Callable[[str], str | None]], ...] = (
    ("ngs", ngs_topic_context),
    ("pharmakogenomik", pharmacogenomics_topic_context),
    ("blutanalyse", blood_analysis_topic_context),
    ("beadchip", beadchip_topic_context),
)


def fastpath_answer(text: str) -> str | None:
    """Exact-match answer across every topic's curated QA set, plus
    BeadChip's bare-reagent-abbreviation lookup — a hit skips the LLM
    entirely, same reasoning as qa_beadchip.reagent_answer."""
    return best_match(text, ALL_QA) or reagent_answer(text)


def topic_context(text: str, log_path: Path = DEFAULT_LOG_PATH) -> str | None:
    """First matching topic's full reference context, for RAG-style
    injection into the LLM call when fastpath_answer found nothing. Also
    logs the question via log_unanswered — the topic matched broadly but
    no curated entry fired, which is exactly the gap worth reviewing at
    the next content update."""
    for name, fn in _TOPIC_CONTEXTS:
        ctx = fn(text)
        if ctx is not None:
            log_unanswered(text, name, log_path)
            return ctx
    return None
