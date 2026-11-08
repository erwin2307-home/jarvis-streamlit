"""Logs domain questions that matched a specialized topic (NGS,
Blutanalyse, Pharmakogenomik, BeadChip) by broad marker word but had no
exact QA entry to answer them — see qa_topics.topic_context, the only
caller. The LLM still answers the turn via RAG context injection as
before; this just records that a specific curated answer is missing, so
these can be reviewed and turned into real entries (qa_ngs.py etc.) at
the next content update instead of silently always falling back to the
LLM for the same gap.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_PATH = Path.home() / "Desktop" / "Jarvis_Unanswered_Fragen.jsonl"


def log_unanswered(question: str, topic: str, log_path: Path = DEFAULT_LOG_PATH) -> None:
    """Appends one JSON line: {timestamp, topic, question}. Append-only
    (never overwrites), so concurrent turns can't clobber each other's
    entries — each write is one line, one open/close."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "topic": topic,
        "question": question,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_unanswered(log_path: Path = DEFAULT_LOG_PATH) -> list[dict[str, str]]:
    """Reads back all logged entries, oldest first. Empty list if the
    log doesn't exist yet. Used when reviewing what to add at the next
    content update."""
    if not log_path.exists():
        return []
    entries: list[dict[str, str]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries
