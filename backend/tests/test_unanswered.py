"""Tests for the unanswered-domain-question logger."""
from jarvis.unanswered import log_unanswered, read_unanswered


def test_log_then_read_back(tmp_path) -> None:
    path = tmp_path / "unanswered.jsonl"
    log_unanswered("was ist crispr", "ngs", path)
    entries = read_unanswered(path)
    assert len(entries) == 1
    assert entries[0]["question"] == "was ist crispr"
    assert entries[0]["topic"] == "ngs"
    assert "timestamp" in entries[0]


def test_multiple_logs_append_as_separate_lines(tmp_path) -> None:
    path = tmp_path / "unanswered.jsonl"
    log_unanswered("frage eins", "ngs", path)
    log_unanswered("frage zwei", "blutanalyse", path)
    entries = read_unanswered(path)
    assert [e["question"] for e in entries] == ["frage eins", "frage zwei"]


def test_read_unanswered_on_missing_file_returns_empty_list(tmp_path) -> None:
    assert read_unanswered(tmp_path / "does_not_exist.jsonl") == []


def test_log_unanswered_creates_parent_dir(tmp_path) -> None:
    path = tmp_path / "nested" / "dir" / "unanswered.jsonl"
    log_unanswered("frage", "pharmakogenomik", path)
    assert path.exists()
