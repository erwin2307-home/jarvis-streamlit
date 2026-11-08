"""Tests for the combined QA fast-path/context module that app.py and
remote_api.py both wire in (see qa_topics.py)."""
from jarvis.unanswered import read_unanswered
from jarvis.qa_topics import fastpath_answer, topic_context


def test_fastpath_answer_works_across_every_topic() -> None:
    assert fastpath_answer("was ist ngs next generation sequencing") is not None
    assert fastpath_answer("was ist pharmakogenomik pharmakogenetik") is not None
    assert fastpath_answer("was ist ein blutbild cbc complete blood count") is not None
    assert fastpath_answer("was ist ein illumina beadchip chip") is not None


def test_fastpath_answer_still_covers_bare_reagent_abbreviation() -> None:
    # BeadChip's reagent_answer() bypass must survive being folded into
    # the combined module.
    assert fastpath_answer("was ist ra1") is not None


def test_fastpath_answer_none_for_unrelated_question() -> None:
    assert fastpath_answer("wie spät ist es") is None


def test_topic_context_returns_context_for_every_topic() -> None:
    assert topic_context("next generation sequencing erklärung") is not None
    assert topic_context("cyp2d6 metabolisierer") is not None
    assert topic_context("hämoglobin referenzwert") is not None
    assert topic_context("illumina beadchip workflow") is not None


def test_topic_context_none_for_unrelated_question() -> None:
    assert topic_context("wie spät ist es") is None


def test_topic_context_logs_question_with_correct_topic_name(tmp_path) -> None:
    log_path = tmp_path / "unanswered.jsonl"
    topic_context("erzähl mir mehr über next generation sequencing", log_path)
    entries = read_unanswered(log_path)
    assert len(entries) == 1
    assert entries[0]["topic"] == "ngs"


def test_topic_context_does_not_log_when_no_topic_matches(tmp_path) -> None:
    log_path = tmp_path / "unanswered.jsonl"
    topic_context("wie spät ist es", log_path)
    assert read_unanswered(log_path) == []
