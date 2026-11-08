"""Tests for the NGS topic detector + RAG context injection."""
from jarvis.qa_lookup import best_match
from jarvis.qa_ngs import NGS_QA, ngs_topic_context


def test_detects_topic_from_direct_mention() -> None:
    assert ngs_topic_context("was ist ngs") is not None


def test_detects_topic_from_loose_natural_phrasing() -> None:
    for q in (
        "wie funktioniert next generation sequencing",
        "was ist eine flow cell",
        "erklär mir fastq dateien",
        "wie liest man einen phred score",
    ):
        assert ngs_topic_context(q) is not None, f"no context for: {q!r}"


def test_detects_troubleshooting_questions() -> None:
    for q in (
        "meine cluster density ist zu niedrig was mache ich",
        "was ist index hopping und wie vermeide ich es",
        "ich habe adapter dimere in meiner library",
    ):
        assert ngs_topic_context(q) is not None, f"no context for: {q!r}"


def test_unrelated_question_gets_no_context() -> None:
    assert ngs_topic_context("wie spät ist es") is None
    assert ngs_topic_context("danke, bis später") is None


def test_context_includes_key_facts() -> None:
    ctx = ngs_topic_context("was ist ngs")
    assert ctx is not None
    for must_contain in ("Flow Cell", "Coverage", "Q30", "WGS", "WES"):
        assert must_contain in ctx, f"missing {must_contain!r} in context"


def test_exact_match_entries_score_above_threshold() -> None:
    assert best_match("was ist ngs next generation sequencing", NGS_QA) is not None
    assert best_match("was ist eine flow cell flowcell", NGS_QA) is not None
