"""Tests for the blood-analysis topic detector + RAG context injection."""
from jarvis.qa_blood_analysis import BLOOD_ANALYSIS_QA, blood_analysis_topic_context
from jarvis.qa_lookup import best_match


def test_detects_topic_from_direct_mention() -> None:
    assert blood_analysis_topic_context("was ist ein blutbild") is not None


def test_detects_topic_from_loose_natural_phrasing() -> None:
    for q in (
        "was bedeutet ein niedriger hämoglobin wert",
        "was ist der unterschied zwischen serum und plasma",
        "was macht man bei einer hämolytischen probe",
        "was sagt der mcv wert aus",
    ):
        assert blood_analysis_topic_context(q) is not None, f"no context for: {q!r}"


def test_unrelated_question_gets_no_context() -> None:
    assert blood_analysis_topic_context("wie spät ist es") is None
    assert blood_analysis_topic_context("danke, bis später") is None


def test_context_includes_key_facts() -> None:
    ctx = blood_analysis_topic_context("was ist ein blutbild")
    assert ctx is not None
    for must_contain in ("Hämoglobin", "Hämatokrit", "Leukozyten", "INR", "Hämolyse"):
        assert must_contain in ctx, f"missing {must_contain!r} in context"


def test_context_does_not_give_diagnosis_for_a_person() -> None:
    ctx = blood_analysis_topic_context("was ist ein blutbild")
    assert ctx is not None
    assert "KEINE Diagnose" in ctx


def test_exact_match_entries_score_above_threshold() -> None:
    assert best_match("was ist ein blutbild cbc complete blood count", BLOOD_ANALYSIS_QA) is not None
    assert best_match("was ist hämoglobin hamoglobin referenzwert", BLOOD_ANALYSIS_QA) is not None
