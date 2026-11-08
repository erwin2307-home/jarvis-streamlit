"""Tests for the pharmacogenomics topic detector + RAG context injection."""
from jarvis.qa_lookup import best_match
from jarvis.qa_pharmacogenomics import PHARMACOGENOMICS_QA, pharmacogenomics_topic_context


def test_detects_topic_from_direct_mention() -> None:
    assert pharmacogenomics_topic_context("was ist pharmakogenomik") is not None


def test_detects_topic_from_gene_drug_pair_mentions() -> None:
    for q in (
        "was hat cyp2d6 mit codein zu tun",
        "warum testet man auf hla-b 5701 vor abacavir",
        "was macht vkorc1 bei warfarin",
        "was ist ein poor metabolizer",
    ):
        assert pharmacogenomics_topic_context(q) is not None, f"no context for: {q!r}"


def test_unrelated_question_gets_no_context() -> None:
    assert pharmacogenomics_topic_context("wie spät ist es") is None
    assert pharmacogenomics_topic_context("danke, bis später") is None


def test_context_includes_key_gene_drug_pairs() -> None:
    ctx = pharmacogenomics_topic_context("was ist pharmakogenomik")
    assert ctx is not None
    for must_contain in ("CYP2D6", "CYP2C19", "VKORC1", "CPIC", "HLA-B"):
        assert must_contain in ctx, f"missing {must_contain!r} in context"


def test_context_does_not_give_patient_specific_dosing_instruction() -> None:
    # Reference knowledge for a lab assistant, not a prescribing tool —
    # guards against a future edit sliding into concrete per-patient
    # dosing numbers.
    ctx = pharmacogenomics_topic_context("was ist pharmakogenomik")
    assert ctx is not None
    assert "KEINE konkrete Dosierungsempfehlung" in ctx


def test_exact_match_entries_score_above_threshold() -> None:
    assert best_match("was ist pharmakogenomik pharmakogenetik", PHARMACOGENOMICS_QA) is not None
    assert best_match("cyp2d6 codein tramadol metabolisierung", PHARMACOGENOMICS_QA) is not None
