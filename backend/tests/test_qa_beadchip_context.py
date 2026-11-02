"""Tests for the broad BeadChip topic detector + RAG context injection.

The exact-match QA pairs (test_qa_lookup.py) only catch phrasings close to
the hand-written triggers. This is the safety net: ANY loosely
BeadChip-related question gets the full reference knowledge injected as
context for the LLM call, instead of falling through to a model that
knows nothing about the topic and hallucinates.
"""
from jarvis.qa_beadchip import beadchip_topic_context


def test_detects_topic_from_direct_mention() -> None:
    assert beadchip_topic_context("was ist ein illumina beadchip") is not None


def test_detects_topic_from_loose_natural_phrasing() -> None:
    # None of these hit the exact-match QA triggers, but all are clearly
    # about the same topic and must still get context injected.
    for q in (
        "erzähl mir alles über den beadchip",
        "was weißt du über illumina",
        "kannst du mir was zum beadchip sagen",
        "wie läuft der infinium workflow ab",
        "was ist proteinase k und wofür braucht man es",
    ):
        assert beadchip_topic_context(q) is not None, f"no context for: {q!r}"


def test_detects_topic_from_bare_reagent_abbreviation() -> None:
    # Real gap found in practice: "was ist ra1" scored too low for the
    # QA_PAIRS instant-match (short query, min_overlap gate) but also
    # didn't trip the topic markers, so it got NEITHER an instant answer
    # NOR RAG context — total silence, worse than before the fix.
    for q in ("was ist ra1", "ist ma1 gefährlich", "was macht xc4", "wofür ist pb1"):
        assert beadchip_topic_context(q) is not None, f"no context for: {q!r}"


def test_detects_topic_from_observed_stt_mishearings_of_beadchip() -> None:
    # German Whisper, even with vocabulary biasing, still mishears
    # "BeadChip" phonetically — observed live as "Beatship" and
    # "Beatschip". These must still trigger context, since a question
    # using only "BeadChip" (no "Illumina") would otherwise get nothing.
    for q in ("was ist ein beatship", "wie funktioniert der beatschip"):
        assert beadchip_topic_context(q) is not None, f"no context for: {q!r}"


def test_detects_troubleshooting_questions() -> None:
    for q in (
        "was ist eine gute call rate",
        "was bedeutet gentrain score",
        "mein illumina chip hat einen fehler was mache ich",
    ):
        assert beadchip_topic_context(q) is not None, f"no context for: {q!r}"


def test_context_includes_troubleshooting_facts() -> None:
    ctx = beadchip_topic_context("was ist ein beadchip")
    assert ctx is not None
    assert "95" in ctx and "98" in ctx  # call rate standard range
    assert "GenTrain" in ctx


def test_unrelated_question_gets_no_context() -> None:
    assert beadchip_topic_context("wie spät ist es") is None
    assert beadchip_topic_context("danke, bis später") is None


def test_context_mentions_every_major_reagent_and_step() -> None:
    ctx = beadchip_topic_context("was ist ein beadchip")
    assert ctx is not None
    for must_contain in (
        "Proteinase K",
        "Amplifikation",
        "Fragmentierung",
        "Präzipitation",
        "Resuspension",
        "Hybridisierung",
        "Staining",
        "iScan",
        "Formamid",
        "RA1",
    ):
        assert must_contain in ctx, f"missing {must_contain!r} in context"
