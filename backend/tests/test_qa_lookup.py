"""Tests for the instant-answer QA lookup (keyword-overlap matching, no LLM)."""
from jarvis.qa_lookup import QaEntry, best_match, make_entry


def test_exact_keyword_overlap_matches() -> None:
    entries = (make_entry("beadchip illumina was ist", "Ein BeadChip ist ein Microarray."),)
    assert best_match("was ist ein illumina beadchip", entries) == "Ein BeadChip ist ein Microarray."


def test_unrelated_query_returns_none() -> None:
    entries = (make_entry("beadchip illumina was ist", "Ein BeadChip ist ein Microarray."),)
    assert best_match("wie spät ist es", entries) is None


def test_empty_text_returns_none() -> None:
    entries = (make_entry("beadchip illumina", "Antwort"),)
    assert best_match("", entries) is None


def test_low_overlap_below_threshold_returns_none() -> None:
    entries = (make_entry("beadchip illumina formamid proteinase chemikalien", "Antwort"),)
    # Only one of five keywords present -> 0.2 overlap, below default 0.5 threshold.
    assert best_match("beadchip", entries) is None


def test_picks_highest_scoring_entry_when_multiple_match() -> None:
    entries = (
        make_entry("beadchip chip microarray sonde", "Allgemeine BeadChip-Antwort."),
        make_entry("formamid gefaehrlich beadchip", "Formamid-Antwort."),
    )
    # "formamid ist beadchip gefaehrlich" overlaps 3/3 with the second entry,
    # only 1/4 with the first -> second should win.
    assert best_match("formamid ist beadchip gefaehrlich", entries) == "Formamid-Antwort."


def test_make_entry_normalizes_german_umlauts_and_case() -> None:
    entry = make_entry("Gefährlich FORMAMID", "x")
    assert entry.keywords == frozenset({"gefährlich", "formamid"})


def test_qa_entry_is_frozen_dataclass() -> None:
    entry = QaEntry(keywords=frozenset({"x"}), answer="y")
    assert entry.answer == "y"


def test_beadchip_dataset_matches_representative_questions() -> None:
    from jarvis.qa_beadchip import BEADCHIP_QA

    assert best_match("was ist ein illumina beadchip", BEADCHIP_QA) is not None
    assert best_match("ist formamid gefährlich", BEADCHIP_QA) is not None
    assert best_match("was ist proteinase k", BEADCHIP_QA) is not None
    assert best_match("wie spät ist es", BEADCHIP_QA) is None


def test_beadchip_reagents_answer_names_proteinase_k() -> None:
    from jarvis.qa_beadchip import BEADCHIP_QA

    answer = best_match("welche chemikalien reagenzien werden verwendet", BEADCHIP_QA)
    assert answer is not None
    assert "Proteinase K" in answer


def test_beadchip_reagents_question_also_matches_misheard_proteinkinase() -> None:
    # STT can mishear "Proteinase K" as "Proteinkinase" (a real, unrelated
    # enzyme name) — the question must still resolve to the right answer.
    from jarvis.qa_beadchip import BEADCHIP_QA

    answer = best_match("welche reagenzien chemikalien proteinkinase", BEADCHIP_QA)
    assert answer is not None
    assert "Proteinase K" in answer


def test_stopwords_alone_never_cause_a_match() -> None:
    """Reproduces the real bug: a short entry whose keywords are mostly
    common German stopwords ("was", "ist") must not match a vague,
    topically-unrelated question just because both happen to say "was
    ist...". Without stopword filtering this scored 0.5 (2 of 4 keywords)
    on pure coincidence."""
    entries = (make_entry("was ist formamid wofür", "Formamid-Antwort."),)
    assert best_match("was ist ra1 ein reagenz", entries) is None


def test_stopwords_filtered_from_both_entry_and_query() -> None:
    entry = make_entry("was ist formamid wofür", "x")
    # "formamid" is the only real content word left after stripping
    # "was"/"ist"/"wofür".
    assert entry.keywords == frozenset({"formamid"})


def test_single_keyword_entry_does_not_outrank_a_more_specific_entry() -> None:
    """Real bug: after stopword stripping, a definition entry can be left
    with just one content keyword (e.g. {"formamid"}), which then scores
    a trivial 1.0 on any query containing that word — even beating a more
    specific, more relevant entry that needs two matching keywords but
    only reaches 0.5. Absolute overlap count must gate this, not just the
    ratio."""
    entries = (
        make_entry("formamid", "Definitions-Antwort (zu generisch, 1 Keyword)."),
        make_entry("formamid gefährlich giftig sicherheit", "Sicherheits-Antwort."),
    )
    assert best_match("ist formamid gefährlich", entries) == "Sicherheits-Antwort."


def test_vague_safety_question_does_not_default_to_first_chemical_entry() -> None:
    """The real-world failure: a vague safety question with no specific
    chemical named must not silently default to whichever chemical entry
    happens to come first in the dataset."""
    from jarvis.qa_beadchip import BEADCHIP_QA

    assert best_match("ist das gefährlich was man da benutzt", BEADCHIP_QA) is None


def test_beadchip_full_workflow_lists_every_major_step_for_24_samples() -> None:
    from jarvis.qa_beadchip import BEADCHIP_QA

    answer = best_match(
        "zähle alle schritte auf von der speichelprobe bis zur iscan auswertung "
        "für den chip mit 24 proben",
        BEADCHIP_QA,
    )
    assert answer is not None
    for must_contain in (
        "Proteinase K",
        "Amplifikation",
        "Fragmentierung",
        "Präzipitation",
        "Resuspension",
        "Hybridisierung",
        "Staining",
        "iScan",
        "24",
    ):
        assert must_contain in answer, f"missing {must_contain!r} in full workflow answer"
