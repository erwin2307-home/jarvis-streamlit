"""Dedicated per-reagent answer lookup — catches bare abbreviation
queries ("ra1", "was ist ra1") that qa_lookup.best_match()'s
min_overlap=2 threshold would otherwise reject."""
import pytest

from jarvis.qa_beadchip import _REAGENT_ANSWERS, reagent_answer


def test_bare_abbreviation_matches():
    assert reagent_answer("ra1") is not None


def test_question_phrasing_matches():
    assert reagent_answer("was ist ra1") == reagent_answer("ra1")


def test_ra1_answer_leads_with_sample_loading_usage():
    answer = reagent_answer("ra1")
    assert answer.startswith("RA1 wird gebraucht, wenn die Probe auf den BeadChip aufgetragen wird")


def test_unrelated_text_returns_none():
    assert reagent_answer("hallo wie geht es dir") is None
    assert reagent_answer("baue einen taschenrechner") is None


def test_case_insensitive():
    assert reagent_answer("RA1") == reagent_answer("ra1")
    assert reagent_answer("Ra1") == reagent_answer("ra1")


@pytest.mark.parametrize("abbrev", list(_REAGENT_ANSWERS))
def test_every_known_reagent_resolves(abbrev):
    answer = reagent_answer(abbrev)
    assert answer == _REAGENT_ANSWERS[abbrev]
    assert len(answer) > 20


def test_picks_first_mentioned_reagent_when_multiple_appear():
    # Word-boundary match, first hit wins — good enough for the realistic
    # case of one reagent per question.
    result = reagent_answer("ra1 und xc3")
    assert result == _REAGENT_ANSWERS["ra1"]
