"""Tests for the LOCOMO judge template, parser, and majority vote."""

from __future__ import annotations

from agent_memory_benchmark.judge.locomo import (
    LOCOMO_JUDGE_USER_TEMPLATE,
    locomo_judge_prompt,
    majority_vote,
    parse_locomo_correct,
)


def test_locomo_judge_prompt_fills_placeholders() -> None:
    filled = locomo_judge_prompt(
        question="Q?",
        gold_answer="shell",
        generated_answer="the shell necklace",
    )
    assert "Question: Q?" in filled
    assert "Gold answer: shell" in filled
    assert "Generated answer: the shell necklace" in filled


def test_locomo_template_preserves_trailing_instruction() -> None:
    """The parse path expects ``{"label": ...}`` JSON — make sure the
    template text still instructs the judge to emit it."""

    assert "label" in LOCOMO_JUDGE_USER_TEMPLATE


def test_parse_correct_from_clean_json() -> None:
    assert parse_locomo_correct('{"label": "CORRECT"}') is True


def test_parse_wrong_from_clean_json() -> None:
    assert parse_locomo_correct('{"label": "WRONG"}') is False


def test_parse_json_label_embedded_in_prose() -> None:
    text = 'The generated answer matches the gold.\n{"label": "CORRECT"}\nEnd of response.'
    assert parse_locomo_correct(text) is True


def test_parse_falls_back_to_substring_correct() -> None:
    """When the judge doesn't emit valid JSON, look for CORRECT/WRONG."""

    assert parse_locomo_correct("I think this is CORRECT.") is True


def test_parse_falls_back_to_substring_wrong() -> None:
    assert parse_locomo_correct("This is WRONG because ...") is False


def test_parse_both_labels_present_returns_wrong() -> None:
    """Prompt violation — judge returned both labels. Treat as WRONG."""

    assert parse_locomo_correct("either CORRECT or WRONG") is False


def test_parse_neither_label_returns_wrong() -> None:
    """Judge failed to output a verdict — default to WRONG."""

    assert parse_locomo_correct("I am not sure.") is False


def test_parse_lowercase_is_not_accepted() -> None:
    """Matches predecessor: label comparison is upper-case only."""

    # "correct" uppercased becomes "CORRECT" so the substring check fires.
    # The test lives to pin the exact behavior so refactors don't drift.
    assert parse_locomo_correct("correct") is True


def test_parse_json_label_lowercase_value_still_correct() -> None:
    """Lowercased label value still contains ``CORRECT`` after .upper()."""

    assert parse_locomo_correct('{"label": "correct"}') is True


def test_majority_vote_simple_majority_true() -> None:
    assert majority_vote([True, True, False]) is True


def test_majority_vote_simple_majority_false() -> None:
    assert majority_vote([False, False, True]) is False


def test_majority_vote_tie_is_false() -> None:
    """Even-numbered runs with a tie — treat as WRONG (no strict majority)."""

    assert majority_vote([True, False]) is False
    assert majority_vote([True, True, False, False]) is False


def test_majority_vote_empty_is_false() -> None:
    assert majority_vote([]) is False


def test_majority_vote_all_true() -> None:
    assert majority_vote([True] * 10) is True


def test_majority_vote_all_false() -> None:
    assert majority_vote([False] * 10) is False
