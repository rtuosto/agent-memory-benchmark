"""Runtime helpers for the LongMemEval judge — dispatch + parser.

Byte-stability of the templates themselves is locked in
``test_judge_prompts_stable.py``; this file covers the routing logic and
the yes/no parser.
"""

from __future__ import annotations

import pytest

from agent_memory_benchmark.judge.longmemeval import (
    LME_ABSTENTION_TEMPLATE,
    LME_GENERAL_TEMPLATE,
    LME_KNOWLEDGE_UPDATE_TEMPLATE,
    LME_PREFERENCE_TEMPLATE,
    LME_TEMPORAL_TEMPLATE,
    LongMemEvalPromptError,
    is_abstention_question,
    longmemeval_anscheck_prompt,
    parse_yes_no,
)


@pytest.mark.parametrize(
    "task,expected_template",
    [
        ("single-session-user", LME_GENERAL_TEMPLATE),
        ("single-session-assistant", LME_GENERAL_TEMPLATE),
        ("multi-session", LME_GENERAL_TEMPLATE),
        ("temporal-reasoning", LME_TEMPORAL_TEMPLATE),
        ("knowledge-update", LME_KNOWLEDGE_UPDATE_TEMPLATE),
        ("single-session-preference", LME_PREFERENCE_TEMPLATE),
    ],
)
def test_prompt_dispatches_by_task(task: str, expected_template: str) -> None:
    prompt = longmemeval_anscheck_prompt(task, "Q?", "A.", "R.", abstention=False)
    assert prompt == expected_template.format("Q?", "A.", "R.")


def test_abstention_flag_overrides_task_routing() -> None:
    # Even a regular task type routes through the abstention template when
    # the question is unanswerable.
    prompt = longmemeval_anscheck_prompt("single-session-user", "Q?", "A.", "R.", abstention=True)
    assert prompt == LME_ABSTENTION_TEMPLATE.format("Q?", "A.", "R.")


def test_unsupported_task_raises_prompt_error() -> None:
    with pytest.raises(LongMemEvalPromptError, match="Unsupported"):
        longmemeval_anscheck_prompt("novel-task", "Q", "A", "R", abstention=False)


def test_is_abstention_question_detects_abs_marker() -> None:
    assert is_abstention_question("qid_abs_42")
    assert is_abstention_question("some_abs_thing")


def test_is_abstention_question_rejects_regular_ids() -> None:
    assert not is_abstention_question("qid_42")
    assert not is_abstention_question("temporal_reasoning_3")


@pytest.mark.parametrize(
    "response,expected",
    [
        ("yes", True),
        ("Yes", True),
        ("YES", True),
        ("yes.", True),
        (" yes, the model is correct", True),
        ("\nYes\n", True),
        ("no", False),
        ("No.", False),
        ("yesterday was fine", False),
        ("the answer is yes", False),  # yes not at start
        ("", False),
        ("   ", False),
    ],
)
def test_parse_yes_no_word_boundary(response: str, expected: bool) -> None:
    assert parse_yes_no(response) is expected
