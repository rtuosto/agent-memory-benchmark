"""Tests for benchmark-owned dataclasses."""

from __future__ import annotations

import pytest

from agent_memory_benchmark import (
    AnswerResult,
    BenchmarkCase,
    QAItem,
    RetrievedUnit,
    Session,
    Turn,
)


def test_turn_is_frozen() -> None:
    turn = Turn(turn_id="t1", speaker="user", text="hi")
    with pytest.raises((AttributeError, TypeError)):
        turn.text = "mutated"  # type: ignore[misc]


def test_session_is_frozen() -> None:
    session = Session(session_index=1, turns=(Turn(turn_id="t1", speaker="a", text="x"),))
    assert session.session_index == 1
    assert len(session.turns) == 1
    with pytest.raises((AttributeError, TypeError)):
        session.session_index = 2  # type: ignore[misc]


def test_qa_item_evidence_defaults_empty() -> None:
    qa = QAItem(question_id="q1", question="Q?", gold="A", question_type="x")
    assert qa.evidence_turn_ids == ()
    assert qa.metadata == {}
    assert qa.category is None


def test_qa_item_preserves_evidence_order() -> None:
    qa = QAItem(
        question_id="q2",
        question="Q?",
        gold="A",
        question_type="x",
        evidence_turn_ids=("t3", "t1", "t2"),
    )
    assert qa.evidence_turn_ids == ("t3", "t1", "t2")


def test_benchmark_case_dataset_typed() -> None:
    case = BenchmarkCase(
        case_id="c1",
        sessions=(),
        qa=(),
        dataset="longmemeval",
    )
    assert case.dataset == "longmemeval"


def test_retrieved_unit_defaults() -> None:
    unit = RetrievedUnit(unit_id="u1", text="hello")
    assert unit.score is None
    assert unit.source_turn_ids == ()


def test_answer_result_mutable_fields() -> None:
    result = AnswerResult(answer="A", retrieval_time_ms=1.0, generation_time_ms=2.0)
    assert result.units_retrieved == 0
    assert result.retrieved == ()
    result.units_retrieved = 3  # AnswerResult is not frozen
    assert result.units_retrieved == 3
