"""Tests for the LOCOMO loader — conversation conversion, filters, descriptor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from agent_memory_benchmark.datasets import (
    DatasetUnavailableError,
    LocomoDataset,
    load_dataset,
)
from agent_memory_benchmark.datasets.locomo import (
    EXCLUDED_CATEGORY,
    _conversation_to_case,
    _parse_qa,
    _parse_sessions,
)
from agent_memory_benchmark.types import BenchmarkCase


def _turn(dia_id: str, speaker: str = "alice", text: str = "hi") -> dict[str, Any]:
    return {"dia_id": dia_id, "speaker": speaker, "text": text}


def _conversation(
    *,
    conversation_id: str = "conv_a",
    sessions: dict[int, list[dict[str, Any]]] | None = None,
    qa: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    sessions = sessions or {
        1: [_turn("D1:1", "alice", "hello"), _turn("D1:2", "bob", "hi")],
        2: [_turn("D2:1", "alice", "later topic")],
    }
    inner: dict[str, Any] = {"speaker_a": "alice", "speaker_b": "bob"}
    for idx, turns in sessions.items():
        inner[f"session_{idx}"] = turns
        inner[f"session_{idx}_date_time"] = f"2024-01-{idx:02d}"
    return {
        "conversation_id": conversation_id,
        "conversation": inner,
        "qa": qa
        or [
            {
                "question": "What did alice say first?",
                "answer": "hello",
                "category": 1,
                "evidence": ["D1:1"],
            }
        ],
    }


def _write_fixture(path: Path, conversations: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(conversations), encoding="utf-8")


def test_conversation_to_case_builds_sessions_and_qa() -> None:
    case = _conversation_to_case(_conversation(), index=0)
    assert isinstance(case, BenchmarkCase)
    assert case.case_id == "conv_a"
    assert case.dataset == "locomo"
    assert len(case.sessions) == 2

    s1, s2 = case.sessions
    assert s1.session_index == 1
    assert s1.session_id == "session_1"
    assert s1.session_time == "2024-01-01"
    assert [t.turn_id for t in s1.turns] == ["D1:1", "D1:2"]
    assert [t.speaker for t in s1.turns] == ["alice", "bob"]
    assert s2.session_index == 2
    assert [t.turn_id for t in s2.turns] == ["D2:1"]

    assert len(case.qa) == 1
    qa = case.qa[0]
    assert qa.question_id == "qa_0"
    assert qa.category == 1
    assert qa.question_type == "cat_1"
    assert qa.gold == "hello"
    assert qa.evidence_turn_ids == ("D1:1",)


def test_category_5_questions_are_filtered() -> None:
    conv = _conversation(
        qa=[
            {"question": "q0", "answer": "a0", "category": 1, "evidence": []},
            {"question": "q1", "answer": "a1", "category": EXCLUDED_CATEGORY, "evidence": []},
            {"question": "q2", "answer": "a2", "category": 3, "evidence": []},
        ]
    )
    case = _conversation_to_case(conv, index=0)
    cats = [q.category for q in case.qa]
    assert cats == [1, 3]


def test_gold_falls_back_to_adversarial_answer() -> None:
    conv = _conversation(
        qa=[
            {
                "question": "unknowable",
                "answer": None,
                "adversarial_answer": "no-op rebuttal",
                "category": 2,
                "evidence": [],
            }
        ]
    )
    case = _conversation_to_case(conv, index=0)
    assert case.qa[0].gold == "no-op rebuttal"


def test_integer_answers_coerce_to_string() -> None:
    conv = _conversation(qa=[{"question": "q", "answer": 42, "category": 1, "evidence": []}])
    case = _conversation_to_case(conv, index=0)
    assert case.qa[0].gold == "42"


def test_none_answer_with_no_adversarial_yields_empty_gold() -> None:
    conv = _conversation(qa=[{"question": "q", "answer": None, "category": 1, "evidence": []}])
    case = _conversation_to_case(conv, index=0)
    assert case.qa[0].gold == ""


def test_qa_without_category_is_silently_skipped() -> None:
    """Guard against dataset rows missing the required bucket key."""

    conv = _conversation(
        qa=[
            {"question": "no-cat", "answer": "a", "evidence": []},
            {"question": "ok", "answer": "b", "category": 2, "evidence": []},
        ]
    )
    case = _conversation_to_case(conv, index=0)
    assert [q.question for q in case.qa] == ["ok"]


def test_evidence_ids_propagate_in_order() -> None:
    conv = _conversation(
        qa=[
            {
                "question": "q",
                "answer": "a",
                "category": 1,
                "evidence": ["D1:2", "D2:1", "D1:1"],
            }
        ]
    )
    case = _conversation_to_case(conv, index=0)
    assert case.qa[0].evidence_turn_ids == ("D1:2", "D2:1", "D1:1")


def test_parse_sessions_sorts_by_index() -> None:
    """Session order must be deterministic even with out-of-order keys."""

    obj = {
        "conversation": {
            "session_3": [_turn("D3:1")],
            "session_1": [_turn("D1:1")],
            "session_2": [_turn("D2:1")],
            "session_1_date_time": "d1",
            "session_2_date_time": "d2",
            "session_3_date_time": "d3",
        }
    }
    sessions = _parse_sessions(obj)
    assert [s.session_index for s in sessions] == [1, 2, 3]


def test_parse_sessions_drops_turns_missing_required_fields() -> None:
    obj = {
        "conversation": {
            "session_1": [
                {"dia_id": "D1:1", "speaker": "a", "text": "ok"},
                {"dia_id": "D1:2", "speaker": "a"},  # missing text
                {"speaker": "a", "text": "orphan"},  # missing dia_id
            ]
        }
    }
    sessions = _parse_sessions(obj)
    assert [t.turn_id for t in sessions[0].turns] == ["D1:1"]


def test_parse_sessions_handles_blip_caption() -> None:
    obj = {
        "conversation": {
            "session_1": [
                {
                    "dia_id": "D1:1",
                    "speaker": "a",
                    "text": "look",
                    "blip_caption": "a sunset",
                }
            ]
        }
    }
    sessions = _parse_sessions(obj)
    assert sessions[0].turns[0].image_caption == "a sunset"


def test_parse_sessions_with_no_inner_returns_empty() -> None:
    assert _parse_sessions({}) == []
    assert _parse_sessions({"conversation": "not a dict"}) == []


def test_parse_qa_skips_non_dict_rows() -> None:
    obj = {"qa": [None, {"question": "ok", "answer": "a", "category": 1}]}
    items = _parse_qa(obj)
    assert [q.question for q in items] == ["ok"]


def test_dataset_iter_yields_conversations_in_order() -> None:
    ds = LocomoDataset(
        [_conversation(conversation_id=f"c{i}") for i in range(3)],
        path=Path("/tmp/fake.json"),
        file_sha256="deadbeef",
    )
    ids = [c.case_id for c in ds]
    assert ids == ["c0", "c1", "c2"]
    assert len(ds) == 3


def test_dataset_limit_truncates_head() -> None:
    ds = LocomoDataset(
        [_conversation(conversation_id=f"c{i}") for i in range(5)],
        path=Path("/tmp/fake.json"),
        file_sha256="deadbeef",
        limit=2,
    )
    assert [c.case_id for c in ds] == ["c0", "c1"]


def test_dataset_limit_zero_yields_nothing() -> None:
    ds = LocomoDataset(
        [_conversation(conversation_id=f"c{i}") for i in range(3)],
        path=Path("/tmp/fake.json"),
        file_sha256="deadbeef",
        limit=0,
    )
    assert list(ds) == []


def test_dataset_rejects_negative_limit() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LocomoDataset(
            [_conversation()],
            path=Path("/tmp/fake.json"),
            file_sha256="deadbeef",
            limit=-1,
        )


def test_from_path_reads_file_and_hashes_bytes(tmp_path: Path) -> None:
    conversations = [_conversation(conversation_id="c0")]
    fixture = tmp_path / "locomo10.json"
    _write_fixture(fixture, conversations)

    ds = LocomoDataset.from_path(fixture)
    assert len(ds) == 1
    assert ds.path == fixture
    assert ds.file_sha256 == hashlib.sha256(fixture.read_bytes()).hexdigest()


def test_from_path_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LocomoDataset.from_path(tmp_path / "nope.json")


def test_from_path_rejects_non_list_top_level(tmp_path: Path) -> None:
    fixture = tmp_path / "locomo10.json"
    fixture.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON array"):
        LocomoDataset.from_path(fixture)


def test_descriptor_hash_stable_for_identical_inputs(tmp_path: Path) -> None:
    fixture = tmp_path / "a.json"
    _write_fixture(fixture, [_conversation(conversation_id="c0")])
    a = LocomoDataset.from_path(fixture)
    b = LocomoDataset.from_path(fixture)
    assert a.descriptor_hash() == b.descriptor_hash()


def test_descriptor_hash_changes_with_file_bytes(tmp_path: Path) -> None:
    f_a = tmp_path / "a.json"
    f_b = tmp_path / "b.json"
    _write_fixture(f_a, [_conversation(conversation_id="c0")])
    _write_fixture(f_b, [_conversation(conversation_id="c1")])
    assert LocomoDataset.from_path(f_a).descriptor_hash() != (
        LocomoDataset.from_path(f_b).descriptor_hash()
    )


def test_descriptor_hash_changes_with_limit(tmp_path: Path) -> None:
    fixture = tmp_path / "a.json"
    _write_fixture(fixture, [_conversation(conversation_id=f"c{i}") for i in range(3)])
    unlimited = LocomoDataset.from_path(fixture)
    limited = LocomoDataset.from_path(fixture, limit=2)
    assert unlimited.descriptor_hash() != limited.descriptor_hash()


def test_load_dataset_locomo_end_to_end(tmp_path: Path) -> None:
    """``load_dataset('locomo', path=...)`` is the public entry point."""

    fixture = tmp_path / "locomo10.json"
    _write_fixture(fixture, [_conversation(conversation_id="only")])
    ds = load_dataset("locomo", path=fixture)
    assert isinstance(ds, LocomoDataset)
    assert len(ds) == 1


def test_load_dataset_locomo_rejects_bad_path_type() -> None:
    with pytest.raises(TypeError, match="path must be"):
        load_dataset("locomo", path=42)  # type: ignore[arg-type]


def test_beam_is_still_unavailable() -> None:
    with pytest.raises(DatasetUnavailableError, match="PR-11"):
        load_dataset("beam")
