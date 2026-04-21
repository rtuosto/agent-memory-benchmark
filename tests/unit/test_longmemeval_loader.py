"""Tests for the LongMemEval loader — row conversion, stratification, descriptor.

These exercise the in-memory transforms only; they never hit Hugging Face.
:meth:`LongMemEvalDataset.load_s` / :meth:`load_m_local` are thin wrappers
around ``datasets.load_dataset`` that defer to the same ``__init__`` path
these tests exercise directly.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_memory_benchmark.datasets import DatasetUnavailableError, load_dataset
from agent_memory_benchmark.datasets.longmemeval import (
    HF_DATASET_ID,
    HF_REVISION,
    LongMemEvalDataset,
    _row_to_case,
    _stratified_indices,
)
from agent_memory_benchmark.types import BenchmarkCase


def _row(
    question_id: str,
    *,
    question_type: str = "single-session-user",
    question: str = "What did I say?",
    answer: str | int | None = "a shell necklace",
    answer_session_ids: list[str] | None = None,
    haystack_session_ids: list[str] | None = None,
    haystack_dates: list[str] | None = None,
    haystack_sessions: list[list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    hs_ids = haystack_session_ids or ["sess_1", "sess_2"]
    hs_dates = haystack_dates or ["2024-01-01", "2024-01-02"]
    hs_sessions = haystack_sessions or [
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        [
            {"role": "user", "content": "question turn"},
        ],
    ]
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": question,
        "answer": answer,
        "answer_session_ids": answer_session_ids or [],
        "haystack_session_ids": hs_ids,
        "haystack_dates": hs_dates,
        "haystack_sessions": hs_sessions,
    }


def test_row_to_case_builds_sessions_and_qa() -> None:
    row = _row("q1", answer_session_ids=["sess_2"])
    case = _row_to_case(row)
    assert isinstance(case, BenchmarkCase)
    assert case.case_id == "q1"
    assert case.dataset == "longmemeval"
    assert len(case.sessions) == 2
    s1, s2 = case.sessions
    assert s1.session_index == 1
    assert s1.session_id == "sess_1"
    assert s1.session_time == "2024-01-01"
    assert [t.turn_id for t in s1.turns] == ["sess_1:1", "sess_1:2"]
    assert [t.speaker for t in s1.turns] == ["user", "assistant"]
    assert s2.session_index == 2
    assert [t.turn_id for t in s2.turns] == ["sess_2:1"]
    assert len(case.qa) == 1
    qa = case.qa[0]
    assert qa.question_id == "q1"
    assert qa.gold == "a shell necklace"
    assert qa.question_type == "single-session-user"
    assert qa.metadata == {"abstention": "0"}


def test_row_to_case_evidence_turn_ids_cover_answer_sessions_only() -> None:
    row = _row(
        "q2",
        answer_session_ids=["sess_2"],
        haystack_session_ids=["sess_1", "sess_2", "sess_3"],
        haystack_dates=["d1", "d2", "d3"],
        haystack_sessions=[
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
            [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}],
            [{"role": "user", "content": "e"}],
        ],
    )
    case = _row_to_case(row)
    assert case.qa[0].evidence_turn_ids == ("sess_2:1", "sess_2:2")


def test_row_to_case_empty_answer_session_ids_yields_no_evidence() -> None:
    case = _row_to_case(_row("q3", answer_session_ids=[]))
    assert case.qa[0].evidence_turn_ids == ()


def test_row_to_case_handles_integer_answer() -> None:
    case = _row_to_case(_row("q4", answer=42))
    assert case.qa[0].gold == "42"


def test_row_to_case_handles_none_answer() -> None:
    case = _row_to_case(_row("q5", answer=None))
    assert case.qa[0].gold == ""


def test_row_to_case_flags_abstention_in_metadata() -> None:
    case = _row_to_case(_row("q6_abs_1"))
    assert case.qa[0].metadata["abstention"] == "1"


def test_row_to_case_rejects_mismatched_haystack_lengths() -> None:
    row = _row("q7")
    row["haystack_dates"] = ["d1"]  # mismatch vs haystack_session_ids length 2
    with pytest.raises(ValueError, match="mismatched haystack_"):
        _row_to_case(row)


def test_row_to_case_strips_turn_content_whitespace() -> None:
    row = _row(
        "q8",
        haystack_sessions=[
            [{"role": "user", "content": "  padded  "}],
            [{"role": "user", "content": "  padded  "}],
        ],
    )
    case = _row_to_case(row)
    assert case.sessions[0].turns[0].text == "padded"


def test_dataset_iter_yields_cases_in_row_order() -> None:
    rows = [_row(f"q{i}") for i in range(3)]
    ds = LongMemEvalDataset(rows, split="s", revision="rev0")
    case_ids = [c.case_id for c in ds]
    assert case_ids == ["q0", "q1", "q2"]
    assert len(ds) == 3


def test_dataset_head_limit_truncates_in_order() -> None:
    rows = [_row(f"q{i}", question_type="general") for i in range(5)]
    ds = LongMemEvalDataset(rows, split="s", revision="rev0", limit=2, limit_strategy="head")
    assert [c.case_id for c in ds] == ["q0", "q1"]


def test_dataset_stratified_limit_interleaves_types() -> None:
    rows: list[dict[str, Any]] = []
    # 3 types × 3 rows each, interleaved "a", "a", "a", "b", "b", "b", "c", "c", "c"
    for t in ("a", "b", "c"):
        for i in range(3):
            rows.append(_row(f"{t}{i}", question_type=t))
    ds = LongMemEvalDataset(rows, split="s", revision="rev0", limit=3, limit_strategy="stratified")
    ids = [c.case_id for c in ds]
    assert ids == ["a0", "b0", "c0"]


def test_dataset_stratified_with_larger_limit_allocates_proportionally() -> None:
    rows = []
    for t in ("a", "b"):
        for i in range(4):
            rows.append(_row(f"{t}{i}", question_type=t))
    ds = LongMemEvalDataset(rows, split="s", revision="rev0", limit=4, limit_strategy="stratified")
    ids = [c.case_id for c in ds]
    # Round-robin a,b,a,b from types with equal allocation of 2 each.
    assert ids == ["a0", "b0", "a1", "b1"]


def test_stratified_indices_handles_zero_or_empty() -> None:
    assert _stratified_indices([], 5) == []
    assert _stratified_indices([_row("q0")], 0) == []


def test_stratified_indices_terminates_when_limit_below_num_types() -> None:
    """Guard against the infinite-loop regression.

    With more question types than slots, the ``max(1, ...)`` floor cannot
    be satisfied; the allocator must fall back to picking the ``limit``
    largest types at one slot each rather than spinning forever.
    """

    rows: list[dict[str, Any]] = []
    # 7 types × 10 rows (a=10, b=9, c=8, ..., g=4). Sizes strictly decreasing.
    sizes = {"a": 10, "b": 9, "c": 8, "d": 7, "e": 6, "f": 5, "g": 4}
    for qt, n in sizes.items():
        for i in range(n):
            rows.append(_row(f"{qt}{i}", question_type=qt))

    idxs = _stratified_indices(rows, limit=3)

    assert len(idxs) == 3
    selected_types = {rows[i]["question_type"] for i in idxs}
    # The 3 largest types (a, b, c) win.
    assert selected_types == {"a", "b", "c"}


def test_stratified_indices_limit_equals_num_types() -> None:
    """Boundary: limit == num_types should give exactly one per type."""

    rows: list[dict[str, Any]] = []
    for qt in ("a", "b", "c"):
        for i in range(5):
            rows.append(_row(f"{qt}{i}", question_type=qt))

    idxs = _stratified_indices(rows, limit=3)
    assert len(idxs) == 3
    assert {rows[i]["question_type"] for i in idxs} == {"a", "b", "c"}


def test_dataset_stratified_limit_1_smoke() -> None:
    """``--limit 1`` is the smoke-test path used in docs; must not hang."""

    rows: list[dict[str, Any]] = []
    for qt in ("a", "b", "c", "d", "e", "f", "g"):
        rows.append(_row(f"{qt}0", question_type=qt))
    ds = LongMemEvalDataset(rows, split="s", revision="rev0", limit=1, limit_strategy="stratified")
    assert len(ds) == 1


def test_dataset_limit_larger_than_rows_keeps_all() -> None:
    rows = [_row(f"q{i}") for i in range(3)]
    ds = LongMemEvalDataset(rows, split="s", revision="rev0", limit=99, limit_strategy="stratified")
    assert len(ds) == 3


def test_dataset_limit_none_materializes_all() -> None:
    rows = [_row(f"q{i}") for i in range(4)]
    ds = LongMemEvalDataset(rows, split="s", revision="rev0", limit=None)
    assert len(ds) == 4


def test_dataset_rejects_negative_limit() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LongMemEvalDataset([_row("q0")], split="s", revision="rev0", limit=-1)


def test_dataset_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="limit_strategy must be"):
        LongMemEvalDataset([_row("q0")], split="s", revision="rev0", limit_strategy="???")


def test_descriptor_hash_stable_for_identical_inputs() -> None:
    rows = [_row(f"q{i}") for i in range(3)]
    a = LongMemEvalDataset(rows, split="s", revision="rev0")
    b = LongMemEvalDataset(rows, split="s", revision="rev0")
    assert a.descriptor_hash() == b.descriptor_hash()


def test_descriptor_hash_changes_with_revision() -> None:
    rows = [_row(f"q{i}") for i in range(3)]
    a = LongMemEvalDataset(rows, split="s", revision="rev0")
    b = LongMemEvalDataset(rows, split="s", revision="rev1")
    assert a.descriptor_hash() != b.descriptor_hash()


def test_descriptor_hash_changes_with_split() -> None:
    rows = [_row(f"q{i}") for i in range(3)]
    a = LongMemEvalDataset(rows, split="s", revision="rev0")
    b = LongMemEvalDataset(rows, split="m", revision="rev0")
    assert a.descriptor_hash() != b.descriptor_hash()


def test_descriptor_hash_changes_with_limit() -> None:
    rows = [_row(f"q{i}") for i in range(5)]
    unlimited = LongMemEvalDataset(rows, split="s", revision="rev0")
    limited = LongMemEvalDataset(rows, split="s", revision="rev0", limit=3, limit_strategy="head")
    assert unlimited.descriptor_hash() != limited.descriptor_hash()


def test_descriptor_hash_differs_between_head_and_stratified() -> None:
    """Same limit, different selection strategy — hashes must diverge so the
    cache treats them as distinct dataset views."""

    rows: list[dict[str, Any]] = []
    for t in ("a", "b"):
        for i in range(3):
            rows.append(_row(f"{t}{i}", question_type=t))
    head = LongMemEvalDataset(rows, split="s", revision="rev0", limit=2, limit_strategy="head")
    stratified = LongMemEvalDataset(
        rows, split="s", revision="rev0", limit=2, limit_strategy="stratified"
    )
    assert head.descriptor_hash() != stratified.descriptor_hash()


def test_load_dataset_longmemeval_requires_split() -> None:
    with pytest.raises(ValueError, match="requires a 'split'"):
        load_dataset("longmemeval")


def test_load_dataset_locomo_is_reserved_for_pr9() -> None:
    with pytest.raises(DatasetUnavailableError, match="PR-9"):
        load_dataset("locomo")


def test_load_dataset_beam_is_reserved_for_pr11() -> None:
    with pytest.raises(DatasetUnavailableError, match="PR-11"):
        load_dataset("beam")


def test_load_dataset_unknown_name_errors() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("squad")


def test_load_dataset_m_split_without_path_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    # We don't actually reach the HF import — the M path check fires first.
    with pytest.raises(ValueError, match="requires a local JSON path"):
        load_dataset("longmemeval", split="m")


def test_hf_revision_is_pinned_not_main() -> None:
    """Floating 'main' revisions are explicitly disallowed (plan invariant)."""

    assert HF_REVISION != "main"
    assert len(HF_REVISION) == 40  # standard git sha-1 hex length
    assert HF_DATASET_ID == "xiaowu0162/longmemeval-cleaned"
