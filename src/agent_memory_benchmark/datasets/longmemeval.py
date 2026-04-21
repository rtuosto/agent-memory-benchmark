"""LongMemEval loader — HF ``xiaowu0162/longmemeval-cleaned`` at a pinned revision.

LongMemEval ships as two JSON files under a dataset repo (no loader script):
``longmemeval_s_cleaned.json`` (~100 Q, the primary benchmark split we run)
and ``longmemeval_m_cleaned.json`` (multi-GB, M-split). Each row is one
*question* packaged with its haystack of prior sessions — the benchmark
treats that as a single :class:`BenchmarkCase` per question.

Revision pinning is mandatory: raw HF URLs without a commit SHA would let a
dataset author silently change inputs under our cache. We pin at module load
by default (``HF_REVISION``) and allow overriding via ``revision=`` on the
classmethod.

Evidence annotations in LongMemEval are session-level (``answer_session_ids``)
rather than turn-level; the loader expands them to every turn inside the
listed sessions so the evidence KPIs have turn IDs to compare against. That
is the best-available granularity the dataset exposes.

``datasets`` is imported lazily inside :meth:`load_s` / :meth:`load_m_local`
so the module stays importable on environments where the HF toolchain (or
its transitive pandas dep) cannot be loaded — tests exercise row conversion
with hand-built rows.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from ..types import BenchmarkCase, DatasetName, QAItem, Session, Turn
from .base import DatasetAdapter

HF_DATASET_ID = "xiaowu0162/longmemeval-cleaned"
HF_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
HF_S_FILENAME = "longmemeval_s_cleaned.json"
HF_M_FILENAME = "longmemeval_m_cleaned.json"

_HF_FILE_URL = "https://huggingface.co/datasets/" + HF_DATASET_ID + "/resolve/{revision}/{filename}"


class LongMemEvalDataset(DatasetAdapter):
    """LongMemEval S/M as a stream of :class:`BenchmarkCase` instances."""

    name: DatasetName = "longmemeval"

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        split: str,
        revision: str,
        limit: int | None = None,
        limit_strategy: str = "full",
    ) -> None:
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")
        if limit_strategy not in ("full", "stratified", "head"):
            raise ValueError(
                f"limit_strategy must be one of 'full' | 'stratified' | 'head', got {limit_strategy!r}"
            )

        materialized = list(rows)
        if limit is None or limit >= len(materialized):
            self._rows: list[dict[str, Any]] = materialized
            self._applied_strategy = "full"
        elif limit_strategy == "head":
            self._rows = materialized[:limit]
            self._applied_strategy = "head"
        else:
            indices = _stratified_indices(materialized, limit)
            self._rows = [materialized[i] for i in indices]
            self._applied_strategy = "stratified"

        self._split = split
        self._revision = revision
        self._limit = limit

    @classmethod
    def load_s(
        cls,
        *,
        revision: str = HF_REVISION,
        limit: int | None = None,
        limit_strategy: str = "stratified",
    ) -> LongMemEvalDataset:
        """Load the S split (HF-hosted, revision-pinned)."""

        rows = _load_hf_json(filename=HF_S_FILENAME, revision=revision)
        return cls(
            rows,
            split="s",
            revision=revision,
            limit=limit,
            limit_strategy=limit_strategy,
        )

    @classmethod
    def load_m_local(
        cls,
        path: str | Path,
        *,
        revision: str = HF_REVISION,
        limit: int | None = None,
        limit_strategy: str = "stratified",
    ) -> LongMemEvalDataset:
        """Load the M split from a local JSON file (the HF-hosted file is multi-GB).

        ``revision`` is recorded in the descriptor hash for bookkeeping even
        though the bytes come from ``path`` — it identifies the schema
        version the user downloaded against.
        """

        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"LongMemEval M data not found: {p}")
        rows = _load_local_json(p)
        return cls(
            rows,
            split="m",
            revision=revision,
            limit=limit,
            limit_strategy=limit_strategy,
        )

    def __iter__(self) -> Iterator[BenchmarkCase]:
        for row in self._rows:
            yield _row_to_case(row)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def split(self) -> str:
        return self._split

    @property
    def revision(self) -> str:
        return self._revision

    def descriptor_hash(self) -> str:
        limit_sig = "all" if self._limit is None else str(self._limit)
        parts = (
            self.name,
            self._split,
            self._revision,
            f"{self._applied_strategy}:{limit_sig}",
        )
        payload = b"\x1e".join(p.encode("utf-8") for p in parts)
        return hashlib.sha256(payload).hexdigest()


def load_longmemeval(
    split: str,
    *,
    m_path: str | Path | None = None,
    revision: str = HF_REVISION,
    limit: int | None = None,
    limit_strategy: str = "stratified",
) -> LongMemEvalDataset:
    """Dispatch to the right class method for ``split``."""

    normalized = split.strip().lower()
    if normalized in ("s", "longmemeval_s", "longmemeval_s_cleaned"):
        return LongMemEvalDataset.load_s(
            revision=revision, limit=limit, limit_strategy=limit_strategy
        )
    if normalized in ("m", "longmemeval_m", "longmemeval_m_cleaned"):
        if not m_path:
            raise ValueError(
                "LongMemEval M split requires a local JSON path (--m-data) because the HF-hosted "
                f"file is multi-GB; download {HF_M_FILENAME!r} from the dataset repo."
            )
        return LongMemEvalDataset.load_m_local(
            m_path, revision=revision, limit=limit, limit_strategy=limit_strategy
        )
    raise ValueError(f"Unknown LongMemEval split {split!r}; use 's' or 'm'.")


def _row_to_case(row: dict[str, Any]) -> BenchmarkCase:
    """Convert one HF row into a :class:`BenchmarkCase`.

    Schema observed in ``xiaowu0162/longmemeval-cleaned`` rows:

    - ``question_id``, ``question_type``, ``question``, ``answer``
    - ``answer_session_ids`` — session IDs that carry the evidence
    - ``haystack_session_ids``, ``haystack_dates``, ``haystack_sessions``
      — parallel lists; ``haystack_sessions[i]`` is the list of
      ``{"role": ..., "content": ...}`` dicts for session i.

    Turn IDs are composed as ``f"{session_id}:{1-based turn idx}"`` — same
    scheme the predecessor used, so evidence matching stays cross-comparable.
    """

    question_id = str(row["question_id"])
    question_type = str(row["question_type"])
    question = str(row["question"])
    answer_raw = row.get("answer")
    gold = "" if answer_raw is None else str(answer_raw)

    answer_session_ids = {str(s) for s in row.get("answer_session_ids") or []}
    haystack_session_ids = [str(s) for s in row.get("haystack_session_ids") or []]
    haystack_dates = [str(d) for d in row.get("haystack_dates") or []]
    haystack_sessions_raw = row.get("haystack_sessions") or []

    if not (len(haystack_session_ids) == len(haystack_dates) == len(haystack_sessions_raw)):
        raise ValueError(
            f"Row {question_id!r} has mismatched haystack_* list lengths: "
            f"ids={len(haystack_session_ids)} dates={len(haystack_dates)} "
            f"sessions={len(haystack_sessions_raw)}"
        )

    sessions: list[Session] = []
    evidence_turn_ids: list[str] = []
    triples = zip(haystack_session_ids, haystack_dates, haystack_sessions_raw, strict=True)
    for session_index, (sid, date, turns_raw) in enumerate(triples, start=1):
        is_evidence_session = sid in answer_session_ids
        turns: list[Turn] = []
        for turn_index, raw_turn in enumerate(turns_raw, start=1):
            role = str(raw_turn.get("role", "user"))
            content = str(raw_turn.get("content", "")).strip()
            turn_id = f"{sid}:{turn_index}"
            turns.append(Turn(turn_id=turn_id, speaker=role, text=content))
            if is_evidence_session:
                evidence_turn_ids.append(turn_id)
        sessions.append(
            Session(
                session_index=session_index,
                turns=tuple(turns),
                session_time=str(date) if date else None,
                session_id=sid,
            )
        )

    metadata: dict[str, str] = {"abstention": "1" if "_abs" in question_id else "0"}
    qa = QAItem(
        question_id=question_id,
        question=question,
        gold=gold,
        question_type=question_type,
        evidence_turn_ids=tuple(evidence_turn_ids),
        metadata=metadata,
    )
    return BenchmarkCase(
        case_id=question_id,
        sessions=tuple(sessions),
        qa=(qa,),
        dataset="longmemeval",
    )


def _stratified_indices(rows: Sequence[dict[str, Any]], limit: int) -> list[int]:
    """Return ``limit`` dataset indices, round-robin by ``question_type``.

    Matches the predecessor's behavior: proportional allocation with a
    min-1-per-type guarantee, then a deterministic alphabetical round-robin
    interleave so early truncation still covers every question type.
    """

    by_type: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_type[str(row["question_type"])].append(i)

    total = len(rows)
    if total == 0 or limit <= 0:
        return []
    take = min(limit, total)
    num_types = len(by_type)

    alloc: dict[str, int] = {
        qt: max(1, round(take * len(idxs) / total)) for qt, idxs in by_type.items()
    }
    diff = sum(alloc.values()) - take
    types_by_size = sorted(by_type, key=lambda t: len(by_type[t]), reverse=True)
    cursor = 0
    while diff > 0:
        qt = types_by_size[cursor % num_types]
        if alloc[qt] > 1:
            alloc[qt] -= 1
            diff -= 1
        cursor += 1
    while diff < 0:
        qt = types_by_size[cursor % num_types]
        if alloc[qt] < len(by_type[qt]):
            alloc[qt] += 1
            diff += 1
        cursor += 1

    per_type_slice: dict[str, list[int]] = {qt: idxs[: alloc[qt]] for qt, idxs in by_type.items()}
    type_order = sorted(per_type_slice)
    positions: dict[str, int] = dict.fromkeys(type_order, 0)
    result: list[int] = []
    while type_order:
        exhausted: list[str] = []
        for qt in type_order:
            if positions[qt] < len(per_type_slice[qt]):
                result.append(per_type_slice[qt][positions[qt]])
                positions[qt] += 1
            else:
                exhausted.append(qt)
        for qt in exhausted:
            type_order.remove(qt)
    return result


def _load_hf_json(*, filename: str, revision: str) -> list[dict[str, Any]]:
    """Pull a JSON file from the HF dataset repo at ``revision`` via ``datasets``.

    Lazy import so this module stays importable on environments where the
    HF stack can't be loaded (e.g. pandas blocked by App Control on Windows).
    """

    from datasets import load_dataset  # type: ignore[import-untyped]

    url = _HF_FILE_URL.format(revision=revision, filename=filename)
    ds = load_dataset("json", data_files=url, split="train")
    return list(ds)


def _load_local_json(path: Path) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("json", data_files=str(path), split="train")
    return list(ds)


__all__ = [
    "HF_DATASET_ID",
    "HF_M_FILENAME",
    "HF_REVISION",
    "HF_S_FILENAME",
    "LongMemEvalDataset",
    "load_longmemeval",
]
