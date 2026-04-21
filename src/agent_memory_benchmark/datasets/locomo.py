"""LOCOMO (locomo10.json) loader.

LOCOMO ships as a single local JSON array (``locomo10.json``) holding 10
long-horizon conversations. Each conversation carries its QA set inline. No
HF redistribution — licenses prohibit bundling, so the benchmark accepts a
user-supplied path and hashes the file bytes into :meth:`descriptor_hash`.

Schema observed (mirrors the predecessor loader at ``~/code/agent-memory``):

- Top-level: list of conversation objects
- Each conversation has ``conversation_id``, ``conversation`` (dict of
  ``session_<N>`` + ``session_<N>_date_time`` pairs, plus ``speaker_a`` /
  ``speaker_b``), and ``qa`` (list of items with ``question``, ``answer``,
  ``category`` 1-5, ``evidence`` list of dia_ids, optional
  ``adversarial_answer``).
- Each turn has a ``dia_id`` (unique within the conversation — e.g.
  ``"D1:1"``) which we use directly as the benchmark :class:`Turn`
  ``turn_id``. No per-session prefix is needed.

**Category 5 questions are excluded from the benchmark** (marked
unanswerable upstream). Cases with an empty QA set after filtering are
still yielded with ``qa=()`` so the ingestion path keeps a stable count of
conversations.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..types import BenchmarkCase, DatasetName, QAItem, Session, Turn
from .base import DatasetAdapter

EXCLUDED_CATEGORY = 5

_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")


class LocomoDataset(DatasetAdapter):
    """LOCOMO conversations as a stream of :class:`BenchmarkCase` instances."""

    name: DatasetName = "locomo"

    def __init__(
        self,
        conversations: list[dict[str, Any]],
        *,
        path: Path,
        file_sha256: str,
        limit: int | None = None,
    ) -> None:
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")

        materialized = list(conversations)
        if limit is None or limit >= len(materialized):
            self._conversations: list[dict[str, Any]] = materialized
            self._applied_limit_sig = "all"
        else:
            self._conversations = materialized[:limit]
            self._applied_limit_sig = f"head:{limit}"

        self._path = path
        self._file_sha256 = file_sha256
        self._limit = limit

    @classmethod
    def from_path(cls, path: str | Path, *, limit: int | None = None) -> LocomoDataset:
        """Load from a local ``locomo10.json`` file."""

        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"LOCOMO data not found: {p}")
        raw_bytes = p.read_bytes()
        file_sha = hashlib.sha256(raw_bytes).hexdigest()
        data = json.loads(raw_bytes.decode("utf-8"))
        if not isinstance(data, list):
            raise ValueError("locomo10.json must be a JSON array at the top level")
        conversations = [obj for obj in data if isinstance(obj, dict)]
        return cls(conversations, path=p, file_sha256=file_sha, limit=limit)

    def __iter__(self) -> Iterator[BenchmarkCase]:
        for index, obj in enumerate(self._conversations):
            yield _conversation_to_case(obj, index=index)

    def __len__(self) -> int:
        return len(self._conversations)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def file_sha256(self) -> str:
        return self._file_sha256

    def descriptor_hash(self) -> str:
        parts = (self.name, self._file_sha256, self._applied_limit_sig)
        payload = b"\x1e".join(p.encode("utf-8") for p in parts)
        return hashlib.sha256(payload).hexdigest()


def load_locomo(path: str | Path, *, limit: int | None = None) -> LocomoDataset:
    return LocomoDataset.from_path(path, limit=limit)


def _conversation_to_case(obj: dict[str, Any], *, index: int) -> BenchmarkCase:
    case_id = str(obj.get("conversation_id") or f"locomo_{index}")
    sessions = _parse_sessions(obj)
    qa_items = _parse_qa(obj)
    return BenchmarkCase(
        case_id=case_id,
        sessions=tuple(sessions),
        qa=tuple(qa_items),
        dataset="locomo",
    )


def _parse_sessions(obj: dict[str, Any]) -> list[Session]:
    """Flatten the ``session_<N>`` / ``session_<N>_date_time`` key pairs."""

    inner = obj.get("conversation")
    if not isinstance(inner, dict):
        return []

    session_indices: list[int] = []
    for key in inner:
        m = _SESSION_KEY_RE.match(key)
        if m:
            session_indices.append(int(m.group(1)))
    session_indices.sort()

    sessions: list[Session] = []
    for idx in session_indices:
        turns_raw = inner.get(f"session_{idx}")
        if not isinstance(turns_raw, list):
            continue
        date_time = inner.get(f"session_{idx}_date_time")
        turns: list[Turn] = []
        for raw in turns_raw:
            if not isinstance(raw, dict):
                continue
            dia_id = raw.get("dia_id")
            speaker = raw.get("speaker")
            text = raw.get("text")
            if dia_id is None or speaker is None or text is None:
                continue
            turns.append(
                Turn(
                    turn_id=str(dia_id),
                    speaker=str(speaker),
                    text=str(text),
                    image_caption=_optional_str(raw.get("blip_caption")),
                )
            )
        sessions.append(
            Session(
                session_index=idx,
                turns=tuple(turns),
                session_time=str(date_time) if date_time else None,
                session_id=f"session_{idx}",
            )
        )
    return sessions


def _parse_qa(obj: dict[str, Any]) -> list[QAItem]:
    qa_raw = obj.get("qa")
    if not isinstance(qa_raw, list):
        return []
    items: list[QAItem] = []
    for row_index, row in enumerate(qa_raw):
        if not isinstance(row, dict):
            continue
        category_raw = row.get("category")
        if not isinstance(category_raw, int):
            continue
        if category_raw == EXCLUDED_CATEGORY:
            continue
        question = str(row.get("question", "")).strip()
        gold = _gold_answer(row)
        evidence_raw = row.get("evidence") or []
        evidence_turn_ids = tuple(str(e) for e in evidence_raw if isinstance(e, (str, int)))
        items.append(
            QAItem(
                question_id=f"qa_{row_index}",
                question=question,
                gold=gold,
                question_type=f"cat_{category_raw}",
                category=category_raw,
                evidence_turn_ids=evidence_turn_ids,
            )
        )
    return items


def _gold_answer(row: dict[str, Any]) -> str:
    answer = row.get("answer")
    if answer is not None:
        return str(answer).strip()
    adversarial = row.get("adversarial_answer")
    if adversarial:
        return str(adversarial).strip()
    return ""


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


__all__ = [
    "EXCLUDED_CATEGORY",
    "LocomoDataset",
    "load_locomo",
]
