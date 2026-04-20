"""Benchmark-owned dataclasses.

These types are the neutral ground that dataset loaders, adapters, and the
runner exchange. They are intentionally minimal and immutable where possible
— the benchmark neither owns a graph nor a retrieval engine, and the types
reflect that.

Memory systems are free to use their own internal types; adapters convert
between a system's types and these at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DatasetName = Literal["longmemeval", "locomo", "beam"]


@dataclass(frozen=True)
class Turn:
    """One utterance in a conversation.

    ``turn_id`` is stable within a case and is the anchor that evidence
    annotations reference. Adapters propagate it back on ``RetrievedUnit``
    so the scorer can compute evidence-keyed retrieval metrics.
    """

    turn_id: str
    speaker: str
    text: str
    timestamp: str | None = None
    image_caption: str | None = None


@dataclass(frozen=True)
class Session:
    """A chronological group of turns that share a session boundary."""

    session_index: int
    turns: tuple[Turn, ...]
    session_time: str | None = None
    session_id: str | None = None


@dataclass(frozen=True)
class QAItem:
    """One question + gold answer + annotations.

    ``evidence_turn_ids`` lists the load-bearing turns for this question. If
    the dataset does not annotate evidence, the tuple is empty and evidence
    KPIs for this question are reported as ``null``.
    """

    question_id: str
    question: str
    gold: str
    question_type: str
    category: int | None = None
    evidence_turn_ids: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkCase:
    """One independent case: a conversation (or haystack) and its QA set."""

    case_id: str
    sessions: tuple[Session, ...]
    qa: tuple[QAItem, ...]
    dataset: DatasetName


@dataclass(frozen=True)
class RetrievedUnit:
    """One unit of retrieved context that the memory system handed to
    generation.

    ``text`` is the verbatim content sent to the answerer. ``source_turn_ids``
    lets the scorer compute turn-level evidence metrics; when the adapter
    cannot supply it, the scorer falls back to substring matching against
    known turn text.
    """

    unit_id: str
    text: str
    score: float | None = None
    source_turn_ids: tuple[str, ...] = ()


@dataclass
class AnswerResult:
    """One answer from the memory system.

    Adapter self-reports ``retrieval_time_ms`` and ``generation_time_ms`` as
    a courtesy; the runner independently measures ``answer_question`` wall
    time and stores both. Discrepancy between the sum of self-reports and
    the runner's measurement is logged per-query as a telemetry-drift signal.
    """

    answer: str
    retrieval_time_ms: float
    generation_time_ms: float
    units_retrieved: int = 0
    tokens_retrieved: int = 0
    retrieved: tuple[RetrievedUnit, ...] = ()


__all__ = [
    "DatasetName",
    "Turn",
    "Session",
    "QAItem",
    "BenchmarkCase",
    "RetrievedUnit",
    "AnswerResult",
]
