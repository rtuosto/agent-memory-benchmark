"""Run-level persistence types: ``QARecord``, ``RunManifest``, ``RunDir``.

One ``answers.json`` per run holds ``meta`` (the :class:`RunManifest` dict) plus
``records`` (a list of :class:`QARecord`). ``meta.json`` is the human-readable
sidecar. ``scorecard.{json,md}`` are derived from ``answers.json`` — anything
downstream (``amb compare``, ``amb summarize``, PR-6's byte-stable judge cache)
reads these files, not the runner's in-memory state.

Why keep these dataclasses flat:

- They serialize with :func:`dataclasses.asdict` without custom hooks.
- ``QARecord`` mirrors the predecessor field names (``retrieval_time_ms``,
  ``generation_time_ms``, ``total_answer_time_ms``, ``judge_runs``) so
  scorecards port forward with no translation.
- New fields for this repo (``answer_discrepancy_ms``, ``replicate_idx``,
  ``evidence_turn_ids``, ``retrieved_turn_ids``) are additive; unknown fields
  on disk are dropped by :func:`_dict_to_record` so schema evolution is a
  one-way ratchet.

``RunDir`` is just a path wrapper — all filenames live in one place so callers
never spell them as strings.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class JudgeRun:
    """One judge verdict. ``runs > 1`` produces a list of these per question."""

    correct: bool
    raw: str


@dataclass
class QARecord:
    """One (question, answer, optional judge-verdict) row.

    ``key`` is the runner-internal ``f"{case_id}::{qa_index}"`` — stable
    across resume. ``question_id`` is the dataset's own id (``q1_abs_42``
    for LongMemEval, numeric row for LOCOMO). Keep both so resume can key
    on ``key`` while the scorecard groups by ``question_type`` /
    ``category``.
    """

    key: str
    benchmark: str
    case_id: str
    question: str
    gold: str
    generated: str
    question_id: str | None = None
    question_type: str | None = None
    category: int | None = None
    qa_index: int | None = None
    replicate_idx: int = 0

    ingestion_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_answer_time_ms: float = 0.0
    answer_discrepancy_ms: float = 0.0

    units_retrieved: int = 0
    tokens_retrieved: int = 0

    evidence_turn_ids: list[str] = field(default_factory=list)
    retrieved_turn_ids: list[str] = field(default_factory=list)

    judge_time_ms: float = 0.0
    judge_runs: list[dict[str, Any]] = field(default_factory=list)

    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class RunManifest:
    """Run-level identity — enough to reproduce the scorecard from scratch.

    Every field flows into ``meta.json`` and (for the subset that also
    belongs in the answer-cache key space) into cache key derivation.
    """

    benchmark: str
    memory_system_id: str
    memory_version: str
    adapter_kind: str
    adapter_target: str
    answer_model_spec: str
    answer_model_resolved: str
    judge_model_spec: str
    judge_model_resolved: str
    judge_temperature: float
    judge_runs: int
    judge_prompt_fingerprint: str
    dataset_name: str
    dataset_split: str | None
    dataset_path: str | None
    dataset_descriptor_hash: str
    hf_revision_sha: str | None
    replicate_idx: int
    replicate_seed: int | None
    benchmark_git_sha: str | None
    benchmark_git_branch: str | None
    benchmark_git_dirty: bool | None
    benchmark_version: str
    protocol_version: str
    tag: str | None
    cli_argv: list[str]
    timestamp_utc: str


@dataclass
class RunDir:
    """Typed wrapper over a run directory. All filenames centralized here."""

    path: Path

    @property
    def answers_path(self) -> Path:
        return self.path / "answers.json"

    @property
    def meta_path(self) -> Path:
        return self.path / "meta.json"

    @property
    def scorecard_json(self) -> Path:
        return self.path / "scorecard.json"

    @property
    def scorecard_md(self) -> Path:
        return self.path / "scorecard.md"


def _dict_to_record(d: dict[str, Any]) -> QARecord:
    known = {f.name for f in fields(QARecord)}
    return QARecord(**{k: v for k, v in d.items() if k in known})


def _dict_to_manifest(d: dict[str, Any]) -> RunManifest:
    known = {f.name for f in fields(RunManifest)}
    return RunManifest(**{k: v for k, v in d.items() if k in known})


def save_run_file(path: Path, manifest: RunManifest, records: list[QARecord]) -> None:
    """Write ``answers.json`` atomically enough for resume semantics.

    The file is rewritten in full on every call (matching the predecessor);
    resume works by reading ``records`` back in and skipping already-present
    ``key``s. Good enough for benchmark-sized runs (~100-2000 Q).
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": asdict(manifest),
        "records": [asdict(r) for r in records],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_run_file(path: Path) -> tuple[RunManifest, dict[str, QARecord]]:
    """Read an ``answers.json`` back into manifest + ``{key: record}`` map."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    manifest = _dict_to_manifest(raw["meta"])
    records = {r["key"]: _dict_to_record(r) for r in raw["records"]}
    return manifest, records


def save_meta_json(path: Path, manifest: RunManifest) -> None:
    """Write the human-readable ``meta.json`` sidecar."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")


_SAFE_COMPONENT = re.compile(r"[^a-zA-Z0-9._-]+")


def sanitize_path_component(value: str) -> str:
    """Collapse non-portable characters into ``-`` for directory names.

    Also normalizes any value that reduces to a dot-only string
    (``"."``, ``".."``, ``"..."``) to ``"unnamed"`` — Windows resolves
    those as relative-path tokens, which would break run-directory
    construction.
    """

    cleaned = _SAFE_COMPONENT.sub("-", value.replace(":", "-").replace("/", "-"))
    stripped = cleaned.strip("-")
    if not stripped or set(stripped) <= {"."}:
        return "unnamed"
    return stripped


def build_run_directory_name(
    *,
    benchmark: str,
    memory_system_id: str,
    answer_model_spec: str,
    timestamp: datetime | None = None,
    tag: str | None = None,
) -> str:
    """Return ``<ts>_<benchmark>_<memsys>_<llm>[_<tag>]``."""

    ts = (timestamp or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    parts = [
        ts,
        sanitize_path_component(benchmark),
        sanitize_path_component(memory_system_id),
        sanitize_path_component(answer_model_spec),
    ]
    if tag:
        parts.append(sanitize_path_component(tag))
    return "_".join(parts)


__all__ = [
    "JudgeRun",
    "QARecord",
    "RunDir",
    "RunManifest",
    "build_run_directory_name",
    "load_run_file",
    "sanitize_path_component",
    "save_meta_json",
    "save_run_file",
]
