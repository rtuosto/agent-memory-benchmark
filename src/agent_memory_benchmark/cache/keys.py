"""Byte-stable cache keys.

The hash inputs and the ``\\x1e`` (ASCII record separator) framing are
regression-locked by ``tests/unit/test_cache_keys.py`` — any change to the
algorithm, the ordering of parts, or the string representations of numeric
fields MUST bump a migration note and produce new golden digests.

The predecessor benchmark (``~/code/agent-memory/benchmark/cache.py``)
contributed the shape of these keys so historical scorecards remain
comparable across repos for the inputs both codebases share. The only
deliberate divergence from the predecessor is the optional ``replicate_idx``
trailing part on :func:`answer_key`, which is omitted when
``replicate_idx == 0`` so single-run hashes remain byte-exact with the
predecessor.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

SEPARATOR = b"\x1e"
"""ASCII record separator byte used between hash-input parts."""

INGESTION_SUBDIR = "ingestion"
ANSWERS_SUBDIR = "answers"
JUDGE_SUBDIR = "judge"
INDEX_NAME = "cache_index.json"


def hash_parts(parts: tuple[str, ...]) -> str:
    """Hash a tuple of strings with ``\\x1e`` between parts.

    Each part is UTF-8-encoded, then a single separator byte is appended.
    The trailing separator after the final part is intentional: it lets
    empty tuples and tuples ending in empty strings produce distinct
    digests, and it matches the predecessor's algorithm byte-for-byte.
    """

    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(SEPARATOR)
    return h.hexdigest()


def hash_bytes(data: bytes) -> str:
    """SHA-256 hex digest of a byte string."""

    return hashlib.sha256(data).hexdigest()


def hash_text(text: str) -> str:
    """SHA-256 hex digest of a UTF-8 text string (no separator framing)."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dataset_file_hash(path: Path) -> str:
    """Hash of a dataset backed by a single local file (LOCOMO, LME-M)."""

    return hash_bytes(path.read_bytes())


def dataset_descriptor_hash(parts: tuple[str, ...]) -> str:
    """Hash for datasets not backed by a single file.

    Typically the tuple is ``(dataset_name, split, hf_revision_sha,
    limit_sig)`` — whatever uniquely identifies the slice the runner is
    iterating over. The exact tuple is the caller's choice; this function
    only frames it.
    """

    return hash_parts(parts)


def prompt_fingerprint(text: str) -> str:
    """SHA-256 fingerprint of a judge prompt template.

    Used as one input to :func:`judge_key`; also stored in the run manifest
    so a mismatch between a stored-answer fingerprint and the current
    judge-prompt bytes invalidates cached verdicts.
    """

    return hash_text(text)


def ingestion_key(
    memory_system_id: str,
    memory_version: str,
    dataset_descriptor_hash: str,
    case_id: str,
) -> str:
    """Cache key for an ingested-state snapshot.

    Parts order matches the predecessor benchmark exactly.
    """

    return hash_parts((memory_system_id, memory_version, dataset_descriptor_hash, case_id))


def answer_key(
    memory_system_id: str,
    memory_version: str,
    dataset_descriptor_hash: str,
    answer_llm_spec: str,
    question_key: str,
    question_text: str,
    *,
    replicate_idx: int = 0,
) -> str:
    """Cache key for a single generated answer.

    ``replicate_idx`` distinguishes multiple noise-aware replicates of the
    same (question, memory, model) triple. When ``replicate_idx == 0`` the
    field is omitted from the hash input so single-run digests remain
    byte-exact with the predecessor benchmark.

    ``question_key`` is typically the dataset's stable question id;
    ``question_text`` is the verbatim question. Both are included so that
    accidental reuse of an id across rewrites still invalidates cache.
    """

    if replicate_idx < 0:
        raise ValueError(f"replicate_idx must be non-negative, got {replicate_idx!r}")
    parts: tuple[str, ...] = (
        memory_system_id,
        memory_version,
        dataset_descriptor_hash,
        answer_llm_spec,
        question_key,
        question_text,
    )
    if replicate_idx != 0:
        parts = (*parts, str(replicate_idx))
    return hash_parts(parts)


def judge_key(
    benchmark_name: str,
    judge_model_spec: str,
    judge_temperature: float,
    judge_runs: int,
    judge_prompt_fingerprint: str,
    question: str,
    gold: str,
    generated: str,
    *,
    question_type: str | None = None,
    question_id: str | None = None,
) -> str:
    """Cache key for a judge verdict over a (question, gold, generated) triple.

    Matches the predecessor byte-for-byte: temperature is formatted with
    ``:.6f``, ``judge_runs`` is stringified with ``str()``, and the trailing
    ``question_type|question_id`` field preserves empty-string framing when
    either or both are ``None``.
    """

    extra = f"{question_type or ''}|{question_id or ''}"
    return hash_parts(
        (
            benchmark_name,
            judge_model_spec,
            f"{judge_temperature:.6f}",
            str(judge_runs),
            judge_prompt_fingerprint,
            question,
            gold,
            generated,
            extra,
        )
    )


def ingestion_state_path(cache_root: Path, memory_system_id: str, key: str) -> Path:
    """Return ``<root>/ingestion/<memory_system_id>/<key>/state.json``.

    Slashes in ``memory_system_id`` are replaced with underscores so the
    path stays a single directory level.
    """

    safe_id = memory_system_id.replace("/", "_")
    return cache_root / INGESTION_SUBDIR / safe_id / key / "state.json"


def answer_cache_path(cache_root: Path, key: str) -> Path:
    """Return ``<root>/answers/<key>.json``."""

    return cache_root / ANSWERS_SUBDIR / f"{key}.json"


def judge_cache_path(cache_root: Path, key: str) -> Path:
    """Return ``<root>/judge/<key>.json``."""

    return cache_root / JUDGE_SUBDIR / f"{key}.json"


__all__ = [
    "ANSWERS_SUBDIR",
    "INDEX_NAME",
    "INGESTION_SUBDIR",
    "JUDGE_SUBDIR",
    "SEPARATOR",
    "answer_cache_path",
    "answer_key",
    "dataset_descriptor_hash",
    "dataset_file_hash",
    "hash_bytes",
    "hash_parts",
    "hash_text",
    "ingestion_key",
    "ingestion_state_path",
    "judge_cache_path",
    "judge_key",
    "prompt_fingerprint",
]
