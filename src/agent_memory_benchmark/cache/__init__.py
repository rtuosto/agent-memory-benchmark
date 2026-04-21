"""Cache layer — byte-stable keys, index, and publish-time safety guard.

Three layers of cached work:

- **Ingestion** — per-case serialized memory-system state. Only populated
  when an adapter implements the persistable shape.
- **Answer** — per-question generation output, keyed on memory system
  identity + dataset + answer-LLM spec + question + replicate index.
- **Judge** — per-question judge verdict, keyed on benchmark + judge model
  + temperature + runs + prompt fingerprint + question/gold/generated.

Key derivation is documented in :mod:`.keys`. Cache-index bookkeeping is in
:mod:`.index`. The publish-time safety gate is in :mod:`.m3_guard`.
"""

from __future__ import annotations

from .index import (
    CacheIndexWriter,
    clear_all,
    clear_kind,
    gc_older_than,
    index_touch,
    load_index,
)
from .keys import (
    ANSWERS_SUBDIR,
    INDEX_NAME,
    INGESTION_SUBDIR,
    JUDGE_SUBDIR,
    SEPARATOR,
    answer_cache_path,
    answer_key,
    dataset_descriptor_hash,
    dataset_file_hash,
    hash_bytes,
    hash_parts,
    hash_text,
    ingestion_key,
    ingestion_state_path,
    judge_cache_path,
    judge_key,
    prompt_fingerprint,
)
from .m3_guard import M3GuardMismatch, check_answer_cache_versions

__all__ = [
    "ANSWERS_SUBDIR",
    "CacheIndexWriter",
    "INDEX_NAME",
    "INGESTION_SUBDIR",
    "JUDGE_SUBDIR",
    "M3GuardMismatch",
    "SEPARATOR",
    "answer_cache_path",
    "answer_key",
    "check_answer_cache_versions",
    "clear_all",
    "clear_kind",
    "dataset_descriptor_hash",
    "dataset_file_hash",
    "gc_older_than",
    "hash_bytes",
    "hash_parts",
    "hash_text",
    "index_touch",
    "ingestion_key",
    "ingestion_state_path",
    "judge_cache_path",
    "judge_key",
    "load_index",
    "prompt_fingerprint",
]
