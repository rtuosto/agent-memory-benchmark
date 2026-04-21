"""M3 publish-time safety gate.

The M3 guard refuses to publish a ``--publishable`` run if any cached
answer entry's stored ``memory_version`` differs from the adapter's
currently-reported value. The cache key already includes
``memory_version``, so a version bump naturally invalidates cache by
*cache miss*; this guard is a second line of defense against hand-edited
or otherwise stale entries that would land under a post-bump hash.

The guard reads only the answer-cache JSON files. It does not inspect
judge or ingestion caches: judge keys are keyed on the prompt fingerprint
(so prompt drift is caught by the fingerprint test), and ingestion state
is invalidated by the same ``memory_version`` input as answers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .keys import ANSWERS_SUBDIR


@dataclass(frozen=True)
class M3GuardMismatch:
    """One offending answer-cache entry.

    ``path`` is the JSON file whose stored ``memory_version`` did not
    match; ``found`` is the stored value; ``unreadable`` is True when the
    file existed but could not be parsed as JSON.
    """

    path: Path
    found: str | None
    unreadable: bool = False


def check_answer_cache_versions(
    cache_root: Path,
    *,
    memory_system_id: str,
    expected_memory_version: str,
) -> list[M3GuardMismatch]:
    """Walk ``<cache_root>/answers`` and return any mismatches.

    Answer-cache files that do not belong to this memory system (checked
    via the stored ``memory_system_id`` field) are skipped. An empty list
    means the cache is safe to publish.
    """

    answers_dir = cache_root / ANSWERS_SUBDIR
    if not answers_dir.is_dir():
        return []
    mismatches: list[M3GuardMismatch] = []
    for entry_path in sorted(answers_dir.glob("*.json")):
        try:
            payload = json.loads(entry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            mismatches.append(
                M3GuardMismatch(path=entry_path, found=None, unreadable=True)
            )
            continue
        if not isinstance(payload, dict):
            mismatches.append(
                M3GuardMismatch(path=entry_path, found=None, unreadable=True)
            )
            continue
        stored_id = payload.get("memory_system_id")
        if stored_id != memory_system_id:
            # Entry belongs to another adapter; not our concern.
            continue
        stored_version = payload.get("memory_version")
        if stored_version != expected_memory_version:
            found = stored_version if isinstance(stored_version, str) else None
            mismatches.append(M3GuardMismatch(path=entry_path, found=found))
    return mismatches


__all__ = ["M3GuardMismatch", "check_answer_cache_versions"]
