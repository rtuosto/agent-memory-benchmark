"""Cache index bookkeeping and directory maintenance.

The cache index is a small JSON file at ``<cache_root>/cache_index.json``
that tracks which key corresponds to which on-disk file, when it was
touched, and any caller-provided metadata. It is maintained on a
best-effort basis: losing or corrupting it never corrupts the actual
cache entries, only the ability to ``amb cache info`` / ``amb cache gc``.

Layout:

.. code-block:: json

    {
      "version": 1,
      "entries": {
        "<key>": {
          "kind": "ingestion | answers | judge",
          "path": "relative/or/absolute/path",
          "updated": "2026-04-20T12:34:56Z",
          "meta": {"arbitrary": "caller-supplied"}
        }
      }
    }
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from types import TracebackType
from typing import Any

from .keys import ANSWERS_SUBDIR, INDEX_NAME, INGESTION_SUBDIR, JUDGE_SUBDIR

_KIND_TO_SUBDIR = {
    "ingestion": INGESTION_SUBDIR,
    "answers": ANSWERS_SUBDIR,
    "judge": JUDGE_SUBDIR,
}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_index(cache_root: Path) -> dict[str, Any]:
    """Load the cache index or return a fresh skeleton if missing/corrupt.

    A corrupt (non-JSON) index file silently yields a fresh empty index —
    we never fail a run because of index bookkeeping.
    """

    path = cache_root / INDEX_NAME
    if not path.is_file():
        return {"version": 1, "entries": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"version": 1, "entries": {}}
    if not isinstance(data, dict):
        return {"version": 1, "entries": {}}
    data.setdefault("version", 1)
    data.setdefault("entries", {})
    return data


def _save_index(cache_root: Path, data: dict[str, Any]) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / INDEX_NAME).write_text(
        json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
    )


def index_touch(
    cache_root: Path,
    *,
    kind: str,
    key: str,
    path: str,
    meta: dict[str, Any] | None = None,
) -> None:
    """Upsert a single entry in the cache index."""

    data = load_index(cache_root)
    entries: dict[str, Any] = data.setdefault("entries", {})
    entries[key] = {
        "kind": kind,
        "path": path,
        "updated": _now_iso(),
        "meta": meta or {},
    }
    _save_index(cache_root, data)


class CacheIndexWriter:
    """Batch updates to ``cache_index.json``; flush on ``__exit__`` or ``flush()``.

    Useful inside the runner's per-query loop so we rewrite the index file
    once per batch rather than once per entry. Pending entries are held in
    memory; a process crash between touches and flush loses those updates
    but never the underlying cache files.
    """

    def __init__(self, cache_root: Path) -> None:
        self._cache_root = cache_root
        self._pending: list[tuple[str, str, str, dict[str, Any]]] = []

    def touch(
        self,
        *,
        kind: str,
        key: str,
        path: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self._pending.append((kind, key, path, meta or {}))

    def flush(self) -> None:
        if not self._pending:
            return
        data = load_index(self._cache_root)
        entries: dict[str, Any] = data.setdefault("entries", {})
        for kind, key, path, meta in self._pending:
            entries[key] = {
                "kind": kind,
                "path": path,
                "updated": _now_iso(),
                "meta": meta,
            }
        self._pending.clear()
        _save_index(self._cache_root, data)

    def __enter__(self) -> CacheIndexWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.flush()


def clear_all(cache_root: Path) -> None:
    """Remove every cache subdir and the index file.

    Idempotent: succeeds when ``cache_root`` does not exist.
    """

    if not cache_root.is_dir():
        return
    for child in cache_root.iterdir():
        if child.name == INDEX_NAME:
            child.unlink(missing_ok=True)
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def clear_kind(cache_root: Path, kind: str) -> None:
    """Remove one cache kind (``"ingestion" | "answers" | "judge"``).

    Unknown kinds are silently ignored to match the predecessor's
    permissive surface.
    """

    subdir = _KIND_TO_SUBDIR.get(kind)
    if subdir is None:
        return
    target = cache_root / subdir
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
    data = load_index(cache_root)
    entries: dict[str, Any] = data.get("entries", {})
    stale_keys = [k for k, v in entries.items() if v.get("kind") == kind]
    for k in stale_keys:
        del entries[k]
    _save_index(cache_root, data)


def gc_older_than(cache_root: Path, *, max_age_days: float) -> list[str]:
    """Delete cache entries whose ``updated`` timestamp is older than ``max_age_days``.

    Walks the index (which is the source of truth for "what files does the
    cache own"). Returns the list of removed keys. Entries without a valid
    timestamp are left alone.
    """

    if max_age_days < 0:
        raise ValueError(f"max_age_days must be >= 0, got {max_age_days!r}")
    cutoff_epoch = time.time() - max_age_days * 86400.0
    data = load_index(cache_root)
    entries: dict[str, Any] = data.get("entries", {})
    removed: list[str] = []
    for key, entry in list(entries.items()):
        ts = entry.get("updated")
        if not isinstance(ts, str):
            continue
        try:
            # Parse the ISO8601 Z-suffixed timestamp we always write.
            entry_epoch = time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"))
            # ``time.mktime`` interprets the struct as local time; subtract the
            # local UTC offset so the comparison is UTC-on-both-sides.
            entry_epoch -= time.timezone
        except (ValueError, OverflowError):
            continue
        if entry_epoch >= cutoff_epoch:
            continue
        path_str = entry.get("path")
        if isinstance(path_str, str):
            p = Path(path_str)
            if not p.is_absolute():
                p = cache_root / p
            if p.is_file():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        del entries[key]
        removed.append(key)
    _save_index(cache_root, data)
    return removed


__all__ = [
    "CacheIndexWriter",
    "clear_all",
    "clear_kind",
    "gc_older_than",
    "index_touch",
    "load_index",
]
