"""``amb cache {info|clear|gc}`` — cache directory maintenance.

Wraps :mod:`agent_memory_benchmark.cache.index` behind a user-friendly
surface. None of these actions touch running processes; the cache is
on-disk state only.

- ``amb cache info`` prints a per-kind summary (entry count, bytes on
  disk) plus the index file size. Safe to run at any time.
- ``amb cache clear --kind <k>`` removes one cache subtree. Requires
  ``--yes`` unless the target is empty, so a typo can't wipe a cache a
  run depends on.
- ``amb cache gc --before <age>`` deletes entries older than the given
  duration. Accepts ``7d`` / ``12h`` / ``30m`` or a plain float (days).

Subparser nesting is flat intentionally — ``amb cache info`` /
``amb cache clear`` / ``amb cache gc``. The extra level is worth it
because these share ``--cache-root`` and a common failure mode
(pointing at a directory that isn't actually a cache).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from ..cache.index import clear_all, clear_kind, gc_older_than, load_index
from ..cache.keys import (
    ANSWERS_SUBDIR,
    INDEX_NAME,
    INGESTION_SUBDIR,
    JUDGE_SUBDIR,
)

_KINDS = ("ingestion", "answers", "judge", "all")
_DURATION_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([dhm]?)\s*$", re.IGNORECASE)


def add_cache_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Register ``amb cache`` + its ``info | clear | gc`` sub-actions."""

    parser: argparse.ArgumentParser = subparsers.add_parser(
        "cache",
        help="Inspect or maintain the cache directory.",
        description="Read-only and maintenance commands for the on-disk cache.",
    )
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    sub = parser.add_subparsers(dest="cache_action", metavar="<action>")
    sub.required = True

    sub.add_parser("info", help="Summarize on-disk cache contents.")

    clear = sub.add_parser(
        "clear",
        help="Remove cache entries (all, or one kind).",
    )
    clear.add_argument(
        "--kind",
        choices=_KINDS,
        required=True,
        help="Cache subtree to clear.",
    )
    clear.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt (non-interactive usage).",
    )

    gc = sub.add_parser(
        "gc",
        help="Garbage-collect entries older than a threshold.",
    )
    gc.add_argument(
        "--before",
        required=True,
        metavar="AGE",
        help="Age threshold; accepts e.g. '7d', '12h', '30m', or a float (days).",
    )
    return parser


def cache_command(args: argparse.Namespace, *, argv: list[str] | None = None) -> int:
    """Synchronous CLI entry; returns process exit code."""

    del argv
    cache_root: Path = args.cache_root.resolve()
    action = args.cache_action

    if action == "info":
        return _cache_info(cache_root)
    if action == "clear":
        return _cache_clear(cache_root, kind=args.kind, confirmed=args.yes)
    if action == "gc":
        return _cache_gc(cache_root, before=args.before)
    print(f"error: unknown cache action {action!r}", file=sys.stderr)
    return 2


def _cache_info(cache_root: Path) -> int:
    if not cache_root.is_dir():
        print(f"cache root does not exist: {cache_root}")
        return 0
    print(f"Cache root: {cache_root}")
    print()
    print(f"{'kind':<12} {'entries':>10} {'bytes':>14}")
    print("-" * 38)
    index = load_index(cache_root)
    entries = index.get("entries", {}) if isinstance(index, dict) else {}
    counts: dict[str, int] = {"ingestion": 0, "answers": 0, "judge": 0}
    if isinstance(entries, dict):
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            kind = entry.get("kind")
            if isinstance(kind, str) and kind in counts:
                counts[kind] += 1

    for kind, subdir in (
        ("ingestion", INGESTION_SUBDIR),
        ("answers", ANSWERS_SUBDIR),
        ("judge", JUDGE_SUBDIR),
    ):
        d = cache_root / subdir
        size = _dir_size(d) if d.is_dir() else 0
        print(f"{kind:<12} {counts[kind]:>10} {size:>14}")

    index_path = cache_root / INDEX_NAME
    if index_path.is_file():
        print(f"\nindex file: {index_path.name} ({index_path.stat().st_size} bytes)")
    else:
        print("\nindex file: (missing)")
    return 0


def _cache_clear(cache_root: Path, *, kind: str, confirmed: bool) -> int:
    if not cache_root.is_dir():
        print(f"cache root does not exist: {cache_root}")
        return 0
    if not confirmed:
        print(
            f"refusing to clear {kind!r} at {cache_root} without --yes",
            file=sys.stderr,
        )
        return 2
    if kind == "all":
        clear_all(cache_root)
        print(f"cleared all cache subtrees under {cache_root}")
    else:
        clear_kind(cache_root, kind)
        print(f"cleared {kind!r} cache under {cache_root}")
    return 0


def _cache_gc(cache_root: Path, *, before: str) -> int:
    try:
        days = _parse_duration_days(before)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not cache_root.is_dir():
        print(f"cache root does not exist: {cache_root}")
        return 0
    removed = gc_older_than(cache_root, max_age_days=days)
    print(f"removed {len(removed)} cache entr{'y' if len(removed) == 1 else 'ies'}")
    return 0


def _parse_duration_days(value: str) -> float:
    """Parse ``"7d"`` / ``"12h"`` / ``"30m"`` / ``"1.5"`` → days (float).

    A bare number is interpreted as days so ``--before 7`` is equivalent
    to ``--before 7d`` — matches how the predecessor spelled this and
    keeps muscle memory working.
    """

    m = _DURATION_RE.match(value)
    if m is None:
        raise ValueError(
            f"could not parse duration {value!r}; use e.g. '7d', '12h', '30m', or '1.5'."
        )
    magnitude = float(m.group(1))
    unit = m.group(2).lower()
    if unit in ("", "d"):
        return magnitude
    if unit == "h":
        return magnitude / 24.0
    if unit == "m":
        return magnitude / 1440.0
    raise ValueError(f"unsupported duration unit {unit!r} in {value!r}")


def _dir_size(path: Path) -> int:
    """Sum ``stat().st_size`` over every file under ``path`` (recursive).

    Used by ``amb cache info``; not stricly accurate (ignores filesystem
    block overhead) but matches what users expect from ``du -sb``.
    """

    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total


__all__ = ["add_cache_subparser", "cache_command"]
