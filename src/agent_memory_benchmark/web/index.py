"""Scan ``results/`` and expose summaries + details of past runs.

The filesystem is the source of truth — no DB, no background indexer.
``ResultIndex.list_runs()`` walks ``results/*/scorecard.json`` on every
call but caches parsed payloads keyed by dir mtime, so a dashboard
reload on a 200-run history is still cheap (one ``stat`` per dir, plus
fresh JSON reads only for dirs that changed).

Tolerant by design: partially-written runs (no ``scorecard.json`` yet,
crashed before ``meta.json``) and pre-rename runs (``ingestion_per_session``
key) are rendered with best-effort placeholders instead of 500-ing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunSummary:
    """One row in the runs list — cheap to compute, safe on partial data."""

    run_id: str
    path: Path
    timestamp: str | None
    benchmark: str | None
    memory_system_id: str | None
    memory_version: str | None
    answer_model: str | None
    judge_model: str | None
    tag: str | None
    n_questions: int | None
    overall_accuracy: float | None
    macro_accuracy: float | None
    throughput_qps: float | None
    complete: bool


@dataclass(frozen=True)
class RunDetail:
    """Full rendering payload for one run — raw JSON + text for templates."""

    summary: RunSummary
    scorecard: dict[str, Any]
    meta: dict[str, Any]
    scorecard_md: str


@dataclass
class _CacheEntry:
    mtime_ns: int
    summary: RunSummary
    scorecard: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    scorecard_md: str = ""


class ResultIndex:
    """Cached, mtime-invalidated view over a ``results/`` directory."""

    def __init__(self, results_dir: Path) -> None:
        self._results_dir = results_dir
        self._cache: dict[str, _CacheEntry] = {}

    @property
    def results_dir(self) -> Path:
        return self._results_dir

    def list_runs(self) -> list[RunSummary]:
        """Return summaries sorted newest-first by directory name.

        Directory names start with an ISO-ish timestamp
        (``YYYY-MM-DD_HHMMSS_...``), so lexical descending equals
        chronological descending without parsing dates.
        """

        if not self._results_dir.exists():
            return []
        summaries: list[RunSummary] = []
        for run_path in self._discover_run_dirs():
            entry = self._get_entry(run_path)
            if entry is None:
                continue
            summaries.append(entry.summary)
        summaries.sort(key=lambda s: s.run_id, reverse=True)
        return summaries

    def _discover_run_dirs(self) -> list[Path]:
        """Yield every run directory under ``results_dir`` (max depth 2).

        A dir counts as a run dir when it contains ``scorecard.json``,
        ``meta.json``, or ``answers.json`` at its top level. Container
        dirs (e.g. ``smoke-probe/`` holding a nested dated run) are
        skipped themselves but recursed into — one level deep — so the
        nested run surfaces. Two levels is the depth users actually
        organize by; deeper nesting isn't an observed pattern and
        would slow down repos with large ``results/``.
        """

        discovered: list[Path] = []
        for child in self._results_dir.iterdir():
            if not child.is_dir():
                continue
            if child.name in {"latest", "latest.txt"}:
                continue
            if _is_run_dir(child):
                discovered.append(child)
                continue
            # Container — peek one level deeper for nested runs.
            try:
                for grandchild in child.iterdir():
                    if not grandchild.is_dir():
                        continue
                    if grandchild.name in {"latest", "latest.txt"}:
                        continue
                    if _is_run_dir(grandchild):
                        discovered.append(grandchild)
            except OSError:
                continue
        return discovered

    def best_baseline(
        self,
        *,
        benchmark: str | None,
        exclude_run_id: str | None = None,
    ) -> RunSummary | None:
        """Return the same-benchmark run with the highest ``overall_accuracy``.

        Self-exclusion lets callers drop the currently-viewed run without
        extra filtering downstream. Returns ``None`` when no other
        complete run exists for ``benchmark``; the detail view then just
        skips rendering the baseline comparison.
        """

        if benchmark is None:
            return None
        best: RunSummary | None = None
        for summary in self.list_runs():
            if summary.run_id == exclude_run_id:
                continue
            if summary.benchmark != benchmark:
                continue
            if summary.overall_accuracy is None:
                continue
            if best is None or summary.overall_accuracy > (best.overall_accuracy or 0):
                best = summary
        return best

    def list_candidates(
        self,
        *,
        benchmark: str | None,
        exclude_run_id: str | None = None,
    ) -> list[RunSummary]:
        """Runs a user can pick as a baseline — same-benchmark, not self."""

        if benchmark is None:
            return []
        return [
            summary
            for summary in self.list_runs()
            if summary.benchmark == benchmark and summary.run_id != exclude_run_id
        ]

    def get_run(self, run_id: str) -> RunDetail | None:
        """Return the full detail for ``run_id`` or ``None`` if unknown.

        ``run_id`` may contain forward-slashes for nested runs (e.g.
        ``engram-minilm-batched-100q/2026-04-21_061014_...``). The
        resolve+relative_to check still pins the target inside
        ``results_dir`` — traversal attempts (``..`` segments, absolute
        paths) are rejected.
        """

        if not run_id or run_id.startswith(("/", "\\")):
            return None
        # Reject Windows drive letters and any ``..`` segment.
        parts = run_id.replace("\\", "/").split("/")
        if any(p in ("..", "") for p in parts):
            return None
        path = self._results_dir / run_id
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            return None
        if not resolved.is_dir():
            return None
        if not resolved.is_relative_to(self._results_dir.resolve()):
            return None
        entry = self._get_entry(path)
        if entry is None:
            return None
        return RunDetail(
            summary=entry.summary,
            scorecard=entry.scorecard,
            meta=entry.meta,
            scorecard_md=entry.scorecard_md,
        )

    def _get_entry(self, path: Path) -> _CacheEntry | None:
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            return None
        run_id = self._run_id_for(path)
        cached = self._cache.get(run_id)
        if cached is not None and cached.mtime_ns == mtime_ns:
            return cached
        entry = self._build_entry(path, run_id, mtime_ns)
        self._cache[run_id] = entry
        return entry

    def _run_id_for(self, path: Path) -> str:
        """Return the run_id used in URLs — the path relative to
        ``results_dir``, with forward-slash separators so it round-trips
        cleanly through ``{run_id:path}`` route params.
        """

        rel = path.relative_to(self._results_dir)
        return rel.as_posix()

    def _build_entry(self, path: Path, run_id: str, mtime_ns: int) -> _CacheEntry:
        scorecard = _read_json(path / "scorecard.json")
        meta = _read_json(path / "meta.json")
        scorecard_md = _read_text(path / "scorecard.md")
        summary = _summarize(path, run_id, scorecard, meta)
        return _CacheEntry(
            mtime_ns=mtime_ns,
            summary=summary,
            scorecard=scorecard,
            meta=meta,
            scorecard_md=scorecard_md,
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _summarize(
    path: Path, run_id: str, scorecard: dict[str, Any], meta: dict[str, Any]
) -> RunSummary:
    quality = scorecard.get("quality", {}) if isinstance(scorecard, dict) else {}
    throughput = scorecard.get("throughput", {}) if isinstance(scorecard, dict) else {}
    timestamp = _extract_timestamp(path.name) or _mtime_iso(path)
    return RunSummary(
        run_id=run_id,
        path=path,
        timestamp=timestamp,
        benchmark=_str_or_none(scorecard.get("benchmark") or meta.get("benchmark")),
        memory_system_id=_str_or_none(meta.get("memory_system_id")),
        memory_version=_str_or_none(meta.get("memory_version")),
        answer_model=_str_or_none(meta.get("answer_model_spec")),
        judge_model=_str_or_none(meta.get("judge_model_spec")),
        tag=_str_or_none(meta.get("tag")),
        n_questions=_int_or_none(scorecard.get("n_questions")),
        overall_accuracy=_float_or_none(quality.get("overall_accuracy")),
        macro_accuracy=_float_or_none(quality.get("macro_accuracy")),
        throughput_qps=_float_or_none(throughput.get("queries_per_sec")),
        complete=bool(scorecard) and bool(meta),
    )


def _is_run_dir(path: Path) -> bool:
    """Return True if the dir holds run artifacts, not just nested dirs.

    The runner writes ``scorecard.json``, ``meta.json`` and
    ``answers.json`` at the top level of every run dir. The presence of
    any one is enough: an in-progress run may only have
    ``answers.json`` written so far. Absence of all three means this is
    a user-curated container (e.g. ``smoke-probe/`` holding a nested
    dated run inside), which shouldn't surface as its own row.
    """

    return any(
        (path / name).is_file()
        for name in ("scorecard.json", "meta.json", "answers.json")
    )


def _mtime_iso(path: Path) -> str | None:
    """Fallback timestamp for runs whose dir name has no ``YYYY-MM-DD_HHMMSS`` prefix.

    Legacy / hand-named dirs (e.g. ``smoke-probe``, ``lme-3b-30q``)
    still deserve a When column that renders as a datetime rather than
    the raw slug. We use the directory mtime — close enough to
    "created" for a never-rewritten scorecard output dir, and always
    available. Emitted as UTC ISO so the browser-side localizer does
    the right thing.
    """

    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(mtime, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_timestamp(run_id: str) -> str | None:
    # Run dir names look like 2026-04-21_044240_<benchmark>_<memory>_<model>_<tag>
    parts = run_id.split("_", 2)
    if len(parts) < 2:
        return None
    date_part, time_part = parts[0], parts[1]
    if len(date_part) != 10 or len(time_part) != 6:
        return None
    try:
        # Emit ISO-8601 with a ``Z`` so the browser can parse as UTC and
        # convert to the viewer's local TZ via ``<time datetime="...">``.
        # The runner writes run directories in UTC (cf.
        # ``runner/results.py``), so the naive parse is safe.
        dt = datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H%M%S")
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


__all__ = ["ResultIndex", "RunDetail", "RunSummary"]
