"""Tests for the :class:`ResultIndex` cache + tolerant parsing."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from agent_memory_benchmark.web.index import ResultIndex


def _write_run(
    root: Path,
    name: str,
    *,
    scorecard: dict | None = None,
    meta: dict | None = None,
    scorecard_md: str | None = "scorecard body",
) -> Path:
    run = root / name
    run.mkdir()
    if scorecard is not None:
        (run / "scorecard.json").write_text(json.dumps(scorecard), encoding="utf-8")
    if meta is not None:
        (run / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if scorecard_md is not None:
        (run / "scorecard.md").write_text(scorecard_md, encoding="utf-8")
    return run


def _sample_scorecard(**overrides) -> dict:
    base = {
        "benchmark": "longmemeval",
        "n_questions": 30,
        "n_cases": 30,
        "quality": {
            "overall_accuracy": 0.57,
            "macro_accuracy": 0.55,
            "per_category": {
                "single-session-user": {"count": 10, "accuracy": 0.6, "token_f1": 0.5},
            },
        },
        "latency_ms": {
            "ingestion_per_case": {"mean": 100.0, "p50": 90.0, "p95": 200.0, "max": 500.0, "n": 30},
        },
        "throughput": {"queries_per_sec": 0.5, "sessions_per_sec": 0.1},
    }
    base.update(overrides)
    return base


def _sample_meta(**overrides) -> dict:
    base = {
        "benchmark": "longmemeval",
        "memory_system_id": "engram_graph",
        "memory_version": "0.4.0",
        "answer_model_spec": "ollama:llama3.1:8b",
        "judge_model_spec": "ollama:llama3.1:8b",
        "tag": "demo",
    }
    base.update(overrides)
    return base


def test_list_runs_sorted_newest_first(tmp_path: Path) -> None:
    _write_run(tmp_path, "2026-04-20_120000_a", scorecard=_sample_scorecard(), meta=_sample_meta())
    _write_run(tmp_path, "2026-04-21_120000_b", scorecard=_sample_scorecard(), meta=_sample_meta())
    _write_run(tmp_path, "2026-04-19_120000_c", scorecard=_sample_scorecard(), meta=_sample_meta())

    index = ResultIndex(tmp_path)
    runs = index.list_runs()
    assert [r.run_id for r in runs] == [
        "2026-04-21_120000_b",
        "2026-04-20_120000_a",
        "2026-04-19_120000_c",
    ]


def test_list_runs_skips_latest_symlink_name(tmp_path: Path) -> None:
    _write_run(tmp_path, "2026-04-21_120000_a", scorecard=_sample_scorecard(), meta=_sample_meta())
    (tmp_path / "latest").mkdir()  # stand-in for symlink/junction
    (tmp_path / "latest.txt").write_text("2026-04-21_120000_a", encoding="utf-8")

    index = ResultIndex(tmp_path)
    runs = index.list_runs()
    assert len(runs) == 1
    assert runs[0].run_id == "2026-04-21_120000_a"


def test_list_runs_tolerates_partial_data(tmp_path: Path) -> None:
    """No scorecard.json yet (run in progress) shouldn't 500 the list."""

    _write_run(tmp_path, "2026-04-21_120000_a", scorecard=None, meta=_sample_meta())
    index = ResultIndex(tmp_path)
    runs = index.list_runs()
    assert len(runs) == 1
    run = runs[0]
    assert run.complete is False
    assert run.overall_accuracy is None
    assert run.memory_system_id == "engram_graph"


def test_get_run_returns_none_for_traversal(tmp_path: Path) -> None:
    _write_run(tmp_path, "real", scorecard=_sample_scorecard(), meta=_sample_meta())
    index = ResultIndex(tmp_path)
    assert index.get_run("../etc/passwd") is None
    assert index.get_run("../../secret") is None
    assert index.get_run("real/sub") is None


def test_get_run_returns_detail_with_md_text(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "2026-04-21_120000_a",
        scorecard=_sample_scorecard(),
        meta=_sample_meta(),
        scorecard_md="# Scorecard\nbody",
    )
    index = ResultIndex(tmp_path)
    detail = index.get_run("2026-04-21_120000_a")
    assert detail is not None
    assert detail.summary.run_id == "2026-04-21_120000_a"
    assert detail.scorecard_md.startswith("# Scorecard")
    assert detail.scorecard["quality"]["overall_accuracy"] == 0.57


def test_cache_invalidates_on_mtime_bump(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        "2026-04-21_120000_a",
        scorecard=_sample_scorecard(quality={"overall_accuracy": 0.1}),
        meta=_sample_meta(),
    )
    index = ResultIndex(tmp_path)
    first = index.get_run("2026-04-21_120000_a")
    assert first is not None

    time.sleep(0.02)
    (run / "scorecard.json").write_text(
        json.dumps(_sample_scorecard(quality={"overall_accuracy": 0.99})),
        encoding="utf-8",
    )
    # Force a new mtime so the cache invalidates even on fast FS.
    new_mtime = time.time()
    os.utime(run, (new_mtime, new_mtime))

    second = index.get_run("2026-04-21_120000_a")
    assert second is not None
    assert second.scorecard["quality"]["overall_accuracy"] == 0.99


def test_list_empty_when_results_dir_missing(tmp_path: Path) -> None:
    index = ResultIndex(tmp_path / "does-not-exist")
    assert index.list_runs() == []


def test_list_runs_recurses_into_container_dirs(tmp_path: Path) -> None:
    """Container dirs (no run artifacts at the top level, but holding
    a nested run dir) should surface their nested runs — not render
    as an empty row themselves."""

    # A top-level run.
    _write_run(
        tmp_path, "2026-04-21_120000_real", scorecard=_sample_scorecard(), meta=_sample_meta()
    )
    # A container with a nested run one level deep.
    container = tmp_path / "smoke-probe"
    container.mkdir()
    _write_run(
        container,
        "2026-04-21_130000_nested",
        scorecard=_sample_scorecard(),
        meta=_sample_meta(),
    )
    # An empty container dir — should not appear at all.
    (tmp_path / "truly-empty").mkdir()

    runs = ResultIndex(tmp_path).list_runs()
    ids = sorted(r.run_id for r in runs)
    assert ids == [
        "2026-04-21_120000_real",
        "smoke-probe/2026-04-21_130000_nested",
    ]


def test_timestamp_falls_back_to_mtime_for_unnamed_runs(tmp_path: Path) -> None:
    """Dirs without a ``YYYY-MM-DD_HHMMSS_...`` prefix should still get a
    When value so the UI doesn't show a raw slug in the datetime column."""

    _write_run(
        tmp_path,
        "smoke-probe",
        scorecard=_sample_scorecard(),
        meta=_sample_meta(),
    )
    runs = ResultIndex(tmp_path).list_runs()
    assert len(runs) == 1
    ts = runs[0].timestamp
    assert ts is not None
    # mtime-derived stamps emit as UTC ISO with Z so the browser can localize.
    assert ts.endswith("Z")
    assert "T" in ts
