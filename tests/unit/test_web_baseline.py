"""Tests for baseline selection and comparison on the run detail page."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent_memory_benchmark.web.app import WebConfig, create_app  # noqa: E402
from agent_memory_benchmark.web.index import ResultIndex  # noqa: E402


def _write(
    root: Path,
    name: str,
    *,
    benchmark: str = "longmemeval",
    overall: float | None = 0.5,
    memory: str = "engram_graph",
) -> None:
    run = root / name
    run.mkdir()
    (run / "scorecard.json").write_text(
        json.dumps(
            {
                "benchmark": benchmark,
                "n_questions": 10,
                "quality": {
                    "overall_accuracy": overall,
                    "macro_accuracy": overall,
                    "per_category": {},
                },
                "latency_ms": {
                    "ingestion_total": 20000.0,
                    "ingestion_per_case": {"mean": 1000.0, "p95": 2000.0, "n": 10},
                    "retrieval_per_query": {"mean": 10.0},
                },
                "throughput": {"queries_per_sec": 0.5, "sessions_per_sec": 0.1},
                "retrieval_footprint": {},
                "evidence": None,
            }
        ),
        encoding="utf-8",
    )
    (run / "meta.json").write_text(
        json.dumps(
            {
                "benchmark": benchmark,
                "memory_system_id": memory,
                "memory_version": "1.0",
                "answer_model_spec": "ollama:llama3.1:8b",
                "judge_model_spec": "ollama:llama3.1:8b",
            }
        ),
        encoding="utf-8",
    )
    (run / "scorecard.md").write_text("# scorecard", encoding="utf-8")


def test_best_baseline_picks_highest_accuracy_for_same_benchmark(tmp_path: Path) -> None:
    _write(tmp_path, "2026-04-20_120000_a", overall=0.40)
    _write(tmp_path, "2026-04-21_120000_b", overall=0.75)
    _write(tmp_path, "2026-04-22_120000_c", overall=0.60)
    index = ResultIndex(tmp_path)
    best = index.best_baseline(benchmark="longmemeval", exclude_run_id="2026-04-22_120000_c")
    assert best is not None
    assert best.run_id == "2026-04-21_120000_b"


def test_best_baseline_excludes_different_benchmarks(tmp_path: Path) -> None:
    _write(tmp_path, "2026-04-20_120000_a", benchmark="longmemeval", overall=0.6)
    _write(tmp_path, "2026-04-21_120000_b", benchmark="locomo", overall=0.99)
    index = ResultIndex(tmp_path)
    best = index.best_baseline(benchmark="longmemeval", exclude_run_id="2026-04-20_120000_a")
    assert best is None


def test_list_candidates_excludes_self(tmp_path: Path) -> None:
    _write(tmp_path, "2026-04-20_120000_a", overall=0.5)
    _write(tmp_path, "2026-04-21_120000_b", overall=0.6)
    index = ResultIndex(tmp_path)
    ids = [s.run_id for s in index.list_candidates(benchmark="longmemeval", exclude_run_id="2026-04-21_120000_b")]
    assert ids == ["2026-04-20_120000_a"]


def _client_with_three_runs(tmp_path: Path) -> TestClient:
    results = tmp_path / "results"
    jobs = tmp_path / "jobs"
    results.mkdir()
    jobs.mkdir()
    _write(results, "2026-04-20_120000_low", overall=0.30)
    _write(results, "2026-04-21_120000_high", overall=0.80)
    _write(results, "2026-04-22_120000_mid", overall=0.55)
    return TestClient(create_app(WebConfig(results_dir=results, jobs_dir=jobs)))


def test_detail_page_renders_auto_baseline(tmp_path: Path) -> None:
    client = _client_with_three_runs(tmp_path)
    response = client.get("/runs/2026-04-22_120000_mid")
    assert response.status_code == 200
    body = response.text
    # Auto-baseline should be the "_high" run (0.80 overall, same benchmark).
    assert "2026-04-21_120000_high" in body
    # Delta is B - A = 0.55 - 0.80 = -0.25 → "-25.00pp".
    assert "-25.00pp" in body


def test_detail_page_baseline_manual_override(tmp_path: Path) -> None:
    client = _client_with_three_runs(tmp_path)
    response = client.get(
        "/runs/2026-04-22_120000_mid?baseline=2026-04-20_120000_low"
    )
    assert response.status_code == 200
    body = response.text
    # Delta is now 0.55 - 0.30 = +0.25 → "+25.00pp".
    assert "+25.00pp" in body


def test_detail_page_baseline_none_disables_compare(tmp_path: Path) -> None:
    client = _client_with_three_runs(tmp_path)
    response = client.get("/runs/2026-04-22_120000_mid?baseline=none")
    assert response.status_code == 200
    body = response.text
    # No delta badges when user selects "none".
    assert "pp" not in body or "vs. baseline" not in body
    assert "vs. baseline" not in body


def test_ingestion_section_renders(tmp_path: Path) -> None:
    client = _client_with_three_runs(tmp_path)
    response = client.get("/runs/2026-04-22_120000_mid?baseline=none")
    body = response.text
    assert "Ingest total" in body
    assert "Per-case mean" in body
    assert "Sessions / sec" in body
