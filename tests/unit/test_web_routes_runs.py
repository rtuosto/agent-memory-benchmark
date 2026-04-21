"""Integration tests for the runs router."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent_memory_benchmark.web.app import WebConfig, create_app  # noqa: E402


def _seed_run(root: Path, name: str) -> None:
    run = root / name
    run.mkdir()
    (run / "scorecard.json").write_text(
        json.dumps(
            {
                "benchmark": "longmemeval",
                "n_questions": 30,
                "n_cases": 30,
                "quality": {
                    "overall_accuracy": 0.57,
                    "macro_accuracy": 0.55,
                    "per_category": {
                        "single-session-user": {
                            "count": 10,
                            "accuracy": 0.6,
                            "token_f1": 0.5,
                        }
                    },
                },
                "latency_ms": {
                    "ingestion_per_case": {
                        "mean": 100.0,
                        "p50": 90.0,
                        "p95": 200.0,
                        "max": 500.0,
                        "n": 30,
                    },
                    "retrieval_per_query": {
                        "mean": 5.0,
                        "p50": 4.0,
                        "p95": 9.0,
                        "max": 12.0,
                        "n": 30,
                    },
                },
                "retrieval_footprint": {
                    "units_per_query": {"mean": 10, "p50": 10, "p95": 15, "max": 20}
                },
                "throughput": {"queries_per_sec": 0.5, "sessions_per_sec": 0.1},
            }
        ),
        encoding="utf-8",
    )
    (run / "meta.json").write_text(
        json.dumps(
            {
                "benchmark": "longmemeval",
                "memory_system_id": "engram_graph",
                "memory_version": "0.4.0",
                "answer_model_spec": "ollama:llama3.1:8b",
                "judge_model_spec": "ollama:llama3.1:8b",
                "tag": "demo",
            }
        ),
        encoding="utf-8",
    )
    (run / "scorecard.md").write_text("# Scorecard — longmemeval\n\nbody", encoding="utf-8")


def _client(tmp_path: Path) -> TestClient:
    results = tmp_path / "results"
    jobs = tmp_path / "jobs"
    results.mkdir()
    jobs.mkdir()
    _seed_run(results, "2026-04-21_120000_demo")
    app = create_app(WebConfig(results_dir=results, jobs_dir=jobs))
    return TestClient(app)


def test_root_redirects_to_runs(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/runs"


def test_runs_list_contains_seeded_run(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/runs")
    assert response.status_code == 200
    assert "2026-04-21_120000_demo" in response.text
    assert "engram_graph" in response.text
    assert "57.00%" in response.text  # overall_accuracy rendered


def test_run_detail_renders_kpis_and_chart_data(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/runs/2026-04-21_120000_demo")
    assert response.status_code == 200
    body = response.text
    # KPI cards rendered.
    assert "55.00%" in body  # macro_accuracy
    assert "# Scorecard" in body  # scorecard.md embedded
    # Canvas elements exist for each chart section.
    assert 'id="chart-latency"' in body
    assert 'id="chart-per-category"' in body
    # Chart data is embedded as JSON — parse it back out and verify shape.
    import json
    import re

    match = re.search(
        r'<script id="chart-data" type="application/json">(.*?)</script>',
        body,
        re.DOTALL,
    )
    assert match is not None
    chart_data = json.loads(match.group(1))
    assert chart_data["per_category"]
    assert any(row["name"] == "ingestion_per_case" for row in chart_data["latency"])


def test_run_detail_404s_on_unknown_run(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/runs/does-not-exist")
    assert response.status_code == 404


def test_run_detail_rejects_traversal(tmp_path: Path) -> None:
    client = _client(tmp_path)
    # FastAPI normalizes ../ so the URL resolves differently; the
    # per-run route's own traversal guard handles explicit attempts.
    response = client.get("/runs/..%2Fetc")
    assert response.status_code in (400, 404)


def test_scorecard_json_passthrough(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/runs/2026-04-21_120000_demo/scorecard.json")
    assert response.status_code == 200
    data = response.json()
    assert data["quality"]["overall_accuracy"] == 0.57
