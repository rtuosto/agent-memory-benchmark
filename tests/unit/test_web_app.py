"""Smoke tests for the ``amb serve`` stub and ``create_app()`` factory.

Only step 1 is wired up, so coverage here is limited to:

- ``/health`` responds 200 with the configured paths echoed back.
- ``web_deps_available()`` correctly detects the extras.
- The CLI stub surfaces a friendly install hint when the extras are
  missing — exercised by monkeypatching the detector, not by uninstalling
  the deps for real.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent_memory_benchmark.web.app import WebConfig, create_app, web_deps_available


def _config(tmp_path: Path) -> WebConfig:
    results = tmp_path / "results"
    jobs = tmp_path / "jobs"
    results.mkdir()
    jobs.mkdir()
    return WebConfig(results_dir=results, jobs_dir=jobs)


def test_health_returns_config_echo(tmp_path: Path) -> None:
    config = _config(tmp_path)
    client = TestClient(create_app(config))
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["results_dir"] == str(config.results_dir)
    assert body["jobs_dir"] == str(config.jobs_dir)


def test_web_deps_available_when_installed() -> None:
    available, hint = web_deps_available()
    assert available is True
    assert hint is None


def test_serve_command_returns_1_when_deps_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI must print an install hint and exit 1, not crash on import."""

    import argparse

    from agent_memory_benchmark.cli import serve_cmd

    def fake_deps() -> tuple[bool, str | None]:
        return False, "web extras missing (fastapi). Install with: pip install -e \".[web]\""

    monkeypatch.setattr(
        "agent_memory_benchmark.web.app.web_deps_available",
        fake_deps,
    )

    args = argparse.Namespace(
        host="127.0.0.1",
        port=8000,
        results_dir=tmp_path / "results",
        jobs_dir=tmp_path / "jobs",
        max_concurrent=1,
        reload=False,
    )
    rc = serve_cmd.serve_command(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "web extras missing" in err
    assert "pip install" in err


def test_serve_subparser_registered() -> None:
    """``amb serve`` must be reachable from the root parser."""

    from agent_memory_benchmark.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args(["serve", "--port", "9001", "--results-dir", "./results"])
    assert args.command == "serve"
    assert args.port == 9001
    assert str(args.results_dir) == "results"
