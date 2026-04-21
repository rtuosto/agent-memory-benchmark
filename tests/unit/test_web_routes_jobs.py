"""Integration tests for the jobs router.

Subprocess-launching paths are stubbed via monkeypatch so CI doesn't
spawn real processes from a test. The concurrency-cap behavior is
covered in :mod:`test_web_jobs`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent_memory_benchmark.web.app import WebConfig, create_app  # noqa: E402
from agent_memory_benchmark.web.jobs import JobSpec  # noqa: E402


def _client(tmp_path: Path) -> TestClient:
    results = tmp_path / "results"
    jobs = tmp_path / "jobs"
    results.mkdir()
    jobs.mkdir()
    app = create_app(WebConfig(results_dir=results, jobs_dir=jobs))
    return TestClient(app)


def _fake_argv(code: str = "pass", exit_code: int = 0) -> list[str]:
    return [sys.executable, "-c", f"{code}\nimport sys; sys.exit({exit_code})"]


def _wait_job_terminal(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/jobs/{job_id}/logs.json")
        assert resp.status_code == 200
        data = resp.json()
        if data["state"] in {"succeeded", "failed", "killed", "orphaned"}:
            return data
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not terminate within {timeout}s")


def test_jobs_list_empty(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.get("/jobs")
    assert resp.status_code == 200
    assert "No jobs yet" in resp.text


def test_new_job_form_renders_with_defaults(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.get("/jobs/new")
    assert resp.status_code == 200
    # Defaults pre-filled.
    assert 'value="ollama:llama3.1:8b"' in resp.text
    assert 'value="full-context"' in resp.text
    # Submit button is "Review" on first render (not "Confirm").
    assert ">Review<" in resp.text


def test_local_job_submits_and_redirects(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(JobSpec, "to_argv", lambda self: _fake_argv("print('ok')"))
    client = _client(tmp_path)
    resp = client.post(
        "/jobs",
        data={
            "dataset": "longmemeval",
            "memory": "full-context",
            "answer_model": "ollama:llama3.1:8b",
            "judge_model": "ollama:llama3.1:8b",
            "split": "s",
            "limit": "1",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/jobs/")
    job_id = resp.headers["location"].rsplit("/", 1)[-1]
    data = _wait_job_terminal(client, job_id)
    assert data["state"] == "succeeded"


def test_openai_judge_triggers_confirm_flow(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/jobs",
        data={
            "dataset": "longmemeval",
            "memory": "full-context",
            "answer_model": "ollama:llama3.1:8b",
            "judge_model": "openai:gpt-4o-mini",
            "split": "s",
            "limit": "10",
        },
    )
    # Confirm page is a 200 re-render of /jobs/new, NOT a redirect.
    assert resp.status_code == 200
    body = resp.text
    assert "Paid API detected" in body
    assert "Confirm and run" in body
    # Hidden confirmed=yes field is present.
    assert 'name="confirmed"' in body and 'value="yes"' in body
    # Cost breakdown mentions the OpenAI model.
    assert "gpt-4o-mini" in body


def test_openai_answer_also_triggers_confirm(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/jobs",
        data={
            "dataset": "longmemeval",
            "memory": "full-context",
            "answer_model": "openai:gpt-4o",
            "judge_model": "ollama:llama3.1:8b",
            "split": "s",
            "limit": "10",
        },
    )
    assert resp.status_code == 200
    assert "Paid API detected" in resp.text


def test_confirmed_yes_bypasses_second_render(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(JobSpec, "to_argv", lambda self: _fake_argv("print('confirmed')"))
    client = _client(tmp_path)
    resp = client.post(
        "/jobs",
        data={
            "dataset": "longmemeval",
            "memory": "full-context",
            "answer_model": "ollama:llama3.1:8b",
            "judge_model": "openai:gpt-4o-mini",
            "split": "s",
            "limit": "1",
            "confirmed": "yes",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    job_id = resp.headers["location"].rsplit("/", 1)[-1]
    data = _wait_job_terminal(client, job_id)
    assert data["state"] == "succeeded"


def test_validation_errors_re_render_with_400(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/jobs",
        data={
            "dataset": "longmemeval",
            "memory": "",  # missing
            "answer_model": "ollama:llama3.1:8b",
            "judge_model": "ollama:llama3.1:8b",
            "split": "s",
        },
    )
    assert resp.status_code == 400
    assert "memory adapter spec is required" in resp.text


def test_longmemeval_without_split_is_rejected(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/jobs",
        data={
            "dataset": "longmemeval",
            "memory": "full-context",
            "answer_model": "ollama:llama3.1:8b",
            "judge_model": "ollama:llama3.1:8b",
        },
    )
    assert resp.status_code == 400
    assert "split is required for longmemeval" in resp.text


def test_job_detail_404_on_unknown(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.get("/jobs/does-not-exist")
    assert resp.status_code == 404


def test_logs_json_returns_state_and_tail(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        JobSpec, "to_argv", lambda self: _fake_argv("print('hello-world')")
    )
    client = _client(tmp_path)
    resp = client.post(
        "/jobs",
        data={
            "dataset": "longmemeval",
            "memory": "full-context",
            "answer_model": "ollama:llama3.1:8b",
            "judge_model": "ollama:llama3.1:8b",
            "split": "s",
            "limit": "1",
        },
        follow_redirects=False,
    )
    job_id = resp.headers["location"].rsplit("/", 1)[-1]
    data = _wait_job_terminal(client, job_id)
    assert data["state"] == "succeeded"
    assert "hello-world" in data["stdout"]
