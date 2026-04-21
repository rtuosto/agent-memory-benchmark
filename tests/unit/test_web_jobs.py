"""Unit tests for :class:`JobManager`.

These exercise subprocess launch + state transitions using tiny
python one-liners so the tests don't depend on the real ``amb run``
entry point. The :class:`JobSpec` ``argv`` helper is verified in
isolation so the form → argv path is covered without running
subprocesses.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from agent_memory_benchmark.web.jobs import JobManager, JobRecord, JobSpec


def _wait_for_state(
    manager: JobManager,
    job_id: str,
    target: set[str],
    *,
    timeout: float = 15.0,
) -> JobRecord:
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = manager.get(job_id)
        if record and record.state in target:
            return record
        time.sleep(0.05)
    record = manager.get(job_id)
    raise AssertionError(
        f"job {job_id} did not reach {target} within {timeout}s; last={record}"
    )


def _python_argv(code: str, exit_code: int = 0) -> list[str]:
    """Argv that runs a short python snippet and exits with ``exit_code``."""

    return [sys.executable, "-c", f"{code}\nimport sys; sys.exit({exit_code})"]


class _FakeSpec(JobSpec):
    """Test helper — bypasses ``to_argv()`` via monkeypatching."""

    pass


def _spec(**overrides) -> JobSpec:
    defaults = {
        "dataset": "longmemeval",
        "memory": "full-context",
        "answer_model": "ollama:llama3.1:8b",
        "judge_model": "ollama:llama3.1:8b",
    }
    defaults.update(overrides)
    return JobSpec(**defaults)


def test_spec_to_argv_builds_amb_run_command() -> None:
    spec = _spec(
        tag="demo",
        limit=10,
        split="s",
        judge_runs=3,
    )
    argv = spec.to_argv()
    assert argv[0] == sys.executable
    assert argv[1:4] == ["-m", "agent_memory_benchmark", "run"]
    assert "longmemeval" in argv
    assert argv[argv.index("--memory") + 1] == "full-context"
    assert argv[argv.index("--answer-model") + 1] == "ollama:llama3.1:8b"
    assert argv[argv.index("--judge-model") + 1] == "ollama:llama3.1:8b"
    assert argv[argv.index("--split") + 1] == "s"
    assert argv[argv.index("--limit") + 1] == "10"
    assert argv[argv.index("--judge-runs") + 1] == "3"
    assert argv[argv.index("--tag") + 1] == "demo"


def test_spec_to_argv_skips_defaults() -> None:
    spec = _spec()
    argv = spec.to_argv()
    assert "--limit" not in argv
    assert "--judge-runs" not in argv
    assert "--tag" not in argv
    assert "--variant" not in argv


def test_submit_launches_and_marks_succeeded(tmp_path: Path, monkeypatch) -> None:
    manager = JobManager(tmp_path)

    def fake_argv(self: JobSpec) -> list[str]:
        return _python_argv("print('hi')", exit_code=0)

    monkeypatch.setattr(JobSpec, "to_argv", fake_argv)

    record = manager.submit(_spec())
    assert record.state in {"queued", "running", "succeeded"}
    terminal = _wait_for_state(manager, record.job_id, {"succeeded", "failed"})
    assert terminal.state == "succeeded"
    assert terminal.exit_code == 0
    assert terminal.started_at is not None
    assert terminal.finished_at is not None

    logs_dir = tmp_path / record.job_id
    assert (logs_dir / "stdout.log").is_file()
    stdout = (logs_dir / "stdout.log").read_text(encoding="utf-8")
    assert "hi" in stdout


def test_submit_failure_marks_failed(tmp_path: Path, monkeypatch) -> None:
    manager = JobManager(tmp_path)
    monkeypatch.setattr(
        JobSpec,
        "to_argv",
        lambda self: _python_argv("print('nope')", exit_code=7),
    )
    record = manager.submit(_spec())
    terminal = _wait_for_state(manager, record.job_id, {"succeeded", "failed"})
    assert terminal.state == "failed"
    assert terminal.exit_code == 7


def test_concurrency_cap_queues_second_job(tmp_path: Path, monkeypatch) -> None:
    manager = JobManager(tmp_path, max_concurrent=1)
    # Slow child so the second submission has something to queue behind.
    monkeypatch.setattr(
        JobSpec,
        "to_argv",
        lambda self: _python_argv("import time; time.sleep(0.5)", exit_code=0),
    )
    first = manager.submit(_spec())
    second = manager.submit(_spec(tag="second"))
    # Immediately after submit, at most one should be running.
    states = [manager.get(first.job_id).state, manager.get(second.job_id).state]  # type: ignore[union-attr]
    assert "running" in states or "succeeded" in states
    assert sum(1 for s in states if s == "running") <= 1
    # Both should eventually succeed.
    _wait_for_state(manager, first.job_id, {"succeeded", "failed"})
    _wait_for_state(manager, second.job_id, {"succeeded", "failed"})
    final = [manager.get(first.job_id).state, manager.get(second.job_id).state]  # type: ignore[union-attr]
    assert final == ["succeeded", "succeeded"]


def test_list_returns_newest_first(tmp_path: Path, monkeypatch) -> None:
    manager = JobManager(tmp_path)
    monkeypatch.setattr(
        JobSpec, "to_argv", lambda self: _python_argv("pass", exit_code=0)
    )
    first = manager.submit(_spec(tag="first"))
    # Ensure the ids differ (they already do because job_id includes urandom,
    # but the sort is by lexical job_id which starts with timestamp → sleep
    # briefly so we get a different second).
    time.sleep(1.1)
    second = manager.submit(_spec(tag="second"))
    _wait_for_state(manager, first.job_id, {"succeeded"})
    _wait_for_state(manager, second.job_id, {"succeeded"})
    listed = manager.list_jobs()
    assert [r.job_id for r in listed[:2]] == [second.job_id, first.job_id]


def test_get_rejects_traversal(tmp_path: Path) -> None:
    manager = JobManager(tmp_path)
    assert manager.get("../etc/passwd") is None
    assert manager.get("..\\secret") is None
    assert manager.get("") is None
    assert manager.get(".hidden") is None


def test_reconcile_marks_running_without_pid_as_orphaned(tmp_path: Path) -> None:
    manager = JobManager(tmp_path)
    job_id = "2026-04-21_120000_aa"
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "state": "running",
                "argv": ["python", "-c", "pass"],
                "created_at": "2026-04-21T12:00:00Z",
                "dataset": "longmemeval",
                "memory": "full-context",
                "answer_model": "ollama:llama3.1:8b",
                "judge_model": "ollama:llama3.1:8b",
                "pid": 999_999_999,
                "spec": {},
            }
        ),
        encoding="utf-8",
    )
    manager.reconcile()
    record = manager.get(job_id)
    assert record is not None
    assert record.state == "orphaned"
    assert record.error is not None


def test_read_logs_returns_tail_and_handles_missing(tmp_path: Path) -> None:
    manager = JobManager(tmp_path)
    job_id = "2026-04-21_120000_bb"
    (tmp_path / job_id).mkdir()
    (tmp_path / job_id / "job.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "state": "succeeded",
                "argv": [],
                "created_at": "2026-04-21T12:00:00Z",
                "dataset": "longmemeval",
                "memory": "full-context",
                "answer_model": "ollama",
                "judge_model": "ollama",
                "spec": {},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / job_id / "stdout.log").write_text(
        "line1\nline2\nline3\n", encoding="utf-8"
    )
    stdout, stderr = manager.read_logs(job_id)
    assert "line3" in stdout
    assert stderr == ""  # file absent


def test_display_command_collapses_python_m_to_amb(tmp_path: Path) -> None:
    record = JobRecord(
        job_id="x",
        state="succeeded",
        argv=[sys.executable, "-m", "agent_memory_benchmark", "run", "longmemeval"],
        created_at="2026-04-21T12:00:00Z",
        dataset="longmemeval",
        memory="full-context",
        answer_model="ollama:llama3.1:8b",
        judge_model="ollama:llama3.1:8b",
    )
    assert record.display_command().startswith("amb run longmemeval")
