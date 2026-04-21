"""Job lifecycle management for benchmark subprocesses.

Each job is one invocation of ``amb run`` as a child process. State
lives on disk under ``<jobs_dir>/<job_id>/``:

- ``job.json`` — canonical record (state, argv, pid, timestamps, exit)
- ``stdout.log`` / ``stderr.log`` — raw subprocess output

Filesystem-as-truth means restarts don't lose job history. On startup,
:meth:`JobManager.reconcile` walks the directory: any ``running`` job
whose PID no longer exists is marked ``orphaned`` (a failed variant
with ``exit_code=None``).

A lightweight concurrency cap (default 1) keeps the user from
saturating a laptop with a queue of multi-hour benchmarks. Extra
submissions go to ``queued`` and are promoted in FIFO order whenever
a running job terminates. The supervisor is one daemon thread per
running job — no event loop, no shared scheduler — which keeps the
moving parts auditable.

Kill is deliberately stubbed (:meth:`kill` is a no-op returning
``False``). Step 8 will implement SIGTERM/SIGKILL with a grace
window; for now, the assumption is that user shuts down the server
to abort a run.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

JobState = Literal["queued", "running", "succeeded", "failed", "killed", "orphaned"]

_TERMINAL_STATES: frozenset[JobState] = frozenset(
    {"succeeded", "failed", "killed", "orphaned"}
)


@dataclass(frozen=True)
class JobSpec:
    """Form-submitted benchmark parameters, pre-validation.

    Captured as a dataclass rather than a dict so the web layer and
    the manager agree on the field surface. ``argv`` is rebuilt from
    these fields so the user can see (and edit, on a future form) the
    exact command line before confirming.
    """

    dataset: str
    memory: str
    answer_model: str
    judge_model: str
    tag: str | None = None
    limit: int | None = None
    split: str | None = None
    data: str | None = None
    judge_runs: int = 1
    variant: str = "beam"

    def to_argv(self) -> list[str]:
        """Build the ``amb run ...`` argv for this spec.

        The web form is the source of truth; any extra knobs (e.g.
        ``--ollama-base-url``) live in user-managed env vars or get
        added to the form later. Keeping this small keeps the confirm
        page honest.
        """

        argv: list[str] = [
            sys.executable,
            "-m",
            "agent_memory_benchmark",
            "run",
            self.dataset,
            "--memory",
            self.memory,
            "--answer-model",
            self.answer_model,
            "--judge-model",
            self.judge_model,
        ]
        if self.split:
            argv += ["--split", self.split]
        if self.data:
            argv += ["--data", self.data]
        if self.limit is not None:
            argv += ["--limit", str(self.limit)]
        if self.judge_runs and self.judge_runs != 1:
            argv += ["--judge-runs", str(self.judge_runs)]
        if self.dataset == "beam" and self.variant and self.variant != "beam":
            argv += ["--variant", self.variant]
        if self.tag:
            argv += ["--tag", self.tag]
        return argv


@dataclass
class JobRecord:
    """On-disk record of one job's state.

    Mutated in place by the supervisor thread; callers should treat
    snapshots returned from :meth:`JobManager.get` as read-only.
    """

    job_id: str
    state: JobState
    argv: list[str]
    created_at: str
    dataset: str
    memory: str
    answer_model: str
    judge_model: str
    tag: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    error: str | None = None
    spec: dict[str, object] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        return self.state in _TERMINAL_STATES

    def display_command(self) -> str:
        """Render ``argv`` as a quoted shell command for UI display.

        Python's path and the ``-m`` flag get collapsed to ``amb`` so
        the user sees the same command they'd type by hand — the
        subprocess form is only meaningful to the supervisor.
        """

        parts = list(self.argv)
        if len(parts) >= 3 and parts[1] == "-m" and parts[2] == "agent_memory_benchmark":
            parts = ["amb"] + parts[3:]
        return " ".join(shlex.quote(p) for p in parts)


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _generate_job_id(now: datetime | None = None) -> str:
    ts = (now or datetime.now(UTC)).strftime("%Y-%m-%d_%H%M%S")
    return f"{ts}_{os.urandom(2).hex()}"


def _pid_alive(pid: int) -> bool:
    """Cross-platform 'is this PID still running' check.

    POSIX uses ``kill(pid, 0)``; Windows uses :func:`os.waitpid` with
    ``WNOHANG`` via a safer substitute because ``kill(0)`` there
    raises on non-child PIDs. We don't need perfect accuracy — a
    false positive just leaves a dead-looking 'running' job until the
    user refreshes again and the supervisor thread has flushed state.
    """

    if sys.platform == "win32":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                ok = ctypes.windll.kernel32.GetExitCodeProcess(
                    handle, ctypes.byref(exit_code)
                )
                STILL_ACTIVE = 259
                return bool(ok) and exit_code.value == STILL_ACTIVE
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class JobManager:
    """Owns a ``jobs/`` directory and a supervisor thread per active job.

    All mutations go through a single :class:`threading.Lock` so the
    supervisor thread and the request handlers can't race on the
    queue. The lock is *not* held across subprocess launches — we copy
    the argv out, release, then spawn.
    """

    def __init__(self, jobs_dir: Path, *, max_concurrent: int = 1) -> None:
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max(1, max_concurrent)
        self._lock = threading.Lock()
        self._supervisors: dict[str, threading.Thread] = {}

    # ── Public API ────────────────────────────────────────────────────

    def reconcile(self) -> None:
        """Fix up stale state left behind by a previous server process.

        Jobs marked ``running`` or ``queued`` when the server died are
        no longer actually running. ``running`` → ``orphaned``; the
        ``queued`` ones are left as-is so they promote naturally on
        the next capacity check.
        """

        for record in self._load_all():
            if record.state == "running":
                if record.pid is not None and _pid_alive(record.pid):
                    # A child genuinely survived the server restart —
                    # we can't re-attach its streams, so treat it as
                    # orphaned. The log files on disk are still valid.
                    self._mark_terminal(record.job_id, "orphaned", exit_code=None,
                                        error="server restarted; process detached")
                else:
                    self._mark_terminal(record.job_id, "orphaned", exit_code=None,
                                        error="server restarted before job finished")
        # Promote queued jobs after reconcile so existing queues resume.
        self._maybe_promote()

    def submit(self, spec: JobSpec) -> JobRecord:
        """Create a job record and launch (or queue) the subprocess.

        Validation happens in the route layer; by the time the spec
        reaches us it's assumed well-formed. The manager's job is to
        persist the record atomically and respect the concurrency cap.
        """

        job_id = _generate_job_id()
        record = JobRecord(
            job_id=job_id,
            state="queued",
            argv=spec.to_argv(),
            created_at=_now_iso(),
            dataset=spec.dataset,
            memory=spec.memory,
            answer_model=spec.answer_model,
            judge_model=spec.judge_model,
            tag=spec.tag,
            spec=asdict(spec),
        )
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        self._write_record(record)
        self._maybe_promote()
        # Return the latest record — it may have promoted straight to running.
        return self.get(job_id) or record

    def list_jobs(self) -> list[JobRecord]:
        """Return all jobs, newest first by job_id."""

        records = self._load_all()
        records.sort(key=lambda r: r.job_id, reverse=True)
        return records

    def get(self, job_id: str) -> JobRecord | None:
        if not self._is_safe_id(job_id):
            return None
        path = self.jobs_dir / job_id / "job.json"
        if not path.is_file():
            return None
        return self._read_record(path)

    def read_logs(self, job_id: str, *, tail_bytes: int = 64_000) -> tuple[str, str]:
        """Return ``(stdout, stderr)`` tails for the detail page.

        The tail size is bounded so a runaway job can't OOM the server
        when a user opens the detail page. The log files themselves
        are preserved in full.
        """

        if not self._is_safe_id(job_id):
            return ("", "")
        job_dir = self.jobs_dir / job_id
        return (
            _read_tail(job_dir / "stdout.log", tail_bytes),
            _read_tail(job_dir / "stderr.log", tail_bytes),
        )

    def kill(self, job_id: str) -> bool:
        """Placeholder — step 8 wires SIGTERM/SIGKILL."""

        return False

    # ── Internal: persistence ─────────────────────────────────────────

    def _write_record(self, record: JobRecord) -> None:
        path = self.jobs_dir / record.job_id / "job.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        tmp.replace(path)

    def _read_record(self, path: Path) -> JobRecord | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        try:
            return JobRecord(**data)
        except TypeError:
            return None

    def _load_all(self) -> list[JobRecord]:
        records: list[JobRecord] = []
        if not self.jobs_dir.exists():
            return records
        for entry in self.jobs_dir.iterdir():
            if not entry.is_dir():
                continue
            record = self._read_record(entry / "job.json")
            if record is not None:
                records.append(record)
        return records

    def _update(self, job_id: str, **changes: object) -> JobRecord | None:
        path = self.jobs_dir / job_id / "job.json"
        record = self._read_record(path)
        if record is None:
            return None
        for key, value in changes.items():
            setattr(record, key, value)
        self._write_record(record)
        return record

    def _mark_terminal(
        self,
        job_id: str,
        state: JobState,
        *,
        exit_code: int | None,
        error: str | None = None,
    ) -> None:
        self._update(
            job_id,
            state=state,
            exit_code=exit_code,
            finished_at=_now_iso(),
            error=error,
        )

    @staticmethod
    def _is_safe_id(job_id: str) -> bool:
        if not job_id:
            return False
        return not ("/" in job_id or "\\" in job_id or job_id.startswith("."))

    # ── Internal: scheduling ──────────────────────────────────────────

    def _maybe_promote(self) -> None:
        """Start queued jobs until the concurrency cap is saturated.

        Called after every state change that could free capacity
        (submit + job-finish in the supervisor). Holding the lock the
        whole way means two supervisors can't both race to promote
        the same queued job.
        """

        with self._lock:
            running = [r for r in self._load_all() if r.state == "running"]
            if len(running) >= self.max_concurrent:
                return
            queued = sorted(
                (r for r in self._load_all() if r.state == "queued"),
                key=lambda r: r.job_id,
            )
            slots = self.max_concurrent - len(running)
            to_start = queued[:slots]
            for record in to_start:
                self._launch(record)

    def _launch(self, record: JobRecord) -> None:
        """Spawn the subprocess and hand off to a supervisor thread.

        Must be called with ``self._lock`` held. Returns immediately
        after ``Popen`` — the supervisor owns the wait + state update.
        """

        job_dir = self.jobs_dir / record.job_id
        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"
        try:
            stdout_f = stdout_path.open("ab", buffering=0)
            stderr_f = stderr_path.open("ab", buffering=0)
            proc = subprocess.Popen(  # noqa: S603  # argv is fully controlled
                record.argv,
                stdout=stdout_f,
                stderr=stderr_f,
                stdin=subprocess.DEVNULL,
                cwd=os.getcwd(),
            )
        except OSError as exc:
            self._update(
                record.job_id,
                state="failed",
                error=f"failed to launch: {exc}",
                finished_at=_now_iso(),
            )
            return

        self._update(
            record.job_id,
            state="running",
            pid=proc.pid,
            started_at=_now_iso(),
        )
        thread = threading.Thread(
            target=self._supervise,
            args=(record.job_id, proc, stdout_f, stderr_f),
            name=f"job-{record.job_id}",
            daemon=True,
        )
        self._supervisors[record.job_id] = thread
        thread.start()

    def _supervise(
        self,
        job_id: str,
        proc: subprocess.Popen[bytes],
        stdout_f: object,
        stderr_f: object,
    ) -> None:
        """Wait for the child and update state on exit.

        Runs off the request thread so long benchmark runs don't block
        the FastAPI event loop. Any exception here is caught and
        recorded; crashing the supervisor would leave the job stuck
        in ``running`` forever.
        """

        try:
            rc = proc.wait()
            state: JobState = "succeeded" if rc == 0 else "failed"
            self._mark_terminal(job_id, state, exit_code=rc)
        except Exception as exc:  # noqa: BLE001  # supervisor boundary
            self._mark_terminal(
                job_id,
                "failed",
                exit_code=None,
                error=f"supervisor error: {type(exc).__name__}: {exc}",
            )
        finally:
            for f in (stdout_f, stderr_f):
                close = getattr(f, "close", None)
                if callable(close):
                    with contextlib.suppress(Exception):
                        close()
            self._supervisors.pop(job_id, None)
            self._maybe_promote()


def _read_tail(path: Path, nbytes: int) -> str:
    if not path.is_file():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > nbytes:
                f.seek(size - nbytes)
                data = f.read()
                # Drop the first (probably partial) line so the UI
                # doesn't show a half-sentence at the top.
                nl = data.find(b"\n")
                if nl >= 0:
                    data = data[nl + 1 :]
            else:
                data = f.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


__all__ = [
    "JobManager",
    "JobRecord",
    "JobSpec",
    "JobState",
]
