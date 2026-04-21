"""FastAPI app factory for the ``amb serve`` dashboard.

The factory takes a :class:`WebConfig` rather than reading argv directly
so tests can build an app pointed at a synthetic ``results/`` fixture
without going through the CLI. Imports of fastapi/jinja2 happen at
module import time — callers must check ``web_deps_available()`` first
if they want a friendly error when the ``[web]`` extra is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI


@dataclass(frozen=True)
class WebConfig:
    """Runtime config for the dashboard app.

    Paths are resolved to absolute form by ``serve_command`` before the
    app is built so request handlers can validate traversal cheaply.
    """

    results_dir: Path
    jobs_dir: Path
    max_concurrent_jobs: int = 1


def web_deps_available() -> tuple[bool, str | None]:
    """Return ``(True, None)`` if the ``[web]`` extra is importable.

    On failure returns ``(False, hint)`` with a ready-to-print install
    string. Kept separate from :func:`create_app` so the CLI can print
    the hint *before* attempting to construct the app.
    """

    try:
        import fastapi  # noqa: F401
        import jinja2  # noqa: F401
        import sse_starlette  # noqa: F401
    except ImportError as exc:
        return False, (
            f"web extras missing ({exc.name}). Install with: "
            "pip install -e \".[web]\""
        )
    return True, None


def create_app(config: WebConfig) -> FastAPI:
    """Build the FastAPI app instance.

    Registers the runs router + static + templates. Later steps add
    compare and jobs routers; ``/health`` stays as a cheap liveness probe.
    """

    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates

    from .index import ResultIndex
    from .jobs import JobManager
    from .routes.jobs import build_router as build_jobs_router
    from .routes.runs import build_router as build_runs_router

    here = Path(__file__).parent
    templates = Jinja2Templates(directory=str(here / "templates"))

    app = FastAPI(
        title="agent-memory-benchmark dashboard",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
    )
    app.state.config = config
    app.state.result_index = ResultIndex(config.results_dir)
    app.state.templates = templates
    job_manager = JobManager(config.jobs_dir, max_concurrent=config.max_concurrent_jobs)
    job_manager.reconcile()
    app.state.job_manager = job_manager

    app.mount("/static", StaticFiles(directory=str(here / "static")), name="static")
    app.include_router(build_runs_router(templates))
    app.include_router(build_jobs_router(templates))

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "results_dir": str(config.results_dir),
            "jobs_dir": str(config.jobs_dir),
        }

    return app


__all__ = ["WebConfig", "create_app", "web_deps_available"]
