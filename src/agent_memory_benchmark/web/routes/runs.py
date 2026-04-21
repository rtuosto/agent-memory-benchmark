"""Routes for browsing past runs.

- ``GET /`` and ``GET /runs`` render the list.
- ``GET /runs/{run_id}`` renders the detail view.
- ``GET /runs/{run_id}/scorecard.json`` and ``/meta.json`` serve the raw
  JSON for external consumers (dashboards, scripts, or tests).

All rendering pulls from :class:`ResultIndex`, whose mtime cache keeps
reloads cheap even on a 200-run history.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse

from ..charts import build_chart_data

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates


def build_router(templates: Jinja2Templates) -> APIRouter:
    router = APIRouter()

    @router.get("/")
    def root() -> RedirectResponse:
        return RedirectResponse(url="/runs", status_code=302)

    @router.get("/runs")
    def list_runs(request: Request):  # type: ignore[no-untyped-def]
        index = request.app.state.result_index
        return templates.TemplateResponse(
            request,
            "runs_list.html",
            {
                "runs": index.list_runs(),
                "results_dir": str(index.results_dir),
            },
        )

    @router.get("/runs/{run_id}")
    def run_detail(run_id: str, request: Request):  # type: ignore[no-untyped-def]
        index = request.app.state.result_index
        detail = index.get_run(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        import json as _json

        return templates.TemplateResponse(
            request,
            "run_detail.html",
            {
                "summary": detail.summary,
                "scorecard": detail.scorecard,
                "meta": detail.meta,
                "scorecard_md": detail.scorecard_md,
                "chart_data_json": _json.dumps(build_chart_data(detail.scorecard)),
            },
        )

    @router.get("/runs/{run_id}/scorecard.json")
    def scorecard_json(run_id: str, request: Request) -> FileResponse:
        path = _safe_file(request.app.state.result_index.results_dir, run_id, "scorecard.json")
        return FileResponse(path, media_type="application/json")

    @router.get("/runs/{run_id}/meta.json")
    def meta_json(run_id: str, request: Request) -> FileResponse:
        path = _safe_file(request.app.state.result_index.results_dir, run_id, "meta.json")
        return FileResponse(path, media_type="application/json")

    return router


def _safe_file(results_dir: Path, run_id: str, filename: str) -> Path:
    """Validate ``results_dir/run_id/filename`` stays inside ``results_dir``."""

    if "/" in run_id or "\\" in run_id or run_id.startswith(".."):
        raise HTTPException(status_code=400, detail="invalid run id")
    candidate = (results_dir / run_id / filename).resolve()
    try:
        candidate.relative_to(results_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="path traversal") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"{filename} not found")
    return candidate


__all__ = ["build_router"]
