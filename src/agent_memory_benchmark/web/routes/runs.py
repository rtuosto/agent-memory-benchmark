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

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, RedirectResponse

from ...results.compare import compare_scorecards
from ..charts import build_chart_data

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates

    from ..index import ResultIndex, RunDetail, RunSummary


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

    # Register the specific raw-file routes BEFORE the greedy
    # ``{run_id:path}`` detail route — otherwise ``:path`` swallows
    # the trailing ``/scorecard.json`` and the JSON endpoints 404.

    @router.get("/runs/{run_id:path}/scorecard.json")
    def scorecard_json(run_id: str, request: Request) -> FileResponse:
        path = _safe_file(request.app.state.result_index.results_dir, run_id, "scorecard.json")
        return FileResponse(path, media_type="application/json")

    @router.get("/runs/{run_id:path}/meta.json")
    def meta_json(run_id: str, request: Request) -> FileResponse:
        path = _safe_file(request.app.state.result_index.results_dir, run_id, "meta.json")
        return FileResponse(path, media_type="application/json")

    @router.get("/runs/{run_id:path}")
    def run_detail(  # type: ignore[no-untyped-def]
        run_id: str,
        request: Request,
        baseline: str | None = Query(
            default=None,
            description="Run ID to compare against. 'none' disables; unset = best.",
        ),
    ):
        index = request.app.state.result_index
        detail = index.get_run(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")

        baseline_summary, baseline_detail, baseline_mode = _resolve_baseline(
            index, detail, baseline
        )
        candidates = index.list_candidates(
            benchmark=detail.summary.benchmark, exclude_run_id=run_id
        )
        compare_table = None
        if baseline_detail is not None:
            compare_table = compare_scorecards(
                baseline_detail.scorecard,
                detail.scorecard,
                a_label=baseline_detail.summary.run_id,
                b_label=detail.summary.run_id,
            )

        import json as _json

        chart_data = build_chart_data(
            detail.scorecard,
            baseline_detail.scorecard if baseline_detail else None,
        )
        chart_data["current_label"] = _short_run_label(detail.summary)
        if baseline_detail is not None:
            chart_data["baseline_label"] = _short_run_label(baseline_detail.summary)

        return templates.TemplateResponse(
            request,
            "run_detail.html",
            {
                "summary": detail.summary,
                "scorecard": detail.scorecard,
                "meta": detail.meta,
                "scorecard_md": detail.scorecard_md,
                "chart_data_json": _json.dumps(chart_data),
                "baseline_summary": baseline_summary,
                "baseline_mode": baseline_mode,  # "auto" | "manual" | "none"
                "candidates": candidates,
                "compare": compare_table,
            },
        )

    return router


def _short_run_label(summary: RunSummary) -> str:
    """Compact identifier for chart legends — memory + tag, falling back to run ID."""

    if summary.memory_system_id and summary.tag:
        return f"{summary.memory_system_id} ({summary.tag})"
    if summary.memory_system_id:
        return summary.memory_system_id
    return summary.run_id


def _resolve_baseline(
    index: ResultIndex,
    detail: RunDetail,
    requested: str | None,
) -> tuple[RunSummary | None, RunDetail | None, str]:
    """Pick the baseline run to compare against, honoring the query param.

    Returns ``(summary, detail, mode)``:

    - ``mode == "none"`` if the user explicitly opted out via ``baseline=none``.
    - ``mode == "manual"`` if a specific ``run_id`` was requested and found.
    - ``mode == "auto"`` if we fell back to the best-known baseline (or
      there was nothing to compare against; then ``summary`` is also ``None``).
    """

    if requested == "none":
        return None, None, "none"
    if requested:
        candidate = index.get_run(requested)
        if candidate is not None and candidate.summary.benchmark == detail.summary.benchmark:
            return candidate.summary, candidate, "manual"
        # Fall through to auto if the requested run is missing / wrong benchmark.
    best = index.best_baseline(
        benchmark=detail.summary.benchmark, exclude_run_id=detail.summary.run_id
    )
    if best is None:
        return None, None, "auto"
    best_detail = index.get_run(best.run_id)
    return best, best_detail, "auto"


def _safe_file(results_dir: Path, run_id: str, filename: str) -> Path:
    """Validate ``results_dir/run_id/filename`` stays inside ``results_dir``.

    ``run_id`` may contain forward-slashes for nested runs — we accept
    them but reject any ``..`` segment plus absolute-looking paths.
    The final ``resolve()`` + ``relative_to`` pairing is the real
    traversal guard.
    """

    if not run_id or run_id.startswith(("/", "\\")):
        raise HTTPException(status_code=400, detail="invalid run id")
    parts = run_id.replace("\\", "/").split("/")
    if any(p in ("..", "") for p in parts):
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
