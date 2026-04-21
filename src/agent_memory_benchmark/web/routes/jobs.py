"""Routes for launching and inspecting benchmark jobs.

- ``GET /jobs`` — list.
- ``GET /jobs/new`` — form.
- ``POST /jobs`` — validate + either confirm (OpenAI) or submit.
- ``GET /jobs/{job_id}`` — detail with log tail.
- ``GET /jobs/{job_id}/logs.json`` — JSON tail for ad-hoc polling.

The two-stage confirm flow is the defensive wrapper around the paid
API roles: if either ``--answer-model`` or ``--judge-model`` is an
``openai:<model>`` spec, ``POST /jobs`` re-renders the form with a
cost breakdown and a hidden ``confirmed=yes`` field. Resubmitting
with that field set bypasses the check. Local-only jobs skip the
whole dance.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from ..cost import PRICE_TABLE_DATE, estimate_cost
from ..jobs import JobManager, JobSpec
from ..models import available_models, memory_adapter_presets

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates


def build_router(templates: Jinja2Templates) -> APIRouter:
    router = APIRouter()

    @router.get("/jobs")
    def list_jobs(request: Request):  # type: ignore[no-untyped-def]
        manager: JobManager = request.app.state.job_manager
        return templates.TemplateResponse(
            request,
            "jobs_list.html",
            {"jobs": manager.list_jobs()},
        )

    @router.get("/jobs/new")
    def new_job_form(request: Request):  # type: ignore[no-untyped-def]
        return templates.TemplateResponse(
            request,
            "job_new.html",
            _form_context(_default_form_values(), errors=[]),
        )

    @router.post("/jobs")
    async def create_job(request: Request):  # type: ignore[no-untyped-def]
        form = await request.form()
        values = {
            "dataset": _str_or_none(form.get("dataset")) or "",
            "memory": _str_or_none(form.get("memory")) or "",
            "answer_model": _str_or_none(form.get("answer_model")) or "",
            "judge_model": _str_or_none(form.get("judge_model")) or "",
            "tag": _str_or_none(form.get("tag")),
            "limit": _str_or_none(form.get("limit")),
            "split": _str_or_none(form.get("split")),
            "data": _str_or_none(form.get("data")),
            "judge_runs": _str_or_none(form.get("judge_runs")) or "1",
            "variant": _str_or_none(form.get("variant")) or "beam",
        }
        confirmed = _str_or_none(form.get("confirmed")) == "yes"

        errors = _validate(values)
        if errors:
            return templates.TemplateResponse(
                request,
                "job_new.html",
                _form_context(values, errors=errors),
                status_code=400,
            )

        spec = _build_spec(values)
        limit_int = int(values["limit"]) if values["limit"] else None
        estimate = estimate_cost(
            dataset=spec.dataset,
            answer_model_spec=spec.answer_model,
            judge_model_spec=spec.judge_model,
            n_questions=limit_int,
            judge_runs=spec.judge_runs,
            variant=spec.variant,
        )

        if estimate.has_paid_call and not confirmed:
            return templates.TemplateResponse(
                request,
                "job_new.html",
                _form_context(
                    values,
                    errors=[],
                    estimate=estimate,
                    needs_confirm=True,
                    preview_argv=spec.to_argv(),
                ),
            )

        manager: JobManager = request.app.state.job_manager
        record = manager.submit(spec)
        return RedirectResponse(url=f"/jobs/{record.job_id}", status_code=303)

    @router.get("/jobs/{job_id}")
    def job_detail(job_id: str, request: Request):  # type: ignore[no-untyped-def]
        manager: JobManager = request.app.state.job_manager
        record = manager.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
        stdout, stderr = manager.read_logs(job_id)
        return templates.TemplateResponse(
            request,
            "job_detail.html",
            {
                "job": record,
                "stdout": stdout,
                "stderr": stderr,
                "is_terminal": record.is_terminal(),
            },
        )

    @router.get("/jobs/{job_id}/logs.json")
    def job_logs_json(job_id: str, request: Request) -> dict[str, object]:
        manager: JobManager = request.app.state.job_manager
        record = manager.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
        stdout, stderr = manager.read_logs(job_id)
        return {
            "job_id": job_id,
            "state": record.state,
            "exit_code": record.exit_code,
            "stdout": stdout,
            "stderr": stderr,
        }

    return router


def _str_or_none(v: object) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _form_context(
    values: Mapping[str, object],
    *,
    errors: list[str],
    estimate: object = None,
    needs_confirm: bool = False,
    preview_argv: list[str] | None = None,
) -> dict[str, object]:
    """Build the template context for every render of ``job_new.html``.

    Centralizes the choice-list lookups so adding a new dropdown only
    touches this helper + the template. The current selections are
    reflected through ``form`` so a user's custom entry survives a
    validation error or confirm page re-render.
    """

    model_choices = available_models()
    memory_choices = memory_adapter_presets()
    ctx: dict[str, object] = {
        "form": values,
        "errors": errors,
        "estimate": estimate,
        "needs_confirm": needs_confirm,
        "model_choices": _choices_including_current(
            model_choices,
            *_as_model_current(values),
        ),
        "memory_choices": _choices_including_current(memory_choices, values.get("memory")),
    }
    if preview_argv is not None:
        ctx["preview_argv"] = preview_argv
        ctx["price_table_date"] = PRICE_TABLE_DATE
    return ctx


def _as_model_current(values: Mapping[str, object]) -> tuple[object, object]:
    return values.get("answer_model"), values.get("judge_model")


def _choices_including_current(choices: list[str], *current: object) -> list[str]:
    """Ensure the user's current selections are in the dropdown.

    A custom ``python:my.pkg:Cls`` memory spec (or a user-typed model)
    must round-trip across re-renders. We append the current value to
    the choice list if it's not already there so the ``<select>``
    can select it on reload.
    """

    out = list(choices)
    for c in current:
        if isinstance(c, str) and c and c not in out:
            out.append(c)
    return out


def _default_form_values() -> dict[str, object]:
    return {
        "dataset": "longmemeval",
        "memory": "full-context",
        "answer_model": "ollama:llama3.1:8b",
        "judge_model": "ollama:llama3.1:8b",
        "tag": None,
        "limit": "10",
        "split": "s",
        "data": None,
        "judge_runs": "1",
        "variant": "beam",
    }


def _validate(values: dict[str, str | None]) -> list[str]:
    errors: list[str] = []
    dataset = str(values.get("dataset") or "")
    if dataset not in ("longmemeval", "locomo", "beam"):
        errors.append("dataset must be one of: longmemeval, locomo, beam")
    if not values.get("memory"):
        errors.append("memory adapter spec is required")
    if not values.get("answer_model"):
        errors.append("answer model spec is required")
    if not values.get("judge_model"):
        errors.append("judge model spec is required (required by amb run)")
    if dataset == "longmemeval" and not values.get("split"):
        errors.append("split is required for longmemeval (e.g. 's')")
    if dataset == "locomo" and not values.get("data"):
        errors.append("data path is required for locomo (path to locomo10.json)")
    limit = values.get("limit")
    if limit:
        try:
            if int(str(limit)) <= 0:
                errors.append("limit must be a positive integer")
        except ValueError:
            errors.append("limit must be an integer")
    runs = values.get("judge_runs")
    if runs:
        try:
            if int(str(runs)) <= 0:
                errors.append("judge_runs must be a positive integer")
        except ValueError:
            errors.append("judge_runs must be an integer")
    return errors


def _build_spec(values: dict[str, str | None]) -> JobSpec:
    limit = values.get("limit")
    runs = values.get("judge_runs")
    return JobSpec(
        dataset=str(values["dataset"]),
        memory=str(values["memory"]),
        answer_model=str(values["answer_model"]),
        judge_model=str(values["judge_model"]),
        tag=_optional_str(values.get("tag")),
        limit=int(str(limit)) if limit else None,
        split=_optional_str(values.get("split")),
        data=_optional_str(values.get("data")),
        judge_runs=int(str(runs)) if runs else 1,
        variant=str(values.get("variant") or "beam"),
    )


def _optional_str(v: object) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


__all__ = ["build_router"]
