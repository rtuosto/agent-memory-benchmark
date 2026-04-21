"""Shape a ``scorecard.json`` payload into Chart.js-friendly arrays.

The template embeds the result as ``<script id="chart-data">``; keeping
the shaping server-side (and unit-testable) avoids spreading key-name
knowledge across JS and Python, and lets us tolerate pre-rename
scorecards (``ingestion_per_session``) without special-casing in the
client.

When a baseline scorecard is supplied, every bucket row carries parallel
``baseline_*`` fields so the client can render the current run in a
warm palette and the baseline in a cool palette without re-indexing
across two lists.
"""

from __future__ import annotations

from typing import Any

# Per-query latency buckets rendered in the "Latency" chart. Ingestion is
# deliberately excluded — it lives in its own Ingestion section (cards +
# own chart), and mixing a 50-second ingestion-per-case bar with 5ms
# retrieval bars on a shared log scale squashes the per-query signal.
_LATENCY_KEYS: tuple[str, ...] = (
    "retrieval_per_query",
    "generation_per_query",
    "answer_total_per_query",
    "answer_discrepancy",
    "judge_per_question",
)

# Ingestion distribution keys coalesced under the canonical "ingestion_per_case"
# label so pre-rename runs (ingestion_per_session) and post-rename runs share
# the same bucket in comparisons. First non-empty payload wins.
_INGESTION_KEYS: tuple[str, ...] = ("ingestion_per_case", "ingestion_per_session")

_FOOTPRINT_KEYS: tuple[str, ...] = ("units_per_query", "tokens_per_query")


def build_chart_data(
    scorecard: dict[str, Any],
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe dict consumed by ``static/charts.js``.

    Top-level keys ``per_category``, ``latency``, ``footprint``,
    ``evidence`` each map to a list (possibly empty) of ``{name, ...}``
    rows; ``has_baseline`` is ``True`` when ``baseline`` was supplied and
    the client should render the cool-palette dataset on top of the
    warm-palette one.

    Per-category rows are emitted for the union of buckets across both
    runs; latency/footprint rows require a ``mean`` on at least one side.
    """

    quality = _as_dict(scorecard.get("quality"))
    latency = _as_dict(scorecard.get("latency_ms"))
    footprint = _as_dict(scorecard.get("retrieval_footprint"))
    evidence = _as_dict(scorecard.get("evidence"))

    b_quality = _as_dict(baseline.get("quality")) if baseline else {}
    b_latency = _as_dict(baseline.get("latency_ms")) if baseline else {}
    b_footprint = _as_dict(baseline.get("retrieval_footprint")) if baseline else {}
    b_evidence = _as_dict(baseline.get("evidence")) if baseline else {}

    # Coalesce ingestion_per_session into ingestion_per_case so pre-rename
    # and post-rename runs align in the per-case chart even when comparing
    # across the rename boundary.
    _coalesce_ingestion(latency)
    _coalesce_ingestion(b_latency)

    return {
        "has_baseline": baseline is not None,
        "per_category": _per_category_paired(
            quality.get("per_category"), b_quality.get("per_category")
        ),
        "latency": _distributions_paired(latency, b_latency, _LATENCY_KEYS),
        "footprint": _distributions_paired(footprint, b_footprint, _FOOTPRINT_KEYS),
        "evidence": _evidence_paired(evidence, b_evidence),
    }


def _coalesce_ingestion(latency: dict[str, Any]) -> None:
    """Promote the pre-rename key so callers can key on ``ingestion_per_case``."""

    if not latency:
        return
    if latency.get("ingestion_per_case") is None and isinstance(
        latency.get("ingestion_per_session"), dict
    ):
        latency["ingestion_per_case"] = latency["ingestion_per_session"]


def _per_category_paired(
    current: Any, baseline: Any
) -> list[dict[str, Any]]:
    cur: dict[str, Any] = current if isinstance(current, dict) else {}
    base: dict[str, Any] = baseline if isinstance(baseline, dict) else {}
    names = list(cur) + [k for k in base if k not in cur]
    buckets: list[dict[str, Any]] = []
    for name in names:
        c_val = cur.get(name)
        b_val = base.get(name)
        c: dict[str, Any] = c_val if isinstance(c_val, dict) else {}
        b: dict[str, Any] = b_val if isinstance(b_val, dict) else {}
        buckets.append(
            {
                "name": str(name),
                "count": _num(c.get("count")),
                "accuracy": _num(c.get("accuracy")),
                "token_f1": _num(c.get("token_f1")),
                "baseline_accuracy": _num(b.get("accuracy")),
                "baseline_token_f1": _num(b.get("token_f1")),
            }
        )
    return buckets


def _distributions_paired(
    current: dict[str, Any],
    baseline: dict[str, Any],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    buckets: list[dict[str, Any]] = []
    for key in keys:
        c_raw = current.get(key)
        b_raw = baseline.get(key)
        c_dist: dict[str, Any] = c_raw if isinstance(c_raw, dict) else {}
        b_dist: dict[str, Any] = b_raw if isinstance(b_raw, dict) else {}
        c_mean = _num(c_dist.get("mean"))
        b_mean = _num(b_dist.get("mean"))
        if c_mean is None and b_mean is None:
            continue
        buckets.append(
            {
                "name": key,
                "mean": c_mean,
                "p50": _num(c_dist.get("p50")),
                "p95": _num(c_dist.get("p95")),
                "max": _num(c_dist.get("max")),
                "n": _num(c_dist.get("n")),
                "baseline_mean": b_mean,
                "baseline_p50": _num(b_dist.get("p50")),
                "baseline_p95": _num(b_dist.get("p95")),
                "baseline_max": _num(b_dist.get("max")),
            }
        )
    return buckets


def _evidence_paired(
    current: dict[str, Any], baseline: dict[str, Any]
) -> list[dict[str, Any]]:
    """Pair completeness/density by tokenizer level for both runs."""

    if not current and not baseline:
        return []
    buckets: list[dict[str, Any]] = []
    for level in ("turn", "unit", "token"):
        c_comp = _dist_mean(current.get(f"{level}_completeness")) if current else None
        c_dens = _dist_mean(current.get(f"{level}_density")) if current else None
        b_comp = _dist_mean(baseline.get(f"{level}_completeness")) if baseline else None
        b_dens = _dist_mean(baseline.get(f"{level}_density")) if baseline else None
        if all(v is None for v in (c_comp, c_dens, b_comp, b_dens)):
            continue
        buckets.append(
            {
                "name": level,
                "completeness": c_comp,
                "density": c_dens,
                "baseline_completeness": b_comp,
                "baseline_density": b_dens,
            }
        )
    return buckets


def _dist_mean(payload: Any) -> float | None:
    if not isinstance(payload, dict):
        return None
    return _num(payload.get("mean"))


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


__all__ = ["build_chart_data"]
