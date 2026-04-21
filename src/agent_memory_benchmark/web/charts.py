"""Shape a ``scorecard.json`` payload into Chart.js-friendly arrays.

The template embeds the result as ``<script id="chart-data">``; keeping
the shaping server-side (and unit-testable) avoids spreading key-name
knowledge across JS and Python, and lets us tolerate pre-rename
scorecards (``ingestion_per_session``) without special-casing in the
client.
"""

from __future__ import annotations

from typing import Any

# Latency buckets rendered in order; the scorecard may omit any of them
# (e.g. ingestion_per_session on post-rename runs, or judge_per_question
# when the benchmark wasn't judged).
_LATENCY_KEYS: tuple[str, ...] = (
    "ingestion_per_case",
    "ingestion_per_session",  # pre-rename compat
    "retrieval_per_query",
    "generation_per_query",
    "answer_total_per_query",
    "answer_discrepancy",
    "judge_per_question",
)

_FOOTPRINT_KEYS: tuple[str, ...] = ("units_per_query", "tokens_per_query")


def build_chart_data(scorecard: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe dict consumed by ``static/charts.js``.

    Keys ``per_category``, ``latency``, ``footprint``, ``evidence`` each
    map to a list (possibly empty) of ``{name, ...}`` rows. The client
    skips chart rendering when the corresponding list is empty, so this
    function is tolerant about missing scorecard sections.
    """

    quality = _as_dict(scorecard.get("quality"))
    latency = _as_dict(scorecard.get("latency_ms"))
    footprint = _as_dict(scorecard.get("retrieval_footprint"))
    evidence = _as_dict(scorecard.get("evidence"))

    return {
        "per_category": _per_category(quality.get("per_category")),
        "latency": _distributions(latency, _LATENCY_KEYS),
        "footprint": _distributions(footprint, _FOOTPRINT_KEYS),
        "evidence": _evidence(evidence),
    }


def _per_category(payload: Any) -> list[dict[str, Any]]:
    buckets: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return buckets
    for name, cat in payload.items():
        if not isinstance(cat, dict):
            continue
        buckets.append(
            {
                "name": str(name),
                "count": _num(cat.get("count")),
                "accuracy": _num(cat.get("accuracy")),
                "token_f1": _num(cat.get("token_f1")),
            }
        )
    return buckets


def _distributions(payload: dict[str, Any], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    buckets: list[dict[str, Any]] = []
    for key in keys:
        dist = payload.get(key)
        if not isinstance(dist, dict):
            continue
        mean = _num(dist.get("mean"))
        if mean is None:
            continue
        buckets.append(
            {
                "name": key,
                "mean": mean,
                "p50": _num(dist.get("p50")),
                "p95": _num(dist.get("p95")),
                "max": _num(dist.get("max")),
                "n": _num(dist.get("n")),
            }
        )
    return buckets


def _evidence(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Pair completeness/density by tokenizer level (turn/unit/token)."""

    if not payload:
        return []
    buckets: list[dict[str, Any]] = []
    for level in ("turn", "unit", "token"):
        completeness = _dist_mean(payload.get(f"{level}_completeness"))
        density = _dist_mean(payload.get(f"{level}_density"))
        if completeness is None and density is None:
            continue
        buckets.append(
            {
                "name": level,
                "completeness": completeness,
                "density": density,
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
