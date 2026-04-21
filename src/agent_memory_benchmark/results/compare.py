"""Pure scorecard-diff logic shared by ``amb compare`` and the web dashboard.

``compare_scorecards(a, b)`` returns a :class:`CompareTable` — a typed,
JSON-safe view that both the CLI text renderer and the Jinja2 templates
consume. Keeping this module framework-free (no argparse, no FastAPI)
means regression tests that lock the diff shape are the single source
of truth; both entry points format the same numbers.

Units are carried on each row (``"pct"``/``"ms"``/``"count"``/``"ratio"``)
so renderers can decide formatting without re-deriving it from the label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Unit = Literal["pct", "ms", "count", "ratio"]


@dataclass(frozen=True)
class CompareRow:
    """One label/A/B/Δ triple, with a unit hint for renderers."""

    label: str
    a: float | None
    b: float | None
    delta: float | None
    unit: Unit


@dataclass(frozen=True)
class CompareSection:
    """A labelled group of rows (e.g. ``"Quality"``, ``"Latency (mean ms)"``)."""

    title: str
    rows: list[CompareRow] = field(default_factory=list)

    @property
    def non_empty(self) -> bool:
        return any(r.a is not None or r.b is not None for r in self.rows)


@dataclass(frozen=True)
class CompareTable:
    """Full diff payload used by both CLI and web renderers.

    ``benchmarks_match`` is a dedicated flag rather than a string
    comparison because renderers often want to surface a warning banner
    when you accidentally diff runs from different benchmarks.
    """

    a_label: str
    b_label: str
    a_benchmark: str | None
    b_benchmark: str | None
    benchmarks_match: bool
    quality: CompareSection
    per_category: CompareSection
    ingestion: CompareSection
    latency: CompareSection
    footprint: CompareSection
    throughput: CompareSection
    evidence: CompareSection

    @property
    def sections(self) -> list[CompareSection]:
        """Ordered, for renderers that want to iterate."""

        return [
            self.quality,
            self.per_category,
            self.ingestion,
            self.latency,
            self.footprint,
            self.throughput,
            self.evidence,
        ]


def compare_scorecards(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    a_label: str = "A",
    b_label: str = "B",
) -> CompareTable:
    """Produce a :class:`CompareTable` for two scorecard payloads.

    Both payloads are expected to be the dicts loaded from
    ``scorecard.json``; missing keys on either side degrade gracefully to
    ``None`` cells rather than raising, so comparing runs with different
    per-category breakdowns or evidence coverage is safe.
    """

    a_bench = _opt_str(a.get("benchmark"))
    b_bench = _opt_str(b.get("benchmark"))

    quality = CompareSection(
        "Quality",
        [
            _pct_row("overall_accuracy", a, b, "quality.overall_accuracy"),
            _pct_row("macro_accuracy", a, b, "quality.macro_accuracy"),
            _pct_row("overall_token_f1", a, b, "quality.overall_token_f1"),
        ],
    )

    per_cat = CompareSection("Per-category accuracy", _per_category_rows(a, b))

    ingestion = CompareSection(
        "Ingestion",
        [
            _ms_row("ingestion_total_ms", a, b, "latency_ms.ingestion_total"),
            _ms_row("ingestion_per_case_mean", a, b, "latency_ms.ingestion_per_case.mean"),
            _ms_row("ingestion_per_case_p95", a, b, "latency_ms.ingestion_per_case.p95"),
            _float_row(
                "sessions_per_sec",
                a,
                b,
                "throughput.sessions_per_sec",
                unit="ratio",
            ),
        ],
    )

    latency = CompareSection(
        "Latency (mean ms)",
        [
            _ms_row(key, a, b, f"latency_ms.{key}.mean")
            for key in (
                "retrieval_per_query",
                "generation_per_query",
                "answer_total_per_query",
                "answer_discrepancy",
                "judge_per_question",
            )
        ],
    )

    footprint = CompareSection(
        "Retrieval footprint (mean per query)",
        [
            _float_row("units", a, b, "retrieval_footprint.units_per_query.mean", unit="count"),
            _float_row("tokens", a, b, "retrieval_footprint.tokens_per_query.mean", unit="count"),
        ],
    )

    throughput = CompareSection(
        "Throughput",
        [
            _float_row("queries_per_sec", a, b, "throughput.queries_per_sec", unit="ratio"),
            _float_row("sessions_per_sec", a, b, "throughput.sessions_per_sec", unit="ratio"),
        ],
    )

    evidence = CompareSection(
        "Evidence KPIs (mean)",
        [
            _pct_row(key, a, b, f"evidence.{key}.mean")
            for key in (
                "turn_completeness",
                "turn_density",
                "unit_completeness",
                "unit_density",
                "token_completeness",
                "token_density",
            )
        ],
    )

    return CompareTable(
        a_label=a_label,
        b_label=b_label,
        a_benchmark=a_bench,
        b_benchmark=b_bench,
        benchmarks_match=(a_bench == b_bench),
        quality=quality,
        per_category=per_cat,
        ingestion=ingestion,
        latency=latency,
        footprint=footprint,
        throughput=throughput,
        evidence=evidence,
    )


def _per_category_rows(a: dict[str, Any], b: dict[str, Any]) -> list[CompareRow]:
    per_a = _dig(a, "quality.per_category") or {}
    per_b = _dig(b, "quality.per_category") or {}
    cats = sorted(set(per_a) | set(per_b))
    rows: list[CompareRow] = []
    for cat in cats:
        av = _safe_get(per_a, cat, "accuracy")
        bv = _safe_get(per_b, cat, "accuracy")
        rows.append(
            CompareRow(label=cat, a=av, b=bv, delta=_delta(av, bv), unit="pct")
        )
    return rows


def _pct_row(label: str, a: dict[str, Any], b: dict[str, Any], path: str) -> CompareRow:
    av = _num(_dig(a, path))
    bv = _num(_dig(b, path))
    return CompareRow(label=label, a=av, b=bv, delta=_delta(av, bv), unit="pct")


def _ms_row(label: str, a: dict[str, Any], b: dict[str, Any], path: str) -> CompareRow:
    av = _num(_dig(a, path))
    bv = _num(_dig(b, path))
    return CompareRow(label=label, a=av, b=bv, delta=_delta(av, bv), unit="ms")


def _float_row(
    label: str,
    a: dict[str, Any],
    b: dict[str, Any],
    path: str,
    *,
    unit: Unit,
) -> CompareRow:
    av = _num(_dig(a, path))
    bv = _num(_dig(b, path))
    return CompareRow(label=label, a=av, b=bv, delta=_delta(av, bv), unit=unit)


def _dig(obj: Any, path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    return cur


def _safe_get(d: Any, *keys: str) -> float | None:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
        if cur is None:
            return None
    return _num(cur)


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return b - a


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = ["CompareRow", "CompareSection", "CompareTable", "compare_scorecards"]
