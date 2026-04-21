"""Render a :class:`Scorecard` to markdown (for PRs) and rich tables (for TTY).

Two audiences, same source:

- ``scorecard.md`` is the artifact a reviewer looks at on GitHub — throughput
  headline on top, then quality, then timing, then footprint. Keep it
  copy-paste-ready. No ANSI codes, no emoji.
- The rich console output is what ``amb run`` prints at the end of a run and
  what ``amb summarize`` uses. Same numbers, colorized.

The rich import is done lazily so the ``rich`` dep is only paid for when
actually rendering — scorecard construction (``build_scorecard``) stays a
pure-stdlib operation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .scorecard import Distribution, Scorecard

if TYPE_CHECKING:
    from ..runner.manifest import RunManifest


def render_scorecard_markdown(sc: Scorecard, *, manifest: RunManifest | None = None) -> str:
    """Return a GitHub-flavored markdown rendering of ``sc``."""

    lines: list[str] = []
    lines.append(f"# Scorecard — {sc.benchmark}")
    lines.append("")

    lines.extend(_methodology_lines(manifest, sc))
    lines.append("")

    lines.append("## Throughput")
    lines.append(_throughput_line("queries_per_sec", sc.throughput_queries_per_sec, " q/s"))
    lines.append(_throughput_line("sessions_per_sec", sc.throughput_sessions_per_sec, " s/s"))
    lines.append("")

    lines.append("## Quality")
    lines.append(f"- Questions: {sc.n_questions}")
    lines.append(f"- Cases: {sc.n_cases}")
    lines.append(_pct_line("Overall accuracy", sc.overall_accuracy))
    lines.append(_pct_line("Macro accuracy", sc.macro_accuracy))
    lines.append(_pct_line("Overall token-F1", sc.overall_token_f1))
    if sc.replicate_mean is not None and sc.replicate_std is not None:
        lines.append(
            f"- Replicates: mean={sc.replicate_mean * 100:.2f}% std={sc.replicate_std * 100:.2f}pp"
        )
    lines.append("")

    if sc.per_category:
        lines.append("### Per-category")
        for key, cat in sc.per_category.items():
            acc = "—" if cat.accuracy is None else f"{cat.accuracy * 100:.2f}%"
            f1 = "—" if cat.token_f1 is None else f"{cat.token_f1 * 100:.2f}%"
            lines.append(f"- **{key}** (n={cat.count}): accuracy={acc}, token-F1={f1}")
        lines.append("")

    lines.append("## Latency (ms)")
    lines.extend(_latency_rows(sc))
    lines.append("")

    lines.append("## Retrieval footprint")
    lines.append(_dist_line("units per query", sc.units_retrieved_per_query))
    lines.append(_dist_line("tokens per query", sc.tokens_retrieved_per_query))
    lines.append("")

    if sc.evidence is not None:
        lines.append("## Evidence KPIs")
        lines.append(
            f"- Questions with evidence annotations: {sc.evidence.n_questions_with_evidence}"
        )
        lines.append(
            f"- Questions with retrieval turn IDs: {sc.evidence.n_questions_with_retrieval}"
        )
        lines.append(_dist_line("turn completeness", sc.evidence.turn_completeness))
        lines.append(_dist_line("turn density", sc.evidence.turn_density))
        lines.append(_dist_line("unit completeness", sc.evidence.unit_completeness))
        lines.append(_dist_line("unit density", sc.evidence.unit_density))
        lines.append(_dist_line("token completeness", sc.evidence.token_completeness))
        lines.append(_dist_line("token density", sc.evidence.token_density))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _methodology_lines(manifest: RunManifest | None, sc: Scorecard) -> list[str]:
    lines = ["## Methodology"]
    if manifest is None:
        lines.append(f"- Benchmark: {sc.benchmark}")
        return lines
    lines.extend(
        [
            f"- Benchmark: {manifest.benchmark}"
            + (f" ({manifest.dataset_split})" if manifest.dataset_split else ""),
            f"- Memory system: {manifest.memory_system_id}@{manifest.memory_version} "
            f"(adapter: {manifest.adapter_kind})",
            f"- Answer model: `{manifest.answer_model_resolved}`",
            f"- Judge model: `{manifest.judge_model_resolved}` "
            f"(temp={manifest.judge_temperature}, runs={manifest.judge_runs})",
            f"- Judge prompt fingerprint: `{manifest.judge_prompt_fingerprint[:16]}…`",
            f"- Dataset descriptor: `{manifest.dataset_descriptor_hash[:16]}…`"
            + (
                f" (HF revision `{manifest.hf_revision_sha[:12]}`)"
                if manifest.hf_revision_sha
                else ""
            ),
            f"- Benchmark version: {manifest.benchmark_version} "
            f"(protocol {manifest.protocol_version})",
        ]
    )
    if manifest.benchmark_git_sha:
        dirty = " (dirty)" if manifest.benchmark_git_dirty else ""
        lines.append(f"- Git: {manifest.benchmark_git_sha[:7]}{dirty}")
    return lines


def _throughput_line(label: str, value: float | None, suffix: str) -> str:
    if value is None:
        return f"- {label}: —"
    return f"- {label}: **{value:.3f}**{suffix}"


def _pct_line(label: str, value: float | None) -> str:
    if value is None:
        return f"- {label}: —"
    return f"- {label}: **{value * 100:.2f}%**"


def _dist_line(label: str, dist: Distribution | None) -> str:
    if dist is None:
        return f"- {label}: —"
    return (
        f"- {label}: mean={dist.mean:.3f} p50={dist.p50:.3f} "
        f"p95={dist.p95:.3f} max={dist.max:.3f} (n={dist.n})"
    )


def _latency_rows(sc: Scorecard) -> list[str]:
    rows: list[tuple[str, Distribution | None]] = [
        ("ingestion per case", sc.ingestion_per_case_ms),
        ("retrieval per query", sc.retrieval_per_query_ms),
        ("generation per query", sc.generation_per_query_ms),
        ("answer total (runner-measured)", sc.answer_total_per_query_ms),
        ("answer discrepancy", sc.answer_discrepancy_ms),
        ("judge per question", sc.judge_per_question_ms),
    ]
    lines = [_dist_line(label, dist) for label, dist in rows]
    lines.append(f"- ingestion total: {sc.ingestion_total_ms:.1f} ms")
    return lines


def print_scorecard_rich(sc: Scorecard, *, manifest: RunManifest | None = None) -> None:
    """Rich-rendered scorecard on the active console. No-op if ``rich`` missing."""

    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:  # pragma: no cover
        print(render_scorecard_markdown(sc, manifest=manifest))
        return

    console = Console()
    console.rule(f"Scorecard — {sc.benchmark}")
    console.print(f"[bold]n_questions[/bold]={sc.n_questions}  [bold]n_cases[/bold]={sc.n_cases}")
    if sc.overall_accuracy is not None:
        console.print(f"Overall accuracy: [green]{sc.overall_accuracy * 100:.2f}%[/green]")
    if sc.macro_accuracy is not None:
        console.print(f"Macro accuracy:   [green]{sc.macro_accuracy * 100:.2f}%[/green]")
    if sc.overall_token_f1 is not None:
        console.print(f"Token-F1:         [green]{sc.overall_token_f1 * 100:.2f}%[/green]")
    if sc.throughput_queries_per_sec is not None:
        console.print(f"Throughput: [cyan]{sc.throughput_queries_per_sec:.3f}[/cyan] q/s")

    if sc.per_category:
        table = Table(title="Per-category quality")
        table.add_column("Bucket")
        table.add_column("Accuracy")
        table.add_column("Token-F1")
        table.add_column("n")
        for key, cat in sc.per_category.items():
            table.add_row(
                key,
                "—" if cat.accuracy is None else f"{cat.accuracy * 100:.2f}%",
                "—" if cat.token_f1 is None else f"{cat.token_f1 * 100:.2f}%",
                str(cat.count),
            )
        console.print(table)

    latency = Table(title="Latency (ms)")
    latency.add_column("Metric")
    latency.add_column("mean")
    latency.add_column("p50")
    latency.add_column("p95")
    latency.add_column("max")
    latency.add_column("n")
    for label, dist in [
        ("ingestion/case", sc.ingestion_per_case_ms),
        ("retrieval/query", sc.retrieval_per_query_ms),
        ("generation/query", sc.generation_per_query_ms),
        ("answer_total/query", sc.answer_total_per_query_ms),
        ("answer_discrepancy", sc.answer_discrepancy_ms),
        ("judge/question", sc.judge_per_question_ms),
    ]:
        if dist is None:
            latency.add_row(label, "—", "—", "—", "—", "0")
        else:
            latency.add_row(
                label,
                f"{dist.mean:.1f}",
                f"{dist.p50:.1f}",
                f"{dist.p95:.1f}",
                f"{dist.max:.1f}",
                str(dist.n),
            )
    console.print(latency)


__all__ = [
    "print_scorecard_rich",
    "render_scorecard_markdown",
]
