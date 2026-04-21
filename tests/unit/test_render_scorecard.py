"""Markdown rendering smoke tests.

Full byte-exact golden lock would be fragile against innocent formatting
tweaks; instead we assert the critical sections + headline numbers are
present.
"""

from __future__ import annotations

from agent_memory_benchmark.results.render import render_scorecard_markdown
from agent_memory_benchmark.results.scorecard import build_scorecard
from agent_memory_benchmark.runner.manifest import QARecord, RunManifest


def _record(key: str, correct: bool = True) -> QARecord:
    return QARecord(
        key=key,
        benchmark="longmemeval",
        case_id=key,
        question="Q?",
        gold="A",
        generated="A",
        question_type="single-session-user",
        qa_index=0,
        ingestion_time_ms=100.0,
        retrieval_time_ms=5.0,
        generation_time_ms=500.0,
        total_answer_time_ms=510.0,
        answer_discrepancy_ms=5.0,
        units_retrieved=10,
        tokens_retrieved=1000,
        judge_runs=[{"correct": correct, "raw": "yes" if correct else "no"}],
    )


def _manifest() -> RunManifest:
    return RunManifest(
        benchmark="longmemeval",
        memory_system_id="full-context",
        memory_version="0.1.0",
        adapter_kind="full-context",
        adapter_target="full-context",
        answer_model_spec="ollama:llama3.1:8b",
        answer_model_resolved="ollama:llama3.1:8b@sha256:abc",
        judge_model_spec="ollama:llama3.1:70b",
        judge_model_resolved="ollama:llama3.1:70b@sha256:def",
        judge_temperature=0.0,
        judge_runs=1,
        judge_prompt_fingerprint="f" * 64,
        dataset_name="longmemeval",
        dataset_split="s",
        dataset_path=None,
        dataset_descriptor_hash="d" * 64,
        hf_revision_sha="0" * 40,
        replicate_idx=0,
        replicate_seed=None,
        benchmark_git_sha="abcdef0123456789" + "0" * 24,
        benchmark_git_branch="main",
        benchmark_git_dirty=False,
        benchmark_version="0.1.0",
        protocol_version="0.1",
        tag=None,
        cli_argv=[],
        timestamp_utc="2026-04-20T00:00:00Z",
    )


def test_markdown_contains_all_sections() -> None:
    sc = build_scorecard([_record("q1"), _record("q2", correct=False)])
    md = render_scorecard_markdown(sc, manifest=_manifest())
    for section in (
        "# Scorecard — longmemeval",
        "## Methodology",
        "## Throughput",
        "## Quality",
        "### Per-category",
        "## Latency (ms)",
        "## Retrieval footprint",
        "## Evidence KPIs",
    ):
        assert section in md, f"missing section {section!r} in:\n{md}"


def test_markdown_shows_accuracy_percentage() -> None:
    sc = build_scorecard([_record("q1"), _record("q2", correct=False)])
    md = render_scorecard_markdown(sc, manifest=_manifest())
    assert "50.00%" in md  # overall accuracy for 1/2 correct


def test_markdown_handles_empty_records() -> None:
    sc = build_scorecard([], benchmark="longmemeval")
    md = render_scorecard_markdown(sc)
    assert "# Scorecard — longmemeval" in md
    assert "—" in md  # placeholders for missing values


def test_markdown_without_manifest_shows_benchmark() -> None:
    sc = build_scorecard([_record("q1")])
    md = render_scorecard_markdown(sc)  # no manifest
    assert "longmemeval" in md
