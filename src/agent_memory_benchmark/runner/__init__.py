"""High-level ``run_benchmark`` entry point.

The CLI (``amb run ...``) shapes its argv into the kwargs here. Library
users can call :func:`run_benchmark` directly with an already-configured
memory adapter / provider / dataset.

Responsibilities split:

- This module owns **assembly** — turning specs into concrete objects
  (providers, adapters, dataset, judge), resolving git state, building a
  :class:`RunManifest`, and constructing a timestamped :class:`RunDir`.
- :class:`.orchestrator.BenchmarkRunner` owns **execution** — the cache-
  aware ingest→answer→judge loop.
- :func:`.results.scorecard.build_scorecard` owns **aggregation** —
  computing KPIs from the list of ``QARecord`` the runner produces.

Keep that separation: assembly-time errors (missing deps, unresolvable
specs, bad dataset paths) fire before the adapter is opened so we don't
leak resources on misconfiguration.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..adapters.factory import resolve_adapter
from ..datasets import load_dataset
from ..datasets.base import DatasetAdapter
from ..judge.longmemeval import LME_JUDGE_FINGERPRINT
from ..llm import build_provider
from ..llm.judge_client import JudgeClient
from ..version import __version__
from .judge_adapter import BenchmarkJudge, LongMemEvalJudge
from .latest import update_latest_pointer
from .manifest import (
    QARecord,
    RunDir,
    RunManifest,
    build_run_directory_name,
    save_meta_json,
    save_run_file,
)
from .orchestrator import BenchmarkRunner

_log = logging.getLogger(__name__)

PROTOCOL_VERSION = "0.1"


async def run_benchmark(
    *,
    dataset_name: str,
    memory_spec: str,
    answer_model_spec: str,
    judge_model_spec: str,
    judge_temperature: float = 0.0,
    judge_runs: int = 1,
    split: str | None = None,
    m_data_path: Path | None = None,
    dataset_limit: int | None = None,
    dataset_limit_strategy: str = "stratified",
    memory_config: dict[str, Any] | None = None,
    results_base: Path = Path("results"),
    cache_root: Path = Path("cache"),
    tag: str | None = None,
    resume: bool = True,
    no_cache: bool = False,
    no_cache_ingestion: bool = False,
    replicate_idx: int = 0,
    replicate_seed: int | None = None,
    cli_argv: Sequence[str] | None = None,
    ollama_base_url: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
) -> RunDir:
    """Execute one benchmark run end to end; return the :class:`RunDir`.

    ``no_cache`` disables answer + judge cache reads (writes still land so
    later runs benefit). ``no_cache_ingestion`` disables ingestion-state
    cache reads specifically — useful when the adapter's state shape
    changes between runs and you want to force a re-ingest without
    bumping ``memory_version``.
    """

    dataset = _load_dataset(
        dataset_name,
        split=split,
        m_data_path=m_data_path,
        limit=dataset_limit,
        limit_strategy=dataset_limit_strategy,
    )
    descriptor_hash = dataset.descriptor_hash()

    answer_provider = build_provider(
        answer_model_spec,
        ollama_base_url=ollama_base_url,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
    )
    judge_provider = build_provider(
        judge_model_spec,
        ollama_base_url=ollama_base_url,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
    )

    adapter = resolve_adapter(
        memory_spec,
        config=memory_config,
        answer_provider=answer_provider,
    )

    answer_resolved = await answer_provider.resolve_spec()
    judge_resolved = await judge_provider.resolve_spec()

    judge_client = JudgeClient(judge_provider, temperature=judge_temperature)
    benchmark_judge = _build_benchmark_judge(
        dataset_name,
        client=judge_client,
        runs=judge_runs,
        temperature=judge_temperature,
    )

    hf_revision = _extract_hf_revision(dataset)
    dataset_path_str = _extract_dataset_path(dataset, m_data_path)

    timestamp = datetime.now(tz=UTC)
    dirname = build_run_directory_name(
        benchmark=dataset_name,
        memory_system_id=adapter.memory_system_id,
        answer_model_spec=answer_model_spec,
        timestamp=timestamp.astimezone(),
        tag=tag,
    )
    results_base = results_base.resolve()
    results_base.mkdir(parents=True, exist_ok=True)
    run_dir = RunDir(results_base / dirname)
    run_dir.path.mkdir(parents=True, exist_ok=True)

    manifest = RunManifest(
        benchmark=dataset_name,
        memory_system_id=adapter.memory_system_id,
        memory_version=adapter.memory_version,
        adapter_kind=_adapter_kind(memory_spec),
        adapter_target=memory_spec,
        answer_model_spec=answer_model_spec,
        answer_model_resolved=answer_resolved,
        judge_model_spec=judge_model_spec,
        judge_model_resolved=judge_resolved,
        judge_temperature=judge_temperature,
        judge_runs=judge_runs,
        judge_prompt_fingerprint=benchmark_judge.bundle_fingerprint,
        dataset_name=dataset_name,
        dataset_split=split,
        dataset_path=dataset_path_str,
        dataset_descriptor_hash=descriptor_hash,
        hf_revision_sha=hf_revision,
        replicate_idx=replicate_idx,
        replicate_seed=replicate_seed,
        benchmark_git_sha=_git_sha(),
        benchmark_git_branch=_git_branch(),
        benchmark_git_dirty=_git_dirty(),
        benchmark_version=__version__,
        protocol_version=PROTOCOL_VERSION,
        tag=tag,
        cli_argv=list(cli_argv) if cli_argv is not None else list(sys.argv),
        timestamp_utc=timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    save_meta_json(run_dir.meta_path, manifest)

    runner = BenchmarkRunner(
        dataset=dataset,
        adapter=adapter,
        judge=benchmark_judge,
        manifest=manifest,
        run_dir=run_dir,
        cache_root=cache_root.resolve(),
        results_base=results_base,
        dataset_descriptor_hash=descriptor_hash,
        answer_model_spec=answer_resolved,
        judge_model_spec=judge_resolved,
        judge_temperature=judge_temperature,
        judge_runs=judge_runs,
        benchmark_name=dataset_name,
        use_ingestion_cache=not no_cache_ingestion and not no_cache,
        use_answer_cache=not no_cache,
        use_judge_cache=not no_cache,
        resume=resume,
        replicate_idx=replicate_idx,
    )

    try:
        records = await runner.run()
    finally:
        await adapter.close()
        await judge_provider.close()
        # FullContext owns answer_provider via the adapter; calling close
        # twice is safe (close() is idempotent on the providers we ship).
        await answer_provider.close()

    _finalize_artifacts(run_dir, manifest=manifest, records=records, results_base=results_base)
    return run_dir


def _load_dataset(
    dataset_name: str,
    *,
    split: str | None,
    m_data_path: Path | None,
    limit: int | None,
    limit_strategy: str,
) -> DatasetAdapter:
    kwargs: dict[str, Any] = {}
    if dataset_name == "longmemeval":
        if split is None:
            raise ValueError("--split is required for LongMemEval (use 's' or 'm').")
        kwargs["split"] = split
        if m_data_path is not None:
            kwargs["m_path"] = m_data_path
        if limit is not None:
            kwargs["limit"] = limit
        kwargs["limit_strategy"] = limit_strategy
    return load_dataset(dataset_name, **kwargs)


def _build_benchmark_judge(
    dataset_name: str,
    *,
    client: JudgeClient,
    runs: int,
    temperature: float,
) -> BenchmarkJudge:
    if dataset_name == "longmemeval":
        return LongMemEvalJudge(
            client,
            runs=runs,
            temperature=temperature,
            bundle_fingerprint=LME_JUDGE_FINGERPRINT,
        )
    raise NotImplementedError(
        f"{dataset_name!r} judge is not wired yet — LOCOMO lands in PR-9, BEAM in PR-11."
    )


def _adapter_kind(spec: str) -> str:
    head = spec.split(":", 1)[0].strip().lower()
    if head in ("full-context", "python"):
        return head
    if head.startswith("http"):
        return "http"
    return head or "unknown"


def _extract_hf_revision(dataset: DatasetAdapter) -> str | None:
    """Pull ``revision`` off the dataset when available (LongMemEval has it)."""

    return getattr(dataset, "revision", None)


def _extract_dataset_path(
    dataset: DatasetAdapter,
    m_data_path: Path | None,
) -> str | None:
    if m_data_path is not None:
        return str(m_data_path.resolve())
    # LOCOMO/BEAM loaders (later PRs) will carry their own path attributes.
    path = getattr(dataset, "path", None)
    return str(path) if path else None


def _git_sha() -> str | None:
    return _run_git(("rev-parse", "HEAD"))


def _git_branch() -> str | None:
    branch = _run_git(("rev-parse", "--abbrev-ref", "HEAD"))
    return None if branch in (None, "HEAD") else branch


def _git_dirty() -> bool | None:
    out = _run_git(("status", "--porcelain"))
    if out is None:
        return None
    return bool(out)


def _run_git(args: tuple[str, ...]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _finalize_artifacts(
    run_dir: RunDir,
    *,
    manifest: RunManifest,
    records: list[QARecord],
    results_base: Path,
) -> None:
    """Write scorecard.{json,md} and update the latest pointer."""

    # Imported lazily so scorecard-layer deps (rich) don't load for callers
    # that only want run_benchmark() to execute without rendering.
    from ..results.render import render_scorecard_markdown
    from ..results.scorecard import build_scorecard, scorecard_to_dict

    save_run_file(run_dir.answers_path, manifest, records)
    scorecard = build_scorecard(records, benchmark=manifest.benchmark)
    run_dir.scorecard_json.write_text(
        json.dumps(scorecard_to_dict(scorecard), indent=2), encoding="utf-8"
    )
    md = render_scorecard_markdown(scorecard, manifest=manifest)
    run_dir.scorecard_md.write_text(md, encoding="utf-8")
    update_latest_pointer(results_base, run_dir.path)


__all__ = [
    "PROTOCOL_VERSION",
    "run_benchmark",
]
