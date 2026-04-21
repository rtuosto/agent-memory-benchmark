"""``amb rejudge`` — re-run the judge over a stored ``answers.json``.

Motivation: switching judge models or bumping ``--judge-runs`` should not
require re-generating answers. Rejudge reads an existing run's records,
runs the new judge over each ``(question, gold, generated)`` triple, and
writes a fresh run directory containing the updated ``answers.json``,
``scorecard.json``, ``scorecard.md``, and ``meta.json``.

The original run directory is left untouched. The new directory lives at
``<original_dir>/rejudged_<ts>/`` by default (``--out`` overrides). The
judge cache is populated for the new judge so later rejudges with the
same arguments hit cache for free.

The manifest carried forward from the original run preserves the
benchmark/dataset/memory/answer-model fields verbatim — only the judge
block and the invocation metadata (git sha, cli argv, timestamp) are
refreshed. This keeps the byte-stable answer-cache keys consistent with
any later ``amb run --resume`` that points at the same results tree.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from ..cache.index import CacheIndexWriter
from ..cache.keys import judge_cache_path, judge_key
from ..llm import build_provider
from ..llm.judge_client import JudgeClient
from ..runner import build_benchmark_judge
from ..runner.judge_adapter import BenchmarkJudge
from ..runner.manifest import (
    QARecord,
    RunDir,
    RunManifest,
    load_run_file,
    save_meta_json,
    save_run_file,
)
from ..types import QAItem


def add_rejudge_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Register ``amb rejudge`` on the root parser."""

    parser: argparse.ArgumentParser = subparsers.add_parser(
        "rejudge",
        help="Re-run the judge over an existing answers.json.",
        description=(
            "Reuse stored generations; swap in a different judge model / "
            "temperature / run count without re-hitting the answer LLM."
        ),
    )
    parser.add_argument("answers_path", type=Path, help="Path to an existing answers.json.")
    parser.add_argument("--judge-model", required=True, help="Judge LLM spec.")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-runs", type=int, default=1)
    parser.add_argument(
        "--out",
        type=Path,
        help="Destination directory. Defaults to <run_dir>/rejudged_<timestamp>/.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("cache"),
        help="Cache root used to populate judge verdicts for future runs.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip writing judge verdicts to the cache (default: write).",
    )
    parser.add_argument(
        "--ollama-base-url",
        help="Override Ollama base URL (default http://localhost:11434).",
    )
    parser.add_argument("--openai-base-url")
    return parser


def rejudge_command(args: argparse.Namespace, *, argv: list[str] | None = None) -> int:
    """Synchronous CLI entry; returns process exit code."""

    answers_path: Path = args.answers_path
    if not answers_path.is_file():
        print(f"error: {answers_path} does not exist or is not a file", file=sys.stderr)
        return 2

    try:
        manifest, record_map = load_run_file(answers_path)
    except (OSError, json.JSONDecodeError, KeyError) as e:
        print(f"error: could not read {answers_path}: {e}", file=sys.stderr)
        return 1

    out_dir = _resolve_out_dir(args.out, answers_path)
    try:
        asyncio.run(
            _do_rejudge(
                manifest=manifest,
                record_map=record_map,
                out_dir=out_dir,
                judge_model_spec=args.judge_model,
                judge_temperature=args.judge_temperature,
                judge_runs_arg=args.judge_runs,
                cache_root=args.cache_root.resolve(),
                no_cache=args.no_cache,
                ollama_base_url=args.ollama_base_url,
                openai_base_url=args.openai_base_url,
                cli_argv=list(argv) if argv is not None else list(sys.argv),
            )
        )
    except Exception as e:  # noqa: BLE001 — CLI boundary
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"Rejudge complete. Output: {out_dir}")
    return 0


def _resolve_out_dir(explicit: Path | None, answers_path: Path) -> Path:
    if explicit is not None:
        return explicit.resolve()
    ts = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H%M%S")
    return (answers_path.parent / f"rejudged_{ts}").resolve()


async def _do_rejudge(
    *,
    manifest: RunManifest,
    record_map: dict[str, QARecord],
    out_dir: Path,
    judge_model_spec: str,
    judge_temperature: float,
    judge_runs_arg: int,
    cache_root: Path,
    no_cache: bool,
    ollama_base_url: str | None,
    openai_base_url: str | None,
    cli_argv: list[str],
) -> None:
    judge_provider = build_provider(
        judge_model_spec,
        ollama_base_url=ollama_base_url,
        openai_base_url=openai_base_url,
    )
    judge_resolved = await judge_provider.resolve_spec()
    client = JudgeClient(judge_provider, temperature=judge_temperature)
    benchmark_judge = _build_benchmark_judge(
        manifest.benchmark,
        client=client,
        runs=judge_runs_arg,
        temperature=judge_temperature,
    )

    records = list(record_map.values())
    out_dir.mkdir(parents=True, exist_ok=True)

    new_manifest = replace(
        manifest,
        judge_model_spec=judge_model_spec,
        judge_model_resolved=judge_resolved,
        judge_temperature=judge_temperature,
        judge_runs=judge_runs_arg,
        judge_prompt_fingerprint=benchmark_judge.bundle_fingerprint,
        cli_argv=cli_argv,
        timestamp_utc=datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    try:
        with CacheIndexWriter(cache_root) as index_writer:
            for record in records:
                qa = _qa_from_record(record)
                outcome = await benchmark_judge.judge(qa, record.generated)
                record.judge_runs = [dict(v) for v in outcome.verdicts]
                record.judge_time_ms = outcome.judge_time_ms

                if not no_cache:
                    jk = judge_key(
                        manifest.benchmark,
                        judge_resolved,
                        judge_temperature,
                        judge_runs_arg,
                        outcome.prompt_fingerprint,
                        record.question,
                        record.gold,
                        record.generated,
                        question_type=record.question_type,
                        question_id=record.question_id,
                    )
                    jpath = judge_cache_path(cache_root, jk)
                    jpath.parent.mkdir(parents=True, exist_ok=True)
                    jpath.write_text(
                        json.dumps(
                            {
                                "judge_runs": record.judge_runs,
                                "judge_time_ms": record.judge_time_ms,
                                "prompt_fingerprint": outcome.prompt_fingerprint,
                            },
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    index_writer.touch(
                        kind="judge",
                        key=jk,
                        path=str(jpath),
                        meta={
                            "benchmark": manifest.benchmark,
                            "question_id": record.question_id,
                            "rejudge": True,
                        },
                    )
    finally:
        await judge_provider.close()

    run_dir = RunDir(out_dir)
    save_meta_json(run_dir.meta_path, new_manifest)
    save_run_file(run_dir.answers_path, new_manifest, records)
    _write_scorecard_artifacts(run_dir, new_manifest, records)


def _write_scorecard_artifacts(
    run_dir: RunDir,
    manifest: RunManifest,
    records: list[QARecord],
) -> None:
    # Lazy import — keeps rejudge usable with no rich dep in CI.
    from ..results.render import render_scorecard_markdown
    from ..results.scorecard import build_scorecard, scorecard_to_dict

    scorecard = build_scorecard(records, benchmark=manifest.benchmark)
    run_dir.scorecard_json.write_text(
        json.dumps(scorecard_to_dict(scorecard), indent=2), encoding="utf-8"
    )
    md = render_scorecard_markdown(scorecard, manifest=manifest)
    run_dir.scorecard_md.write_text(md, encoding="utf-8")


def _qa_from_record(record: QARecord) -> QAItem:
    """Reconstruct a minimal :class:`QAItem` from a stored :class:`QARecord`.

    ``evidence_turn_ids`` is preserved so LOCOMO-style category evidence
    remains available to future judges that consult it; metadata and the
    full session corpus are not needed — the judge prompts key off the
    question + gold + generated triple plus ``question_type`` /
    ``question_id``.
    """

    return QAItem(
        question_id=record.question_id or "",
        question=record.question,
        gold=record.gold,
        question_type=record.question_type or "",
        category=record.category,
        evidence_turn_ids=tuple(record.evidence_turn_ids),
        metadata=dict(record.metadata),
    )


def _build_benchmark_judge(
    dataset_name: str,
    *,
    client: JudgeClient,
    runs: int,
    temperature: float,
) -> BenchmarkJudge:
    return build_benchmark_judge(dataset_name, client=client, runs=runs, temperature=temperature)


__all__ = ["add_rejudge_subparser", "rejudge_command"]
