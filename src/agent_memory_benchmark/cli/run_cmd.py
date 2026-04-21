"""``amb run <dataset>`` — the primary benchmark entry point.

Argument shapes track the plan's CLI sketch closely. A few deliberate
choices worth noting:

- ``--judge-model`` is **required** (no default). The plan enforces
  conscious opt-in for paid-API calls; Ollama-local judges still have
  to be named explicitly.
- ``--memory-config key=value`` repeats to build the target adapter's
  kwargs dict; values are parsed as JSON when the string is valid JSON,
  otherwise left as strings (so ``--memory-config timeout=30`` yields an
  int, ``--memory-config model=llama3`` yields a string).
- ``--limit-strategy`` defaults to ``stratified`` for ``longmemeval`` to
  keep early-truncation runs balanced across question types.

``run_command`` is invoked synchronously by the CLI dispatcher; it opens
its own asyncio event loop for :func:`run_benchmark`. Keep it
``argparse.Namespace`` in, ``int`` out so the CLI composition stays plain.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from ..runner import run_benchmark


def add_run_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Register the ``run`` subcommand on the root parser."""

    parser: argparse.ArgumentParser = subparsers.add_parser(
        "run",
        help="Execute a benchmark run end-to-end.",
        description="Run a benchmark dataset through a memory adapter + LLM pair.",
    )
    _add_shared_run_arguments(parser, include_memory=True)
    return parser


def _add_shared_run_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_memory: bool,
) -> None:
    """Add the argument surface shared by ``run`` and ``baseline``.

    ``include_memory=False`` is used by ``baseline``, which hardwires
    ``--memory full-context`` in its command handler — the flag would
    only confuse users who already picked the shortcut.
    """

    parser.add_argument(
        "dataset",
        choices=("longmemeval", "locomo", "beam"),
        help="Dataset name. longmemeval is wired in PR-7; locomo (PR-9) / beam (PR-11) will "
        "raise DatasetUnavailableError until their loaders land.",
    )
    if include_memory:
        parser.add_argument(
            "--memory",
            required=True,
            help="Memory adapter spec (e.g. 'full-context', 'python:pkg.mod:Cls', "
            "'http://host').",
        )
    parser.add_argument("--answer-model", required=True, help="LLM spec: <provider>:<model>.")
    parser.add_argument(
        "--judge-model",
        required=True,
        help="Judge LLM spec. Required to enforce conscious opt-in to paid-API usage.",
    )
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-runs", type=int, default=1)
    parser.add_argument("--split", help="Dataset split (required for longmemeval: 's' or 'm').")
    parser.add_argument(
        "--m-data",
        type=Path,
        help="Local path to LongMemEval M JSON (multi-GB; not auto-downloaded).",
    )
    parser.add_argument("--limit", type=int, help="Truncate dataset to N questions.")
    parser.add_argument(
        "--limit-strategy",
        choices=("full", "stratified", "head"),
        default="stratified",
    )
    if include_memory:
        parser.add_argument(
            "--memory-config",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="Key=value pairs forwarded to the adapter. Repeat as needed.",
        )
        parser.add_argument(
            "--session-mapper",
            metavar="pkg.module:function",
            help="Callable that converts benchmark Session objects before ingest. "
            "Use when the target's class signature already matches MemorySystemShape "
            "but its ingest_session expects a different Session/Turn type. For "
            "class-signature divergence, use a wrapper shim under compat/ instead.",
        )
        parser.add_argument(
            "--result-mapper",
            metavar="pkg.module:function",
            help="Callable that converts the target's answer return into the "
            "benchmark's AnswerResult. Pair with --session-mapper when needed.",
        )
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--tag", help="Short tag appended to the run directory name.")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", default=True)
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable answer + judge cache reads (writes still happen).",
    )
    parser.add_argument("--no-cache-ingestion", action="store_true")
    parser.add_argument("--replicate-idx", type=int, default=0)
    parser.add_argument("--replicate-seed", type=int)
    parser.add_argument(
        "--ollama-base-url", help="Override Ollama base URL (default http://localhost:11434)."
    )
    parser.add_argument("--openai-base-url")


def run_command(args: argparse.Namespace, *, argv: list[str] | None = None) -> int:
    """Synchronous CLI entry; returns process exit code."""

    # baseline_cmd calls through here with memory-specific flags absent;
    # fill in the full-context defaults so downstream code doesn't branch.
    raw_memory_config = getattr(args, "memory_config", []) or []
    try:
        memory_config = _parse_memory_config(raw_memory_config)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    memory_spec = getattr(args, "memory", None) or "full-context"
    session_mapper = getattr(args, "session_mapper", None)
    result_mapper = getattr(args, "result_mapper", None)

    try:
        run_dir = asyncio.run(
            run_benchmark(
                dataset_name=args.dataset,
                memory_spec=memory_spec,
                answer_model_spec=args.answer_model,
                judge_model_spec=args.judge_model,
                judge_temperature=args.judge_temperature,
                judge_runs=args.judge_runs,
                split=args.split,
                m_data_path=args.m_data,
                dataset_limit=args.limit,
                dataset_limit_strategy=args.limit_strategy,
                memory_config=memory_config or None,
                session_mapper_spec=session_mapper,
                result_mapper_spec=result_mapper,
                results_base=args.out,
                cache_root=args.cache_root,
                tag=args.tag,
                resume=args.resume,
                no_cache=args.no_cache,
                no_cache_ingestion=args.no_cache_ingestion,
                replicate_idx=args.replicate_idx,
                replicate_seed=args.replicate_seed,
                cli_argv=argv if argv is not None else sys.argv,
                ollama_base_url=args.ollama_base_url,
                openai_base_url=args.openai_base_url,
            )
        )
    except Exception as e:  # noqa: BLE001  # CLI boundary — any failure becomes a printed error
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"Run complete. Output: {run_dir.path}")
    _render_summary(run_dir)
    return 0


def _parse_memory_config(items: list[str]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--memory-config entry {item!r} must be KEY=VALUE")
        key, _, raw = item.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"--memory-config entry {item!r} has empty key")
        raw = raw.strip()
        try:
            config[key] = json.loads(raw)
        except json.JSONDecodeError:
            config[key] = raw
    return config


def _render_summary(run_dir: Any) -> None:
    from ..results.render import print_scorecard_rich
    from ..results.scorecard import build_scorecard
    from ..runner.manifest import load_run_file

    try:
        manifest, records = load_run_file(run_dir.answers_path)
    except (OSError, KeyError) as e:  # pragma: no cover
        print(f"(could not render rich summary: {e})", file=sys.stderr)
        return
    scorecard = build_scorecard(list(records.values()), benchmark=manifest.benchmark)
    print_scorecard_rich(scorecard, manifest=manifest)


__all__ = [
    "_add_shared_run_arguments",
    "add_run_subparser",
    "run_command",
]
