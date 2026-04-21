"""``amb baseline`` — ``amb run ... --memory full-context`` shortcut.

Semantically identical to ``amb run``; the wrapper exists because the
full-context null-memory run is the recommended smoke test for *any*
combination of dataset + answer model, and hiding the ``--memory`` knob
makes it unambiguous which baseline is being executed. All other flags
are passed through to :func:`~.run_cmd.run_command`.
"""

from __future__ import annotations

import argparse

from .run_cmd import _add_shared_run_arguments, run_command


def add_baseline_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Register ``amb baseline`` on the root parser."""

    parser: argparse.ArgumentParser = subparsers.add_parser(
        "baseline",
        help="Run the full-context null-memory baseline (no retrieval).",
        description=(
            "Shortcut for 'amb run ... --memory full-context'. Intended as a "
            "sanity-check that a dataset + answer model combination works end "
            "to end; the baseline retrieves nothing, so evidence KPIs report "
            "n_questions_with_retrieval = 0."
        ),
    )
    _add_shared_run_arguments(parser, include_memory=False)
    return parser


def baseline_command(args: argparse.Namespace, *, argv: list[str] | None = None) -> int:
    """Synchronous CLI entry; delegates to :func:`run_command` with a pinned spec."""

    # argparse won't have populated these on the baseline subparser — fill
    # them in so run_command doesn't branch. getattr with a default keeps
    # this robust if someone adds the flags to baseline in the future.
    args.memory = "full-context"
    args.memory_config = []
    args.memory_header = []
    args.session_mapper = None
    args.result_mapper = None
    return run_command(args, argv=argv)


__all__ = ["add_baseline_subparser", "baseline_command"]
