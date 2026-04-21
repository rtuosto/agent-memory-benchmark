"""Tests for ``cli/baseline_cmd.py`` arg shape + dispatch to run_command."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from agent_memory_benchmark.cli.baseline_cmd import (
    add_baseline_subparser,
    baseline_command,
)
from agent_memory_benchmark.cli.main import build_parser


class TestBaselineSubparser:
    def test_omits_memory_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "baseline",
                "longmemeval",
                "--answer-model",
                "ollama:llama3.1:8b",
                "--judge-model",
                "ollama:llama3.1:70b",
                "--split",
                "s",
                "--limit",
                "3",
            ]
        )
        assert args.command == "baseline"
        assert args.dataset == "longmemeval"
        assert not hasattr(args, "memory")
        assert not hasattr(args, "memory_config")
        assert not hasattr(args, "session_mapper")

    def test_rejects_memory_flag(self) -> None:
        """``--memory`` on baseline is nonsensical — parser must error."""

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "baseline",
                    "longmemeval",
                    "--memory",
                    "full-context",
                    "--answer-model",
                    "ollama:x",
                    "--judge-model",
                    "ollama:y",
                ]
            )

    def test_shares_run_style_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "baseline",
                "longmemeval",
                "--answer-model",
                "ollama:x",
                "--judge-model",
                "ollama:y",
                "--out",
                "./my/out",
                "--cache-root",
                "./my/cache",
                "--no-cache",
                "--replicate-idx",
                "2",
            ]
        )
        assert isinstance(args.out, Path)
        assert isinstance(args.cache_root, Path)
        assert args.no_cache is True
        assert args.replicate_idx == 2


class TestBaselineCommand:
    def test_forwards_with_full_context_memory(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``baseline_command`` must fill in ``memory=full-context`` before dispatch."""

        captured: dict[str, Any] = {}

        def fake_run_command(args: argparse.Namespace, *, argv: list[str] | None) -> int:
            captured["memory"] = args.memory
            captured["memory_config"] = args.memory_config
            captured["session_mapper"] = args.session_mapper
            captured["result_mapper"] = args.result_mapper
            captured["argv"] = argv
            return 0

        monkeypatch.setattr(
            "agent_memory_benchmark.cli.baseline_cmd.run_command",
            fake_run_command,
        )

        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers(dest="command")
        add_baseline_subparser(subs)
        args = parser.parse_args(
            [
                "baseline",
                "longmemeval",
                "--answer-model",
                "ollama:x",
                "--judge-model",
                "ollama:y",
            ]
        )
        rc = baseline_command(args, argv=["baseline", "longmemeval"])
        assert rc == 0
        assert captured["memory"] == "full-context"
        assert captured["memory_config"] == []
        assert captured["session_mapper"] is None
        assert captured["result_mapper"] is None
        assert captured["argv"] == ["baseline", "longmemeval"]
