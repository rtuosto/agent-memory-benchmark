"""Tests for ``cli/run_cmd.py`` arg parsing + value coercion.

The full ``run_command`` integration path goes through network-ish code
(``build_provider``) so these tests focus on the pieces we can verify
without running the orchestrator: arg shape, memory-config parsing, and
error handling when the command can't assemble.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_memory_benchmark.cli.main import build_parser
from agent_memory_benchmark.cli.run_cmd import (
    _parse_memory_config,
    _resolve_num_ctx,
    add_run_subparser,
)


def test_parse_memory_config_parses_json_and_strings() -> None:
    cfg = _parse_memory_config(
        [
            "timeout=30",
            "model=llama3.1:8b",
            "flag=true",
            'json_obj={"k": 1}',
        ]
    )
    assert cfg["timeout"] == 30
    assert cfg["model"] == "llama3.1:8b"
    assert cfg["flag"] is True
    assert cfg["json_obj"] == {"k": 1}


def test_parse_memory_config_rejects_missing_equals() -> None:
    with pytest.raises(ValueError, match="must be KEY=VALUE"):
        _parse_memory_config(["bare_token"])


def test_parse_memory_config_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="empty key"):
        _parse_memory_config(["=value"])


def test_parser_rejects_missing_required_args() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "longmemeval"])


def test_parser_accepts_full_longmemeval_invocation() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "longmemeval",
            "--memory",
            "full-context",
            "--answer-model",
            "ollama:llama3.1:8b",
            "--judge-model",
            "ollama:llama3.1:70b",
            "--split",
            "s",
            "--limit",
            "5",
        ]
    )
    assert args.command == "run"
    assert args.dataset == "longmemeval"
    assert args.memory == "full-context"
    assert args.answer_model == "ollama:llama3.1:8b"
    assert args.judge_model == "ollama:llama3.1:70b"
    assert args.split == "s"
    assert args.limit == 5
    assert args.resume is True
    assert args.no_cache is False
    assert args.limit_strategy == "stratified"


def test_parser_no_resume_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "longmemeval",
            "--memory",
            "full-context",
            "--answer-model",
            "ollama:x",
            "--judge-model",
            "ollama:y",
            "--no-resume",
        ]
    )
    assert args.resume is False


def test_parser_adapters_and_out_paths_are_paths() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "longmemeval",
            "--memory",
            "full-context",
            "--answer-model",
            "ollama:x",
            "--judge-model",
            "ollama:y",
            "--out",
            "./my/results",
            "--cache-root",
            "./my/cache",
        ]
    )
    assert isinstance(args.out, Path)
    assert isinstance(args.cache_root, Path)


def test_add_run_subparser_exposes_dataset_choices() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_run_subparser(subparsers)
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "unknown-bench",
                "--memory",
                "full-context",
                "--answer-model",
                "x",
                "--judge-model",
                "y",
            ]
        )


def test_parser_captures_session_and_result_mapper_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "longmemeval",
            "--memory",
            "python:pkg.mod:Cls",
            "--answer-model",
            "ollama:x",
            "--judge-model",
            "ollama:y",
            "--session-mapper",
            "agent_memory_benchmark.compat.engram_shim:_to_engram_session",
            "--result-mapper",
            "agent_memory_benchmark.compat.engram_shim:_from_engram_answer",
        ]
    )
    assert args.session_mapper == "agent_memory_benchmark.compat.engram_shim:_to_engram_session"
    assert args.result_mapper == "agent_memory_benchmark.compat.engram_shim:_from_engram_answer"


class TestNumCtxFlags:
    def test_specific_beats_shared(self) -> None:
        import argparse

        args = argparse.Namespace(answer_num_ctx=131072, judge_num_ctx=None, num_ctx=8192)
        assert _resolve_num_ctx(args, "answer_num_ctx") == 131072
        assert _resolve_num_ctx(args, "judge_num_ctx") == 8192

    def test_shared_applies_when_specific_absent(self) -> None:
        import argparse

        args = argparse.Namespace(answer_num_ctx=None, judge_num_ctx=None, num_ctx=65536)
        assert _resolve_num_ctx(args, "answer_num_ctx") == 65536
        assert _resolve_num_ctx(args, "judge_num_ctx") == 65536

    def test_none_when_nothing_set(self) -> None:
        import argparse

        args = argparse.Namespace(answer_num_ctx=None, judge_num_ctx=None, num_ctx=None)
        assert _resolve_num_ctx(args, "answer_num_ctx") is None
        assert _resolve_num_ctx(args, "judge_num_ctx") is None

    def test_parser_accepts_num_ctx_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "longmemeval",
                "--memory",
                "full-context",
                "--answer-model",
                "ollama:x",
                "--judge-model",
                "ollama:y",
                "--num-ctx",
                "131072",
                "--answer-num-ctx",
                "65536",
                "--judge-num-ctx",
                "8192",
            ]
        )
        assert args.num_ctx == 131072
        assert args.answer_num_ctx == 65536
        assert args.judge_num_ctx == 8192

    def test_baseline_also_accepts_num_ctx(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "baseline",
                "longmemeval",
                "--answer-model",
                "ollama:x",
                "--judge-model",
                "ollama:y",
                "--num-ctx",
                "131072",
            ]
        )
        assert args.num_ctx == 131072


def test_parser_mapper_flags_default_to_none() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "longmemeval",
            "--memory",
            "full-context",
            "--answer-model",
            "ollama:x",
            "--judge-model",
            "ollama:y",
        ]
    )
    assert args.session_mapper is None
    assert args.result_mapper is None
