"""Tests for ``cli/summarize_cmd.py`` — rebuild + render a scorecard."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent_memory_benchmark.cli.summarize_cmd import summarize_command
from agent_memory_benchmark.runner.manifest import QARecord, RunManifest


def _sample_manifest() -> RunManifest:
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
        judge_prompt_fingerprint="deadbeef" * 8,
        dataset_name="longmemeval",
        dataset_split="s",
        dataset_path=None,
        dataset_descriptor_hash="0" * 64,
        hf_revision_sha="1" * 40,
        replicate_idx=0,
        replicate_seed=None,
        benchmark_git_sha="abcdef0",
        benchmark_git_branch="main",
        benchmark_git_dirty=False,
        benchmark_version="0.1.0",
        protocol_version="0.1",
        tag=None,
        cli_argv=["amb", "run"],
        timestamp_utc="2026-04-20T12:00:00Z",
    )


def _sample_record(key: str = "case1::0") -> QARecord:
    return QARecord(
        key=key,
        benchmark="longmemeval",
        case_id="case1",
        question="q",
        gold="g",
        generated="g",
        question_id="q1",
        question_type="general",
        category=None,
        qa_index=0,
        replicate_idx=0,
        total_answer_time_ms=100.0,
        judge_runs=[{"correct": True, "raw": "yes"}],
    )


def _write_answers(path: Path) -> None:
    import json

    payload = {
        "meta": asdict(_sample_manifest()),
        "records": [asdict(_sample_record())],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class TestSummarizeCommand:
    def test_missing_file_returns_2(self, tmp_path: Path, capsys: Any) -> None:
        args = argparse.Namespace(
            answers_path=tmp_path / "missing.json",
            format="markdown",
        )
        rc = summarize_command(args, argv=None)
        assert rc == 2

    def test_corrupt_file_returns_1(self, tmp_path: Path, capsys: Any) -> None:
        p = tmp_path / "answers.json"
        p.write_text("{ not json", encoding="utf-8")
        args = argparse.Namespace(answers_path=p, format="markdown")
        rc = summarize_command(args, argv=None)
        assert rc == 1

    def test_markdown_format(self, tmp_path: Path, capsys: Any) -> None:
        p = tmp_path / "answers.json"
        _write_answers(p)
        args = argparse.Namespace(answers_path=p, format="markdown")
        rc = summarize_command(args, argv=None)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Scorecard" in out
        assert "longmemeval" in out

    def test_rich_format_does_not_crash(self, tmp_path: Path, capsys: Any) -> None:
        p = tmp_path / "answers.json"
        _write_answers(p)
        args = argparse.Namespace(answers_path=p, format="rich")
        rc = summarize_command(args, argv=None)
        assert rc == 0
        # Rich output varies by tty detection; just verify something printed.
        out = capsys.readouterr().out
        assert out  # non-empty


class TestSummarizeSubparser:
    def test_parses(self) -> None:
        from agent_memory_benchmark.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["summarize", "some/path/answers.json", "--format", "markdown"]
        )
        assert args.command == "summarize"
        assert args.format == "markdown"
        assert isinstance(args.answers_path, Path)
