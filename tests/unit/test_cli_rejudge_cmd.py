"""Tests for ``cli/rejudge_cmd.py`` — re-judge an existing answers.json."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from agent_memory_benchmark.cli.rejudge_cmd import (
    _build_benchmark_judge,
    _qa_from_record,
    add_rejudge_subparser,
    rejudge_command,
)
from agent_memory_benchmark.llm import ChatResult
from agent_memory_benchmark.llm.judge_client import JudgeClient
from agent_memory_benchmark.runner.manifest import QARecord, RunManifest


class _FakeProvider:
    """Matches the :class:`LLMProvider` protocol with deterministic output."""

    spec = "fake:model"
    model = "model"

    def __init__(self, reply: str = "yes") -> None:
        self._reply = reply
        self.calls = 0

    async def chat(
        self,
        *,
        system: str = "",
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        self.calls += 1
        return ChatResult(text=self._reply, model=self.model)

    async def resolve_spec(self) -> str:
        return self.spec

    async def close(self) -> None:
        return None


def _sample_manifest(**overrides: Any) -> RunManifest:
    base = dict(
        benchmark="longmemeval",
        memory_system_id="full-context",
        memory_version="0.1.0",
        adapter_kind="full-context",
        adapter_target="full-context",
        answer_model_spec="ollama:llama3.1:8b",
        answer_model_resolved="ollama:llama3.1:8b@sha256:abc",
        judge_model_spec="ollama:llama3.1:70b",
        judge_model_resolved="ollama:llama3.1:70b@sha256:old",
        judge_temperature=0.0,
        judge_runs=1,
        judge_prompt_fingerprint="0" * 64,
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
    base.update(overrides)
    return RunManifest(**base)  # type: ignore[arg-type]


def _sample_record(
    key: str = "case1::0",
    generated: str = "a plausible answer",
    question_type: str = "single-session-user",
    question_id: str = "q1_single_session_user_1",
) -> QARecord:
    return QARecord(
        key=key,
        benchmark="longmemeval",
        case_id="case1",
        question="What is the capital of France?",
        gold="Paris",
        generated=generated,
        question_id=question_id,
        question_type=question_type,
        category=None,
        qa_index=0,
        replicate_idx=0,
        total_answer_time_ms=100.0,
        judge_runs=[{"correct": False, "raw": "no"}],  # intentionally stale
    )


def _write_answers(path: Path, *, records: list[QARecord] | None = None) -> None:
    payload = {
        "meta": asdict(_sample_manifest()),
        "records": [asdict(r) for r in (records or [_sample_record()])],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class TestRejudgeHelpers:
    def test_qa_from_record_preserves_fields(self) -> None:
        rec = _sample_record()
        qa = _qa_from_record(rec)
        assert qa.question == rec.question
        assert qa.gold == rec.gold
        assert qa.question_id == rec.question_id
        assert qa.question_type == rec.question_type

    def test_build_benchmark_judge_rejects_unsupported(self) -> None:
        client = JudgeClient(_FakeProvider())
        with pytest.raises(ValueError, match="Unknown dataset_name"):
            _build_benchmark_judge("squad", client=client, runs=1, temperature=0.0)

    def test_build_benchmark_judge_routes_locomo(self) -> None:
        client = JudgeClient(_FakeProvider())
        judge = _build_benchmark_judge("locomo", client=client, runs=10, temperature=0.0)
        from agent_memory_benchmark.runner.judge_adapter import LocomoJudge

        assert isinstance(judge, LocomoJudge)

    def test_build_benchmark_judge_routes_beam(self) -> None:
        client = JudgeClient(_FakeProvider())
        judge = _build_benchmark_judge("beam", client=client, runs=1, temperature=0.0)
        from agent_memory_benchmark.runner.judge_adapter import BeamJudge

        assert isinstance(judge, BeamJudge)


class TestRejudgeCommand:
    def test_missing_answers_returns_2(self, tmp_path: Path, capsys: Any) -> None:
        args = argparse.Namespace(
            answers_path=tmp_path / "nope.json",
            judge_model="ollama:x",
            judge_temperature=0.0,
            judge_runs=1,
            out=None,
            cache_root=tmp_path / "cache",
            no_cache=True,
            ollama_base_url=None,
            openai_base_url=None,
        )
        rc = rejudge_command(args, argv=None)
        assert rc == 2

    def test_corrupt_answers_returns_1(self, tmp_path: Path) -> None:
        p = tmp_path / "answers.json"
        p.write_text("{ not json", encoding="utf-8")
        args = argparse.Namespace(
            answers_path=p,
            judge_model="ollama:x",
            judge_temperature=0.0,
            judge_runs=1,
            out=None,
            cache_root=tmp_path / "cache",
            no_cache=True,
            ollama_base_url=None,
            openai_base_url=None,
        )
        rc = rejudge_command(args, argv=None)
        assert rc == 1

    def test_happy_path_rewrites_judge_verdicts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        answers = tmp_path / "answers.json"
        _write_answers(answers)

        fake_provider = _FakeProvider(reply="yes")

        def fake_build_provider(spec: str, **kwargs: Any) -> _FakeProvider:
            return fake_provider

        monkeypatch.setattr(
            "agent_memory_benchmark.cli.rejudge_cmd.build_provider",
            fake_build_provider,
        )

        out_dir = tmp_path / "out"
        args = argparse.Namespace(
            answers_path=answers,
            judge_model="ollama:new-judge",
            judge_temperature=0.0,
            judge_runs=1,
            out=out_dir,
            cache_root=tmp_path / "cache",
            no_cache=True,
            ollama_base_url=None,
            openai_base_url=None,
        )
        rc = rejudge_command(args, argv=["rejudge"])
        assert rc == 0

        # Output dir has all four artifacts.
        assert (out_dir / "answers.json").is_file()
        assert (out_dir / "scorecard.json").is_file()
        assert (out_dir / "scorecard.md").is_file()
        assert (out_dir / "meta.json").is_file()

        # Judge verdict was rewritten: the fake always says "yes".
        written = json.loads((out_dir / "answers.json").read_text(encoding="utf-8"))
        assert written["records"][0]["judge_runs"][0]["correct"] is True
        # Manifest reflects the new judge model.
        assert written["meta"]["judge_model_spec"] == "ollama:new-judge"

    def test_writes_judge_cache_when_enabled(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        answers = tmp_path / "answers.json"
        _write_answers(answers)

        fake_provider = _FakeProvider(reply="yes")
        monkeypatch.setattr(
            "agent_memory_benchmark.cli.rejudge_cmd.build_provider",
            lambda spec, **kw: fake_provider,
        )

        out_dir = tmp_path / "out"
        cache_root = tmp_path / "cache"
        args = argparse.Namespace(
            answers_path=answers,
            judge_model="ollama:new-judge",
            judge_temperature=0.0,
            judge_runs=1,
            out=out_dir,
            cache_root=cache_root,
            no_cache=False,
            ollama_base_url=None,
            openai_base_url=None,
        )
        rc = rejudge_command(args, argv=["rejudge"])
        assert rc == 0
        judge_dir = cache_root / "judge"
        assert judge_dir.is_dir()
        cached_files = list(judge_dir.glob("*.json"))
        assert len(cached_files) == 1
        payload = json.loads(cached_files[0].read_text(encoding="utf-8"))
        assert payload["judge_runs"][0]["correct"] is True

    def test_default_out_is_sibling_of_answers(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        answers = run_dir / "answers.json"
        _write_answers(answers)

        monkeypatch.setattr(
            "agent_memory_benchmark.cli.rejudge_cmd.build_provider",
            lambda spec, **kw: _FakeProvider(reply="yes"),
        )

        args = argparse.Namespace(
            answers_path=answers,
            judge_model="ollama:new-judge",
            judge_temperature=0.0,
            judge_runs=1,
            out=None,
            cache_root=tmp_path / "cache",
            no_cache=True,
            ollama_base_url=None,
            openai_base_url=None,
        )
        rc = rejudge_command(args, argv=["rejudge"])
        assert rc == 0
        rejudged = [p for p in run_dir.iterdir() if p.name.startswith("rejudged_")]
        assert len(rejudged) == 1
        assert (rejudged[0] / "answers.json").is_file()


class TestRejudgeSubparser:
    def test_required_judge_model(self) -> None:
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        add_rejudge_subparser(sub)
        with pytest.raises(SystemExit):
            parser.parse_args(["rejudge", "foo.json"])

    def test_happy_parse(self) -> None:
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        add_rejudge_subparser(sub)
        args = parser.parse_args(
            [
                "rejudge",
                "foo.json",
                "--judge-model",
                "ollama:x",
                "--judge-temperature",
                "0.3",
                "--judge-runs",
                "5",
                "--no-cache",
            ]
        )
        assert args.judge_model == "ollama:x"
        assert args.judge_temperature == 0.3
        assert args.judge_runs == 5
        assert args.no_cache is True
