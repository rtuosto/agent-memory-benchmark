"""End-to-end orchestrator tests using fake LLM providers + FullContextAdapter.

Exercises the cache-aware ingest → answer → judge loop without touching
Ollama or OpenAI. The fake provider returns deterministic canned replies
keyed by the last chars of the user message so each test can vary what
the "LLM" says.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from agent_memory_benchmark.adapters.full_context import FullContextAdapter
from agent_memory_benchmark.datasets.base import DatasetAdapter
from agent_memory_benchmark.judge.longmemeval import LME_JUDGE_FINGERPRINT
from agent_memory_benchmark.llm import ChatResult
from agent_memory_benchmark.llm.judge_client import JudgeClient
from agent_memory_benchmark.runner.judge_adapter import LongMemEvalJudge
from agent_memory_benchmark.runner.manifest import RunDir, RunManifest
from agent_memory_benchmark.runner.orchestrator import BenchmarkRunner
from agent_memory_benchmark.types import BenchmarkCase, QAItem, Session, Turn


class FakeProvider:
    """Canned-response LLM. Matches the ``LLMProvider`` Protocol shape."""

    def __init__(self, *, model: str, answer: str = "a shell necklace") -> None:
        self.model = model
        self.spec = f"fake:{model}"
        self._answer = answer
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        *,
        system: str = "",
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        return ChatResult(text=self._answer, model=self.model, prompt_tokens=0, completion_tokens=0)

    async def resolve_spec(self) -> str:
        return f"{self.spec}@sha256:{'0' * 64}"

    async def close(self) -> None:
        return None


def _case(case_id: str = "c1", qa_id: str = "q1") -> BenchmarkCase:
    session = Session(
        session_index=1,
        turns=(
            Turn(turn_id="t1", speaker="user", text="I love shell necklaces."),
            Turn(turn_id="t2", speaker="assistant", text="Nice!"),
        ),
        session_time="2026-04-20",
        session_id="sess_1",
    )
    qa = QAItem(
        question_id=qa_id,
        question="What did I say I love?",
        gold="a shell necklace",
        question_type="single-session-user",
        evidence_turn_ids=("t1",),
    )
    return BenchmarkCase(
        case_id=case_id,
        sessions=(session,),
        qa=(qa,),
        dataset="longmemeval",
    )


class StaticDataset(DatasetAdapter):
    """Tiny in-memory dataset. Only used inside tests."""

    name = "longmemeval"

    def __init__(self, cases: list[BenchmarkCase]) -> None:
        self._cases = cases

    def __iter__(self) -> Iterator[BenchmarkCase]:
        yield from self._cases

    def __len__(self) -> int:
        return len(self._cases)

    def descriptor_hash(self) -> str:
        return "deadbeef" * 8


def _manifest() -> RunManifest:
    return RunManifest(
        benchmark="longmemeval",
        memory_system_id="full-context",
        memory_version="0.1.0",
        adapter_kind="full-context",
        adapter_target="full-context",
        answer_model_spec="fake:answer",
        answer_model_resolved="fake:answer@sha256:" + "0" * 64,
        judge_model_spec="fake:judge",
        judge_model_resolved="fake:judge@sha256:" + "0" * 64,
        judge_temperature=0.0,
        judge_runs=1,
        judge_prompt_fingerprint=LME_JUDGE_FINGERPRINT,
        dataset_name="longmemeval",
        dataset_split="s",
        dataset_path=None,
        dataset_descriptor_hash="deadbeef" * 8,
        hf_revision_sha=None,
        replicate_idx=0,
        replicate_seed=None,
        benchmark_git_sha=None,
        benchmark_git_branch=None,
        benchmark_git_dirty=None,
        benchmark_version="0.1.0",
        protocol_version="0.1",
        tag=None,
        cli_argv=[],
        timestamp_utc="2026-04-20T00:00:00Z",
    )


def _build_runner(
    *,
    cases: list[BenchmarkCase],
    tmp_path: Path,
    answer_text: str = "a shell necklace",
    judge_text: str = "yes",
    resume: bool = True,
    use_ingestion_cache: bool = True,
    use_answer_cache: bool = True,
    use_judge_cache: bool = True,
) -> tuple[BenchmarkRunner, FakeProvider, FakeProvider, Path]:
    answer_provider = FakeProvider(model="answer", answer=answer_text)
    judge_provider = FakeProvider(model="judge", answer=judge_text)
    adapter = FullContextAdapter(answer_provider)
    judge_client = JudgeClient(judge_provider, temperature=0.0)
    judge = LongMemEvalJudge(
        judge_client,
        runs=1,
        temperature=0.0,
        bundle_fingerprint=LME_JUDGE_FINGERPRINT,
    )
    run_dir_path = tmp_path / "run"
    run_dir = RunDir(run_dir_path)
    cache_root = tmp_path / "cache"
    results_base = tmp_path / "results"
    results_base.mkdir(parents=True, exist_ok=True)

    runner = BenchmarkRunner(
        dataset=StaticDataset(cases),
        adapter=adapter,
        judge=judge,
        manifest=_manifest(),
        run_dir=run_dir,
        cache_root=cache_root,
        results_base=results_base,
        dataset_descriptor_hash="deadbeef" * 8,
        answer_model_spec="fake:answer@sha256:" + "0" * 64,
        judge_model_spec="fake:judge@sha256:" + "0" * 64,
        judge_temperature=0.0,
        judge_runs=1,
        benchmark_name="longmemeval",
        use_ingestion_cache=use_ingestion_cache,
        use_answer_cache=use_answer_cache,
        use_judge_cache=use_judge_cache,
        resume=resume,
    )
    return runner, answer_provider, judge_provider, cache_root


def test_end_to_end_single_case(tmp_path: Path) -> None:
    runner, answer_p, judge_p, cache_root = _build_runner(cases=[_case()], tmp_path=tmp_path)
    records = asyncio.run(runner.run())

    assert len(records) == 1
    rec = records[0]
    assert rec.case_id == "c1"
    assert rec.question_id == "q1"
    assert rec.generated == "a shell necklace"
    assert rec.judge_runs == [{"correct": True, "raw": "yes"}]
    assert rec.total_answer_time_ms > 0
    # FullContext reports 2 turns = units_retrieved; tokens is whitespace est.
    assert rec.units_retrieved == 2
    assert rec.tokens_retrieved > 0
    assert rec.evidence_turn_ids == ["t1"]
    # answers.json was persisted.
    assert runner._run_dir.answers_path.is_file()  # noqa: SLF001 (test accesses internal path)
    payload = json.loads(runner._run_dir.answers_path.read_text(encoding="utf-8"))  # noqa: SLF001
    assert payload["records"][0]["generated"] == "a shell necklace"
    # Cache writes happened for answer + judge.
    assert any(cache_root.rglob("state.json"))
    assert any((cache_root / "answers").rglob("*.json"))
    assert any((cache_root / "judge").rglob("*.json"))


def test_answer_cache_hit_skips_llm_call(tmp_path: Path) -> None:
    cases = [_case()]
    # First run — primes both caches.
    runner1, ap1, _, _ = _build_runner(cases=cases, tmp_path=tmp_path)
    asyncio.run(runner1.run())
    first_answer_calls = len(ap1.calls)
    assert first_answer_calls >= 1

    # Second run — fresh runner, same tmp_path → answer + judge should both hit.
    runner2, ap2, jp2, _ = _build_runner(cases=cases, tmp_path=tmp_path)
    # Delete answers.json to force re-execution of the loop without resume.
    runner2._run_dir.answers_path.unlink(missing_ok=True)  # noqa: SLF001
    asyncio.run(runner2.run())
    # Answer LLM not called (answer cache hit).
    assert len(ap2.calls) == 0
    # Judge LLM not called (judge cache hit — we re-derive verdicts from cache).
    assert len(jp2.calls) == 0


def test_no_cache_forces_regeneration(tmp_path: Path) -> None:
    cases = [_case()]
    runner1, ap1, jp1, _ = _build_runner(cases=cases, tmp_path=tmp_path)
    asyncio.run(runner1.run())

    runner2, ap2, jp2, _ = _build_runner(
        cases=cases,
        tmp_path=tmp_path,
        use_answer_cache=False,
        use_judge_cache=False,
    )
    runner2._run_dir.answers_path.unlink(missing_ok=True)  # noqa: SLF001
    asyncio.run(runner2.run())
    assert len(ap2.calls) >= 1
    assert len(jp2.calls) >= 1


def test_resume_skips_already_completed_qa(tmp_path: Path) -> None:
    cases = [_case()]
    runner1, _, _, _ = _build_runner(cases=cases, tmp_path=tmp_path)
    asyncio.run(runner1.run())

    runner2, ap2, jp2, _ = _build_runner(cases=cases, tmp_path=tmp_path, resume=True)
    records = asyncio.run(runner2.run())
    assert len(records) == 1
    # Resume hit — no LLM traffic for answer or judge.
    assert ap2.calls == []
    assert jp2.calls == []


def test_judge_parses_no_verdict_when_response_starts_with_no(tmp_path: Path) -> None:
    runner, _, _, _ = _build_runner(
        cases=[_case()], tmp_path=tmp_path, judge_text="no, that is not right"
    )
    records = asyncio.run(runner.run())
    assert records[0].judge_runs == [{"correct": False, "raw": "no, that is not right"}]


def test_ingestion_state_cache_round_trips(tmp_path: Path) -> None:
    """Second fresh runner reloads state instead of calling ingest_session."""

    runner1, _, _, cache_root = _build_runner(cases=[_case()], tmp_path=tmp_path)
    asyncio.run(runner1.run())
    # Verify the ingestion state file was written by FullContextAdapter.
    state_files = list(cache_root.rglob("state.json"))
    assert state_files, "expected ingestion cache state.json to be written"

    # Wipe the run dir + answers.json so orchestrator can't resume by key.
    runner2, ap2, _, _ = _build_runner(cases=[_case()], tmp_path=tmp_path)
    runner2._run_dir.answers_path.unlink(missing_ok=True)  # noqa: SLF001
    records = asyncio.run(runner2.run())
    assert records[0].ingestion_time_ms == 0.0  # cache hit → runner-measured zero
