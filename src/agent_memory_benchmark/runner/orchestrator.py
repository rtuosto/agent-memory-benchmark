"""The ingest→answer→judge loop. Cache-aware, resume-aware.

The orchestrator is the only place in the benchmark where the runner
actually talks to a memory adapter. Everything else (judge templates,
dataset loaders, scorecards) is accessed via the objects passed in here.

Cache behavior:

- **Ingestion cache** (``cache/ingestion/<mem>/<key>/state.json``) — only
  consulted when the adapter reports ``supports_persistence``. A hit
  replaces ``_ingest_case`` with ``adapter.load_state`` + a recorded
  ``ingest_total_ms == 0`` (the cached state had no cost to regenerate in
  this run). A miss runs through every session in ``case.sessions`` and
  writes the saved state back when the adapter allows it.
- **Answer cache** (``cache/answers/<key>.json``) — keyed by the
  byte-stable :func:`~..cache.keys.answer_key`. On hit we still *re-judge*
  unless the judge cache has that verdict too — this matches the
  predecessor and lets ``amb rejudge`` reuse stored generations without
  re-hitting the answer LLM.
- **Judge cache** (``cache/judge/<key>.json``) — keyed by the template
  fingerprint + question+gold+generated triple. A verdict is only written
  after we've parsed yes/no successfully, and the fingerprint used in the
  key is the *template* fingerprint (not the formatted prompt) so
  re-baselining a template invalidates the right cohort.

Resume: ``answers.json`` is read back in at startup; any QA whose key is
already present is skipped. The runner's in-memory ``records`` list and
the file on disk are kept in lock-step via :func:`save_run_file` after
every QA so Ctrl-C mid-run doesn't lose progress.

Telemetry drift: ``answer_total - (retrieval_self + generation_self)`` is
stored per-record as ``answer_discrepancy_ms``. Big numbers are a signal
that the adapter is under-reporting time somewhere (serialization,
transport, queueing). The scorecard exposes the distribution.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from ..adapters.base import MemoryAdapter
from ..cache.index import CacheIndexWriter
from ..cache.keys import (
    answer_cache_path,
    answer_key,
    ingestion_key,
    ingestion_state_path,
    judge_cache_path,
    judge_key,
)
from ..datasets.base import DatasetAdapter
from ..types import AnswerResult, BenchmarkCase, QAItem
from .judge_adapter import BenchmarkJudge
from .manifest import (
    QARecord,
    RunDir,
    RunManifest,
    load_run_file,
    save_meta_json,
    save_run_file,
)

_log = logging.getLogger(__name__)


class BenchmarkRunner:
    """Drives one benchmark run from dataset to scorecard artifacts.

    Construct once per run. The orchestrator is intentionally stateless
    between cases — state lives in the adapter, the cache directory, and
    the in-memory ``records`` list that's persisted after every QA.
    """

    def __init__(
        self,
        *,
        dataset: DatasetAdapter,
        adapter: MemoryAdapter,
        judge: BenchmarkJudge,
        manifest: RunManifest,
        run_dir: RunDir,
        cache_root: Path,
        results_base: Path,
        dataset_descriptor_hash: str,
        answer_model_spec: str,
        judge_model_spec: str,
        judge_temperature: float,
        judge_runs: int,
        benchmark_name: str,
        use_ingestion_cache: bool = True,
        use_answer_cache: bool = True,
        use_judge_cache: bool = True,
        resume: bool = True,
        replicate_idx: int = 0,
    ) -> None:
        self._dataset = dataset
        self._adapter = adapter
        self._judge = judge
        self._manifest = manifest
        self._run_dir = run_dir
        self._cache_root = cache_root
        self._results_base = results_base
        self._dataset_descriptor_hash = dataset_descriptor_hash
        self._answer_model_spec = answer_model_spec
        self._judge_model_spec = judge_model_spec
        self._judge_temperature = judge_temperature
        self._judge_runs = judge_runs
        self._benchmark_name = benchmark_name
        self._use_ingestion_cache = use_ingestion_cache
        self._use_answer_cache = use_answer_cache
        self._use_judge_cache = use_judge_cache
        self._resume = resume
        self._replicate_idx = replicate_idx

    async def run(self) -> list[QARecord]:
        """Execute the full benchmark. Returns the in-memory records."""

        self._run_dir.path.mkdir(parents=True, exist_ok=True)
        save_meta_json(self._run_dir.meta_path, self._manifest)

        records, existing_keys = self._load_resume_state()

        with CacheIndexWriter(self._cache_root) as index_writer:
            for case in self._dataset:
                await self._adapter.reset()
                ingest_total_ms = await self._ingest_case(case, index_writer=index_writer)

                for qa_index, qa in enumerate(case.qa):
                    key = _record_key(case.case_id, qa_index, self._replicate_idx)
                    if self._resume and key in existing_keys:
                        continue

                    record = await self._answer_and_judge_qa(
                        case=case,
                        qa=qa,
                        qa_index=qa_index,
                        key=key,
                        ingest_total_ms=ingest_total_ms,
                        index_writer=index_writer,
                    )
                    records.append(record)
                    existing_keys.add(key)
                    save_run_file(self._run_dir.answers_path, self._manifest, records)

        return records

    def _load_resume_state(self) -> tuple[list[QARecord], set[str]]:
        if not (self._resume and self._run_dir.answers_path.is_file()):
            return [], set()
        try:
            _, record_map = load_run_file(self._run_dir.answers_path)
        except (json.JSONDecodeError, KeyError, OSError) as e:
            _log.warning(
                "Could not resume from %s (%s); starting fresh.",
                self._run_dir.answers_path,
                e,
            )
            return [], set()
        records = list(record_map.values())
        return records, set(record_map.keys())

    async def _ingest_case(
        self,
        case: BenchmarkCase,
        *,
        index_writer: CacheIndexWriter,
    ) -> float:
        """Load cached state or ingest every session; return total ingest ms."""

        ing_key = ingestion_key(
            self._adapter.memory_system_id,
            self._adapter.memory_version,
            self._dataset_descriptor_hash,
            case.case_id,
        )
        state_path = ingestion_state_path(self._cache_root, self._adapter.memory_system_id, ing_key)

        if (
            self._use_ingestion_cache
            and self._adapter.supports_persistence
            and state_path.is_file()
        ):
            try:
                await self._adapter.load_state(state_path.parent)
                return 0.0
            except (NotImplementedError, json.JSONDecodeError, KeyError, OSError, ValueError) as e:
                _log.debug("ingestion cache miss on %s: %s", state_path, e)

        ingest_total = 0.0
        for session in sorted(case.sessions, key=lambda s: s.session_index):
            t0 = time.perf_counter()
            await self._adapter.ingest_session(session, case.case_id)
            ingest_total += (time.perf_counter() - t0) * 1000.0

        if self._use_ingestion_cache and self._adapter.supports_persistence:
            try:
                await self._adapter.save_state(state_path.parent)
                index_writer.touch(
                    kind="ingestion",
                    key=ing_key,
                    path=str(state_path),
                    meta={"case_id": case.case_id},
                )
            except NotImplementedError:
                pass
            except OSError as e:
                _log.warning("could not save ingestion state for %s: %s", case.case_id, e)

        return ingest_total

    async def _answer_and_judge_qa(
        self,
        *,
        case: BenchmarkCase,
        qa: QAItem,
        qa_index: int,
        key: str,
        ingest_total_ms: float,
        index_writer: CacheIndexWriter,
    ) -> QARecord:
        """Produce one :class:`QARecord` — using caches where possible."""

        ak = answer_key(
            self._adapter.memory_system_id,
            self._adapter.memory_version,
            self._dataset_descriptor_hash,
            self._answer_model_spec,
            key,
            qa.question,
            replicate_idx=self._replicate_idx,
        )
        ans_path = answer_cache_path(self._cache_root, ak)

        record = self._load_cached_answer(ans_path) if self._use_answer_cache else None
        if record is not None:
            record.ingestion_time_ms = ingest_total_ms
        else:
            record = await self._generate_answer(
                case=case,
                qa=qa,
                qa_index=qa_index,
                key=key,
                ingest_total_ms=ingest_total_ms,
            )

        if not record.judge_runs:
            await self._judge_record(record, qa, index_writer=index_writer)

        self._save_answer_cache(ans_path, record)
        index_writer.touch(
            kind="answers",
            key=ak,
            path=str(ans_path),
            meta={"case_id": case.case_id, "question_id": qa.question_id},
        )
        return record

    async def _generate_answer(
        self,
        *,
        case: BenchmarkCase,
        qa: QAItem,
        qa_index: int,
        key: str,
        ingest_total_ms: float,
    ) -> QARecord:
        t_ans = time.perf_counter()
        answer = await self._adapter.answer_question(qa.question, case.case_id)
        total_ms = (time.perf_counter() - t_ans) * 1000.0
        discrepancy = total_ms - (answer.retrieval_time_ms + answer.generation_time_ms)

        return QARecord(
            key=key,
            benchmark=self._benchmark_name,
            case_id=case.case_id,
            question=qa.question,
            gold=qa.gold,
            generated=answer.answer,
            question_id=qa.question_id,
            question_type=qa.question_type,
            category=qa.category,
            qa_index=qa_index,
            replicate_idx=self._replicate_idx,
            ingestion_time_ms=ingest_total_ms,
            retrieval_time_ms=answer.retrieval_time_ms,
            generation_time_ms=answer.generation_time_ms,
            total_answer_time_ms=total_ms,
            answer_discrepancy_ms=discrepancy,
            units_retrieved=answer.units_retrieved,
            tokens_retrieved=answer.tokens_retrieved,
            evidence_turn_ids=list(qa.evidence_turn_ids),
            retrieved_turn_ids=_retrieved_turn_ids(answer),
            evidence_texts=_evidence_texts(case, qa),
            retrieved_texts=[unit.text for unit in answer.retrieved],
            metadata=dict(qa.metadata),
        )

    async def _judge_record(
        self,
        record: QARecord,
        qa: QAItem,
        *,
        index_writer: CacheIndexWriter,
    ) -> None:
        cached = self._load_cached_judge(qa, record.generated) if self._use_judge_cache else None
        if cached is not None:
            runs_raw = cached.get("judge_runs", [])
            assert isinstance(runs_raw, list)
            record.judge_runs = [dict(jr) for jr in runs_raw]
            time_raw = cached.get("judge_time_ms", 0.0)
            record.judge_time_ms = float(time_raw) if isinstance(time_raw, (int, float)) else 0.0
            return

        outcome = await self._judge.judge(qa, record.generated)
        record.judge_runs = [dict(v) for v in outcome.verdicts]
        record.judge_time_ms = outcome.judge_time_ms

        jk = judge_key(
            self._benchmark_name,
            self._judge_model_spec,
            self._judge_temperature,
            self._judge_runs,
            outcome.prompt_fingerprint,
            qa.question,
            qa.gold,
            record.generated,
            question_type=qa.question_type,
            question_id=qa.question_id,
        )
        jpath = judge_cache_path(self._cache_root, jk)
        jpath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "judge_runs": record.judge_runs,
            "judge_time_ms": record.judge_time_ms,
            "prompt_fingerprint": outcome.prompt_fingerprint,
        }
        jpath.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        index_writer.touch(
            kind="judge",
            key=jk,
            path=str(jpath),
            meta={"benchmark": self._benchmark_name, "question_id": qa.question_id},
        )

    def _load_cached_judge(self, qa: QAItem, generated: str) -> dict[str, object] | None:
        """Look up a judge verdict by its byte-stable key.

        The lookup mirrors the write path: we compute the same key from
        the (qa, generated) triple + template fingerprint the judge
        adapter would use, then try to read ``judge/<key>.json``. Cache
        miss returns ``None`` — caller does the real judge call.
        """

        try:
            from ..judge.longmemeval import (
                LME_PROMPT_FINGERPRINTS,
                is_abstention_question,
            )
            from .judge_adapter import _template_key_for
        except ImportError:  # pragma: no cover
            return None

        if self._benchmark_name != "longmemeval":
            return None
        abstention = is_abstention_question(qa.question_id)
        template_key = _template_key_for(qa.question_type, abstention=abstention)
        fp = LME_PROMPT_FINGERPRINTS[template_key]
        jk = judge_key(
            self._benchmark_name,
            self._judge_model_spec,
            self._judge_temperature,
            self._judge_runs,
            fp,
            qa.question,
            qa.gold,
            generated,
            question_type=qa.question_type,
            question_id=qa.question_id,
        )
        jpath = judge_cache_path(self._cache_root, jk)
        if not jpath.is_file():
            return None
        try:
            payload = json.loads(jpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            _log.debug("judge cache read failed for %s: %s", jpath, e)
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _load_cached_answer(self, path: Path) -> QARecord | None:
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            _log.debug("answer cache read failed for %s: %s", path, e)
            return None
        from .manifest import _dict_to_record

        return _dict_to_record(data["record"])

    def _save_answer_cache(self, path: Path, record: QARecord) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"record": asdict(record)}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _record_key(case_id: str, qa_index: int, replicate_idx: int) -> str:
    base = f"{case_id}::{qa_index}"
    if replicate_idx == 0:
        return base
    return f"{base}::r{replicate_idx}"


def _retrieved_turn_ids(answer: AnswerResult) -> list[str]:
    """Union of ``source_turn_ids`` across retrieved units, preserving order.

    Informational only — stored on :class:`QARecord` for diagnostics. The
    scorer does NOT use these IDs for evidence attribution; it works from
    retrieved + evidence text (see :func:`_evidence_texts`).
    """

    seen: dict[str, None] = {}
    for unit in answer.retrieved:
        for tid in unit.source_turn_ids:
            if tid not in seen:
                seen[tid] = None
    return list(seen.keys())


def _evidence_texts(case: BenchmarkCase, qa: QAItem) -> list[str]:
    """Look up the verbatim text of each evidence turn in ``case``.

    The benchmark owns evidence attribution, so we materialize the evidence
    texts at record-creation time rather than asking the memory system to
    report which turns it retrieved. Text is looked up by ``turn_id`` across
    every session in ``case.sessions`` — LongMemEval's ``answer_session_ids``
    convention expands to every turn within those sessions, so missing IDs
    should be rare in practice.

    Unknown evidence turn IDs are silently skipped rather than erroring —
    dataset schema drift shouldn't abort the whole run. The resulting list
    can be shorter than ``qa.evidence_turn_ids``; the scorer tolerates that.
    """

    turn_text: dict[str, str] = {}
    for session in case.sessions:
        for turn in session.turns:
            turn_text[turn.turn_id] = turn.text
    return [turn_text[tid] for tid in qa.evidence_turn_ids if tid in turn_text]


__all__ = ["BenchmarkRunner"]
