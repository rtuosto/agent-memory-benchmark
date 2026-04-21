"""Scorecard assembly — aggregate a list of ``QARecord`` into KPIs.

Four families, per plan:

1. **Quality** — overall/macro accuracy, per-category accuracy, token-F1.
2. **Wall-time performance** — `{mean, p50, p95, max}` distributions for
   ingestion, retrieval, generation, runner-measured answer total, and
   the ``answer_discrepancy`` drift signal.
3. **Retrieval footprint** — units and tokens retrieved per query.
4. **Retrieval quality vs evidence annotations** — six KPIs (turn/unit/
   token × completeness/density) computed by the benchmark itself via
   text attribution. The benchmark holds the evidence turns' text (from
   the dataset) and the retrieved units' text (from the adapter's
   ``AnswerResult.retrieved``) and matches them with SQuAD-normalized
   token overlap. Memory systems are NOT required to self-report which
   turn each retrieved unit came from — ``RetrievedUnit.source_turn_ids``
   is stored on ``QARecord`` as provenance if provided, but scoring does
   not depend on it.
5. **Throughput** — ``queries_per_sec`` / ``sessions_per_sec`` headline
   scalars that show up at the top of the markdown scorecard.

Token normalization for F1 and evidence-token metrics follows SQuAD:
lowercase, strip articles (``a``/``an``/``the``), strip punctuation, then
whitespace-split. Deterministic and dataset-neutral; no external deps.
Future ``--token-backend tiktoken:cl100k_base`` support lands in PR-12.
"""

from __future__ import annotations

import collections
import re
import string
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any

from ..runner.manifest import QARecord


@dataclass
class Distribution:
    """``{mean, p50, p95, max, n}`` wrapper used for every latency KPI."""

    mean: float
    p50: float
    p95: float
    max: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {"mean": self.mean, "p50": self.p50, "p95": self.p95, "max": self.max, "n": self.n}


@dataclass
class CategoryStats:
    accuracy: float | None
    count: int
    token_f1: float | None


@dataclass
class EvidenceStats:
    """Six evidence KPIs; ``None`` when no question had the underlying data."""

    turn_completeness: Distribution | None
    turn_density: Distribution | None
    unit_completeness: Distribution | None
    unit_density: Distribution | None
    token_completeness: Distribution | None
    token_density: Distribution | None
    n_questions_with_evidence: int
    n_questions_with_retrieval: int


@dataclass
class Scorecard:
    benchmark: str
    n_questions: int
    n_cases: int

    overall_accuracy: float | None
    macro_accuracy: float | None
    overall_token_f1: float | None
    per_category: dict[str, CategoryStats] = field(default_factory=dict)

    ingestion_per_session_ms: Distribution | None = None
    ingestion_total_ms: float = 0.0
    retrieval_per_query_ms: Distribution | None = None
    generation_per_query_ms: Distribution | None = None
    answer_total_per_query_ms: Distribution | None = None
    answer_discrepancy_ms: Distribution | None = None
    judge_per_question_ms: Distribution | None = None

    units_retrieved_per_query: Distribution | None = None
    tokens_retrieved_per_query: Distribution | None = None

    evidence: EvidenceStats | None = None

    throughput_queries_per_sec: float | None = None
    throughput_sessions_per_sec: float | None = None

    replicate_mean: float | None = None
    replicate_std: float | None = None
    judge_std_by_question: list[float] = field(default_factory=list)


def build_scorecard(records: list[QARecord], *, benchmark: str | None = None) -> Scorecard:
    """Aggregate ``records`` into a :class:`Scorecard`.

    ``benchmark`` overrides the auto-detected value (first record's
    ``benchmark`` field); passing it explicitly matters for empty runs
    where the default ``"?"`` sentinel is meaningless.
    """

    if not records:
        return Scorecard(
            benchmark=benchmark or "?",
            n_questions=0,
            n_cases=0,
            overall_accuracy=None,
            macro_accuracy=None,
            overall_token_f1=None,
        )

    resolved = benchmark or records[0].benchmark
    n_cases = len({r.case_id for r in records})

    overall_acc, per_cat_acc, judge_stds = _accuracy_breakdown(records)
    overall_f1, per_cat_f1 = _token_f1_breakdown(records)
    per_category = _merge_category_stats(per_cat_acc, per_cat_f1)
    macro_acc = _macro(per_category)

    ing_per_session = _distribution(
        [r.ingestion_time_ms for r in records if r.ingestion_time_ms > 0]
    )
    ingestion_total = sum(r.ingestion_time_ms for r in records)
    ret = _distribution([r.retrieval_time_ms for r in records])
    gen = _distribution([r.generation_time_ms for r in records])
    ans_total = _distribution([r.total_answer_time_ms for r in records])
    disc = _distribution([r.answer_discrepancy_ms for r in records])
    judge_time = _distribution([r.judge_time_ms for r in records if r.judge_time_ms > 0])

    units = _distribution([float(r.units_retrieved) for r in records if r.units_retrieved > 0])
    tokens = _distribution([float(r.tokens_retrieved) for r in records if r.tokens_retrieved > 0])

    evidence = _evidence_stats(records)
    throughput_q, throughput_s = _throughput(records)
    rep_mean, rep_std = _replicate_stats(records)

    return Scorecard(
        benchmark=resolved,
        n_questions=len(records),
        n_cases=n_cases,
        overall_accuracy=overall_acc,
        macro_accuracy=macro_acc,
        overall_token_f1=overall_f1,
        per_category=per_category,
        ingestion_per_session_ms=ing_per_session,
        ingestion_total_ms=ingestion_total,
        retrieval_per_query_ms=ret,
        generation_per_query_ms=gen,
        answer_total_per_query_ms=ans_total,
        answer_discrepancy_ms=disc,
        judge_per_question_ms=judge_time,
        units_retrieved_per_query=units,
        tokens_retrieved_per_query=tokens,
        evidence=evidence,
        throughput_queries_per_sec=throughput_q,
        throughput_sessions_per_sec=throughput_s,
        replicate_mean=rep_mean,
        replicate_std=rep_std,
        judge_std_by_question=judge_stds,
    )


def _accuracy_breakdown(
    records: list[QARecord],
) -> tuple[float | None, dict[str, tuple[float, int]], list[float]]:
    judged = [r for r in records if r.judge_runs]
    if not judged:
        return None, {}, []

    judge_stds: list[float] = []
    per_bucket: dict[str, list[float]] = {}
    all_question_means: list[float] = []

    for rec in judged:
        scores = [1.0 if bool(jr.get("correct")) else 0.0 for jr in rec.judge_runs]
        if len(scores) > 1:
            judge_stds.append(pstdev(scores))
        question_mean = mean(scores)
        all_question_means.append(question_mean)
        bucket = _category_key(rec)
        per_bucket.setdefault(bucket, []).append(question_mean)

    overall = mean(all_question_means) if all_question_means else None
    per_cat = {k: (mean(vals), len(vals)) for k, vals in sorted(per_bucket.items())}
    return overall, per_cat, judge_stds


def _token_f1_breakdown(
    records: list[QARecord],
) -> tuple[float | None, dict[str, tuple[float, int]]]:
    if not records:
        return None, {}
    scores = [token_f1(r.generated, r.gold) for r in records]
    per_bucket: dict[str, list[float]] = {}
    for rec, f1 in zip(records, scores, strict=True):
        per_bucket.setdefault(_category_key(rec), []).append(f1)
    overall = mean(scores) if scores else None
    per_cat = {k: (mean(vals), len(vals)) for k, vals in sorted(per_bucket.items())}
    return overall, per_cat


def _merge_category_stats(
    acc: dict[str, tuple[float, int]],
    f1: dict[str, tuple[float, int]],
) -> dict[str, CategoryStats]:
    out: dict[str, CategoryStats] = {}
    for key in sorted(set(acc) | set(f1)):
        acc_entry = acc.get(key)
        f1_entry = f1.get(key)
        out[key] = CategoryStats(
            accuracy=None if acc_entry is None else acc_entry[0],
            count=(acc_entry or f1_entry or (0.0, 0))[1],
            token_f1=None if f1_entry is None else f1_entry[0],
        )
    return out


def _macro(per_category: dict[str, CategoryStats]) -> float | None:
    vals = [s.accuracy for s in per_category.values() if s.accuracy is not None]
    return mean(vals) if vals else None


def _category_key(record: QARecord) -> str:
    if record.category is not None:
        return f"category_{record.category}"
    if record.question_type:
        return record.question_type
    return "uncategorized"


def _distribution(values: list[float]) -> Distribution | None:
    if not values:
        return None
    sorted_values = sorted(values)
    return Distribution(
        mean=mean(sorted_values),
        p50=_percentile(sorted_values, 0.50),
        p95=_percentile(sorted_values, 0.95),
        max=sorted_values[-1],
        n=len(sorted_values),
    )


def _percentile(sorted_values: list[float], p: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list."""

    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def _throughput(records: list[QARecord]) -> tuple[float | None, float | None]:
    total_answer_s = sum(r.total_answer_time_ms for r in records) / 1000.0
    queries = len(records) if total_answer_s > 0 else 0
    throughput_q = queries / total_answer_s if total_answer_s > 0 else None

    total_ingest_s = sum(r.ingestion_time_ms for r in records) / 1000.0
    # Each case's sessions were ingested once per case; records share
    # ingestion_time_ms by case. Count *unique* cases for session-rate.
    unique_cases = len({r.case_id for r in records})
    throughput_s = unique_cases / total_ingest_s if total_ingest_s > 0 else None
    return throughput_q, throughput_s


def _replicate_stats(records: list[QARecord]) -> tuple[float | None, float | None]:
    """Replicate mean/std when the run has more than one replicate_idx."""

    by_rep: dict[int, list[float]] = {}
    for rec in records:
        if not rec.judge_runs:
            continue
        scores = [1.0 if bool(jr.get("correct")) else 0.0 for jr in rec.judge_runs]
        by_rep.setdefault(rec.replicate_idx, []).append(mean(scores))
    if len(by_rep) < 2:
        return None, None
    rep_means = [mean(v) for v in by_rep.values()]
    return mean(rep_means), pstdev(rep_means)


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip articles + punctuation."""

    lowered = s.lower()
    depunct = lowered.translate(_PUNCT_TABLE)
    no_articles = _ARTICLES_RE.sub(" ", depunct)
    return _WHITESPACE_RE.sub(" ", no_articles).strip()


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token-level F1 between prediction and reference strings."""

    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = collections.Counter(pred_tokens) & collections.Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


EVIDENCE_COVERAGE_THRESHOLD = 0.5
"""Minimum token-overlap fraction for an evidence turn to count as "covered".

A retrieved unit "covers" an evidence turn when the multiset intersection of
their SQuAD-normalized tokens is at least this fraction of the evidence
turn's token count. 0.5 (half the evidence turn's tokens reappearing in a
retrieved unit) balances paraphrase-tolerance against false positives; the
constant is exposed so tests and tuning can reference it directly.
"""


def _evidence_stats(records: list[QARecord]) -> EvidenceStats:
    """Compute evidence KPIs via benchmark-owned text attribution.

    The benchmark attributes retrieval to evidence by matching normalized
    token multisets — memory systems only need to return the *text* they
    retrieved. ``RetrievedUnit.source_turn_ids`` (if populated) is stored
    on :class:`QARecord` as provenance but is *not* used here.

    Six KPIs, per plan:

    - ``turn_completeness`` = fraction of evidence turns covered (recall).
    - ``turn_density`` = fraction of retrieved units touching any evidence
      turn (same as ``unit_density`` under text attribution — no per-unit
      turn mapping exists without source IDs).
    - ``unit_completeness`` = same as ``turn_completeness`` under text
      attribution (both count covered evidence turns over total evidence).
    - ``unit_density`` = fraction of retrieved units touching evidence.
    - ``token_completeness`` = recalled evidence tokens / total evidence
      tokens (token-level recall).
    - ``token_density`` = recalled evidence tokens / total retrieved tokens
      (token-level precision).

    Turn- and unit-level numbers collapse onto each other under text
    attribution. If a future adapter provides reliable per-unit source
    turn mappings, the two could diverge; for now we report them as
    distinct keys so the JSON shape matches the plan while flagging the
    equivalence in the docstring.
    """

    turn_comps: list[float] = []
    turn_dens: list[float] = []
    unit_comps: list[float] = []
    unit_dens: list[float] = []
    tok_comps: list[float] = []
    tok_dens: list[float] = []

    n_with_evidence = 0
    n_with_retrieval = 0

    for rec in records:
        has_evidence = bool(rec.evidence_texts)
        has_retrieval = bool(rec.retrieved_texts)
        if has_evidence:
            n_with_evidence += 1
        if has_retrieval:
            n_with_retrieval += 1
        if not (has_evidence and has_retrieval):
            continue

        evidence_token_lists = [normalize_answer(t).split() for t in rec.evidence_texts]
        retrieved_token_lists = [normalize_answer(t).split() for t in rec.retrieved_texts]

        evidence_token_lists = [tl for tl in evidence_token_lists if tl]
        retrieved_token_lists = [tl for tl in retrieved_token_lists if tl]
        if not evidence_token_lists or not retrieved_token_lists:
            continue

        evidence_counters = [collections.Counter(tl) for tl in evidence_token_lists]
        retrieved_counters = [collections.Counter(tl) for tl in retrieved_token_lists]

        covered_evidence = 0
        for e_tokens, e_counter in zip(evidence_token_lists, evidence_counters, strict=True):
            if any(
                _coverage_fraction(r_counter, e_counter, len(e_tokens))
                >= EVIDENCE_COVERAGE_THRESHOLD
                for r_counter in retrieved_counters
            ):
                covered_evidence += 1

        units_touching = 0
        for r_counter in retrieved_counters:
            if any(
                _coverage_fraction(r_counter, e_counter, len(e_tokens))
                >= EVIDENCE_COVERAGE_THRESHOLD
                for e_tokens, e_counter in zip(evidence_token_lists, evidence_counters, strict=True)
            ):
                units_touching += 1

        completeness = covered_evidence / len(evidence_token_lists)
        density = units_touching / len(retrieved_token_lists)
        turn_comps.append(completeness)
        unit_comps.append(completeness)
        turn_dens.append(density)
        unit_dens.append(density)

        retrieved_union: collections.Counter[str] = collections.Counter()
        for r_counter in retrieved_counters:
            retrieved_union.update(r_counter)
        evidence_union: collections.Counter[str] = collections.Counter()
        for e_counter in evidence_counters:
            evidence_union.update(e_counter)
        recalled = sum((retrieved_union & evidence_union).values())
        total_evidence_tokens = sum(evidence_union.values())
        total_retrieved_tokens = sum(retrieved_union.values())
        if total_evidence_tokens:
            tok_comps.append(recalled / total_evidence_tokens)
        if total_retrieved_tokens:
            tok_dens.append(recalled / total_retrieved_tokens)

    return EvidenceStats(
        turn_completeness=_distribution(turn_comps),
        turn_density=_distribution(turn_dens),
        unit_completeness=_distribution(unit_comps),
        unit_density=_distribution(unit_dens),
        token_completeness=_distribution(tok_comps),
        token_density=_distribution(tok_dens),
        n_questions_with_evidence=n_with_evidence,
        n_questions_with_retrieval=n_with_retrieval,
    )


def _coverage_fraction(
    retrieved: collections.Counter[str],
    evidence: collections.Counter[str],
    evidence_total: int,
) -> float:
    """Fraction of evidence tokens present in a retrieved-unit multiset."""

    if evidence_total == 0:
        return 0.0
    overlap = sum((retrieved & evidence).values())
    return overlap / evidence_total


def scorecard_to_dict(sc: Scorecard) -> dict[str, Any]:
    """Serialize a :class:`Scorecard` to the public ``scorecard.json`` shape."""

    def _dist(d: Distribution | None) -> dict[str, float | int] | None:
        return d.as_dict() if d else None

    return {
        "benchmark": sc.benchmark,
        "n_questions": sc.n_questions,
        "n_cases": sc.n_cases,
        "quality": {
            "overall_accuracy": sc.overall_accuracy,
            "macro_accuracy": sc.macro_accuracy,
            "overall_token_f1": sc.overall_token_f1,
            "per_category": {
                k: {
                    "accuracy": v.accuracy,
                    "token_f1": v.token_f1,
                    "count": v.count,
                }
                for k, v in sc.per_category.items()
            },
            "replicate_mean": sc.replicate_mean,
            "replicate_std": sc.replicate_std,
            "judge_std_by_question": sc.judge_std_by_question,
        },
        "latency_ms": {
            "ingestion_per_session": _dist(sc.ingestion_per_session_ms),
            "ingestion_total": sc.ingestion_total_ms,
            "retrieval_per_query": _dist(sc.retrieval_per_query_ms),
            "generation_per_query": _dist(sc.generation_per_query_ms),
            "answer_total_per_query": _dist(sc.answer_total_per_query_ms),
            "answer_discrepancy": _dist(sc.answer_discrepancy_ms),
            "judge_per_question": _dist(sc.judge_per_question_ms),
        },
        "retrieval_footprint": {
            "units_per_query": _dist(sc.units_retrieved_per_query),
            "tokens_per_query": _dist(sc.tokens_retrieved_per_query),
        },
        "evidence": None
        if sc.evidence is None
        else {
            "turn_completeness": _dist(sc.evidence.turn_completeness),
            "turn_density": _dist(sc.evidence.turn_density),
            "unit_completeness": _dist(sc.evidence.unit_completeness),
            "unit_density": _dist(sc.evidence.unit_density),
            "token_completeness": _dist(sc.evidence.token_completeness),
            "token_density": _dist(sc.evidence.token_density),
            "n_questions_with_evidence": sc.evidence.n_questions_with_evidence,
            "n_questions_with_retrieval": sc.evidence.n_questions_with_retrieval,
        },
        "throughput": {
            "queries_per_sec": sc.throughput_queries_per_sec,
            "sessions_per_sec": sc.throughput_sessions_per_sec,
        },
    }


__all__ = [
    "CategoryStats",
    "Distribution",
    "EvidenceStats",
    "Scorecard",
    "build_scorecard",
    "normalize_answer",
    "scorecard_to_dict",
    "token_f1",
]
