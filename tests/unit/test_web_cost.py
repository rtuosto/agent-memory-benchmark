"""Tests for the OpenAI cost estimator."""

from __future__ import annotations

from agent_memory_benchmark.web.cost import estimate_cost


def test_local_only_job_is_free() -> None:
    est = estimate_cost(
        dataset="longmemeval",
        answer_model_spec="ollama:llama3.1:8b",
        judge_model_spec="ollama:llama3.1:8b",
        n_questions=10,
    )
    assert est.has_paid_call is False
    assert est.total_low_usd == 0.0
    assert est.total_high_usd == 0.0
    assert est.answer.is_paid is False
    assert est.judge.is_paid is False


def test_openai_judge_flags_paid_call_and_prices() -> None:
    est = estimate_cost(
        dataset="longmemeval",
        answer_model_spec="ollama:llama3.1:8b",
        judge_model_spec="openai:gpt-4o-mini",
        n_questions=10,
    )
    assert est.has_paid_call is True
    assert est.judge.is_paid is True
    assert est.judge.known_price is True
    assert est.judge.price_per_1k_input == 0.00015
    assert est.total_low_usd is not None
    assert est.total_low_usd > 0
    # Judge is 10 q × 2000 input tok × 0.00015 + 10 × 100 output × 0.00060
    # = 0.003 + 0.0006 = 0.0036, with ±30% band → low ~0.0025.
    assert 0.001 <= est.total_low_usd <= 0.01


def test_openai_answer_also_flags_paid() -> None:
    est = estimate_cost(
        dataset="longmemeval",
        answer_model_spec="openai:gpt-4o",
        judge_model_spec="ollama:llama3.1:8b",
        n_questions=5,
    )
    assert est.has_paid_call is True
    assert est.answer.is_paid is True
    assert est.answer.model == "gpt-4o"
    assert est.judge.is_paid is False
    assert est.total_high_usd is not None
    assert est.total_high_usd > 0


def test_unknown_openai_model_records_warning() -> None:
    est = estimate_cost(
        dataset="longmemeval",
        answer_model_spec="openai:gpt-99-future",
        judge_model_spec="ollama:llama3.1:8b",
        n_questions=10,
    )
    assert est.has_paid_call is True
    assert est.answer.known_price is False
    assert est.answer.low_usd is None
    assert est.total_low_usd is None
    assert "gpt-99-future" in est.unknown_prices


def test_n_questions_overrides_dataset_default() -> None:
    small = estimate_cost(
        dataset="beam",
        answer_model_spec="openai:gpt-4o-mini",
        judge_model_spec="ollama:llama3.1:8b",
        n_questions=5,
    )
    full = estimate_cost(
        dataset="beam",
        answer_model_spec="openai:gpt-4o-mini",
        judge_model_spec="ollama:llama3.1:8b",
        n_questions=None,
    )
    # Full BEAM is 2000 questions vs 5 — cost scales linearly.
    assert full.total_high_usd is not None and small.total_high_usd is not None
    assert full.total_high_usd > small.total_high_usd * 100


def test_judge_runs_multiply_judge_calls() -> None:
    single = estimate_cost(
        dataset="longmemeval",
        answer_model_spec="ollama:llama3.1:8b",
        judge_model_spec="openai:gpt-4o-mini",
        n_questions=10,
        judge_runs=1,
    )
    ten = estimate_cost(
        dataset="longmemeval",
        answer_model_spec="ollama:llama3.1:8b",
        judge_model_spec="openai:gpt-4o-mini",
        n_questions=10,
        judge_runs=10,
    )
    assert ten.judge.n_calls == single.judge.n_calls * 10
    assert ten.total_high_usd is not None and single.total_high_usd is not None
    assert ten.total_high_usd > single.total_high_usd * 5


def test_beam_10m_variant_profile_is_much_larger() -> None:
    beam = estimate_cost(
        dataset="beam",
        answer_model_spec="openai:gpt-4o",
        judge_model_spec="ollama:llama3.1:8b",
        n_questions=1,
        variant="beam",
    )
    beam_10m = estimate_cost(
        dataset="beam",
        answer_model_spec="openai:gpt-4o",
        judge_model_spec="ollama:llama3.1:8b",
        n_questions=1,
        variant="beam-10m",
    )
    assert beam.answer.input_tokens_per_call == 200_000
    assert beam_10m.answer.input_tokens_per_call == 2_000_000
    assert beam_10m.total_high_usd is not None and beam.total_high_usd is not None
    assert beam_10m.total_high_usd > beam.total_high_usd * 5
