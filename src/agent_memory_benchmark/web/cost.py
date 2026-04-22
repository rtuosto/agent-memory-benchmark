"""Cost estimator for paid-API (OpenAI) answer + judge models.

The estimator is deliberately coarse. Its job is to prevent the user
from clicking *Run* on a job that's about to burn $400 without
realising it — not to predict the final invoice to the penny. Every
estimate is returned as a ``(low_usd, high_usd)`` range with the
per-call token assumptions spelled out so the user can sanity-check
the inputs.

Price constants below are USD per 1K tokens. They are pinned at
2026-01 pricing and change over time; when the next price revision
lands, bump the table and the snapshot date in :data:`PRICE_TABLE_DATE`.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICE_TABLE_DATE = "2026-01"

_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.010),
    "gpt-4o-2024-11-20": (0.0025, 0.010),
    "gpt-4o-2024-08-06": (0.0025, 0.010),
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4o-mini-2024-07-18": (0.00015, 0.00060),
    "gpt-4-turbo": (0.010, 0.030),
    "gpt-4-turbo-2024-04-09": (0.010, 0.030),
    "gpt-4": (0.030, 0.060),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "o1": (0.015, 0.060),
    "o1-mini": (0.003, 0.012),
    "o3-mini": (0.0011, 0.0044),
}


@dataclass(frozen=True)
class DatasetProfile:
    """Rough token budgets per question for a given dataset.

    The numbers are intentionally on the high side — the goal is to
    warn the user about worst-case cost, not the mean.
    """

    name: str
    default_n_questions: int
    answer_input_tokens_per_q: int
    answer_output_tokens_per_q: int
    judge_input_tokens_per_q: int
    judge_output_tokens_per_q: int


_PROFILES: dict[str, DatasetProfile] = {
    "longmemeval": DatasetProfile(
        name="longmemeval",
        default_n_questions=100,
        answer_input_tokens_per_q=50_000,
        answer_output_tokens_per_q=200,
        judge_input_tokens_per_q=2_000,
        judge_output_tokens_per_q=100,
    ),
    "locomo": DatasetProfile(
        name="locomo",
        default_n_questions=200,
        answer_input_tokens_per_q=30_000,
        answer_output_tokens_per_q=200,
        judge_input_tokens_per_q=2_000,
        judge_output_tokens_per_q=100,
    ),
    "beam": DatasetProfile(
        name="beam",
        default_n_questions=2_000,
        answer_input_tokens_per_q=200_000,
        answer_output_tokens_per_q=300,
        judge_input_tokens_per_q=2_000,
        judge_output_tokens_per_q=100,
    ),
    "beam-10m": DatasetProfile(
        name="beam-10m",
        default_n_questions=2_000,
        answer_input_tokens_per_q=2_000_000,
        answer_output_tokens_per_q=300,
        judge_input_tokens_per_q=2_000,
        judge_output_tokens_per_q=100,
    ),
}


@dataclass(frozen=True)
class RoleEstimate:
    """Cost estimate for a single provider role (answer or judge)."""

    role: str  # "answer" | "judge"
    model_spec: str
    is_paid: bool
    model: str | None  # OpenAI model name if paid
    known_price: bool
    n_calls: int
    input_tokens_per_call: int
    output_tokens_per_call: int
    price_per_1k_input: float | None
    price_per_1k_output: float | None
    low_usd: float | None
    high_usd: float | None
    note: str | None = None


@dataclass(frozen=True)
class CostEstimate:
    """Combined estimate for one benchmark job."""

    dataset: str
    n_questions: int
    judge_runs: int
    answer: RoleEstimate
    judge: RoleEstimate
    total_low_usd: float | None
    total_high_usd: float | None
    has_paid_call: bool
    unknown_prices: list[str]


def _parse_openai_spec(spec: str) -> str | None:
    """Return the OpenAI model name for ``openai:<model>``; else ``None``."""

    if not spec:
        return None
    lower = spec.strip().lower()
    if not lower.startswith("openai:"):
        return None
    return spec.strip().split(":", 1)[1]


def _lookup_price(model: str) -> tuple[float, float] | None:
    if model in _PRICES:
        return _PRICES[model]
    for key, price in _PRICES.items():
        if model.startswith(key):
            return price
    return None


def _role_estimate(
    *,
    role: str,
    model_spec: str,
    n_calls: int,
    input_tokens_per_call: int,
    output_tokens_per_call: int,
) -> RoleEstimate:
    """Build a :class:`RoleEstimate` for a single provider call path.

    The ``low_usd`` / ``high_usd`` band is ``+/- 30%`` around the point
    estimate derived from the dataset profile — wide enough that a
    known-chatty memory system or a long BEAM conversation still lands
    inside the user's mental budget.
    """

    model = _parse_openai_spec(model_spec)
    if model is None:
        return RoleEstimate(
            role=role,
            model_spec=model_spec,
            is_paid=False,
            model=None,
            known_price=False,
            n_calls=n_calls,
            input_tokens_per_call=input_tokens_per_call,
            output_tokens_per_call=output_tokens_per_call,
            price_per_1k_input=None,
            price_per_1k_output=None,
            low_usd=0.0,
            high_usd=0.0,
            note="local provider — free",
        )

    prices = _lookup_price(model)
    if prices is None:
        return RoleEstimate(
            role=role,
            model_spec=model_spec,
            is_paid=True,
            model=model,
            known_price=False,
            n_calls=n_calls,
            input_tokens_per_call=input_tokens_per_call,
            output_tokens_per_call=output_tokens_per_call,
            price_per_1k_input=None,
            price_per_1k_output=None,
            low_usd=None,
            high_usd=None,
            note=f"no price entry for model {model!r} — cannot estimate",
        )

    price_in, price_out = prices
    point = (
        n_calls * input_tokens_per_call / 1000.0 * price_in
        + n_calls * output_tokens_per_call / 1000.0 * price_out
    )
    return RoleEstimate(
        role=role,
        model_spec=model_spec,
        is_paid=True,
        model=model,
        known_price=True,
        n_calls=n_calls,
        input_tokens_per_call=input_tokens_per_call,
        output_tokens_per_call=output_tokens_per_call,
        price_per_1k_input=price_in,
        price_per_1k_output=price_out,
        low_usd=round(point * 0.7, 4),
        high_usd=round(point * 1.3, 4),
    )


def estimate_cost(
    *,
    dataset: str,
    answer_model_spec: str,
    judge_model_spec: str,
    n_questions: int | None = None,
    judge_runs: int = 1,
    variant: str = "beam",
) -> CostEstimate:
    """Estimate the OpenAI cost of a benchmark run.

    Both answer and judge are costed so the user sees the full bill
    before confirming. Local (Ollama / full-context) roles contribute
    ``$0`` to the total but still appear in the breakdown.

    ``n_questions`` overrides the dataset default — respect a user's
    ``--limit`` so a 10-question smoke test doesn't display the
    full-dataset headline cost.
    """

    profile_key = "beam-10m" if dataset == "beam" and variant == "beam-10m" else dataset
    profile = _PROFILES.get(profile_key)
    if profile is None:
        profile = _PROFILES["longmemeval"]

    n = n_questions if n_questions is not None else profile.default_n_questions
    n = max(1, n)

    answer = _role_estimate(
        role="answer",
        model_spec=answer_model_spec,
        n_calls=n,
        input_tokens_per_call=profile.answer_input_tokens_per_q,
        output_tokens_per_call=profile.answer_output_tokens_per_q,
    )
    judge = _role_estimate(
        role="judge",
        model_spec=judge_model_spec,
        n_calls=n * max(1, judge_runs),
        input_tokens_per_call=profile.judge_input_tokens_per_q,
        output_tokens_per_call=profile.judge_output_tokens_per_q,
    )

    unknown_prices: list[str] = []
    for role in (answer, judge):
        if role.is_paid and not role.known_price and role.model is not None:
            unknown_prices.append(role.model)

    has_paid = answer.is_paid or judge.is_paid

    if answer.low_usd is None or judge.low_usd is None:
        total_low: float | None = None
        total_high: float | None = None
    else:
        total_low = round(answer.low_usd + judge.low_usd, 4)
        total_high = round((answer.high_usd or 0.0) + (judge.high_usd or 0.0), 4)

    return CostEstimate(
        dataset=dataset,
        n_questions=n,
        judge_runs=max(1, judge_runs),
        answer=answer,
        judge=judge,
        total_low_usd=total_low,
        total_high_usd=total_high,
        has_paid_call=has_paid,
        unknown_prices=unknown_prices,
    )


__all__ = [
    "PRICE_TABLE_DATE",
    "CostEstimate",
    "DatasetProfile",
    "RoleEstimate",
    "estimate_cost",
]
