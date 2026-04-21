"""Scorecard aggregation + rendering."""

from __future__ import annotations

from .render import print_scorecard_rich, render_scorecard_markdown
from .scorecard import (
    CategoryStats,
    Distribution,
    EvidenceStats,
    Scorecard,
    build_scorecard,
    normalize_answer,
    scorecard_to_dict,
    token_f1,
)

__all__ = [
    "CategoryStats",
    "Distribution",
    "EvidenceStats",
    "Scorecard",
    "build_scorecard",
    "normalize_answer",
    "print_scorecard_rich",
    "render_scorecard_markdown",
    "scorecard_to_dict",
    "token_f1",
]
