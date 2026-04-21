"""Port a predecessor (``agent-memory``) ``scorecard.json`` into this repo's schema.

The predecessor repo emitted a flat scorecard:

    overall_accuracy, macro_accuracy, overall_f1
    per_category.<cat>.{mean, n}
    per_category_f1.<cat>.{mean, n}
    latency_ms.{retrieval_ms, generation_ms, total_answer_ms, ingestion_ms,
                units_retrieved, tokens_retrieved}.{mean, p50, p95, max}
    evidence_{session_density, completeness, ordering}_mean
    needle_density_{mean, n}
    per_category_evidence.<cat>.{session_density, completeness, ordering}

This repo's schema is nested under ``quality.*``, ``latency_ms.*`` (per-query /
per-session buckets), ``retrieval_footprint.*``, ``evidence.*`` (turn/unit/token
× completeness/density), and ``throughput.*``. The two are close enough that
accuracy, F1, latency, and retrieval footprint map cleanly. Evidence and
throughput do not: the predecessor's session-level metrics and this repo's
turn/unit/token decomposition measure different things, so they are deliberately
dropped rather than mis-mapped.

Usage:
    python scripts/port_predecessor_scorecard.py <dir-or-scorecard.json> [...]

For each input path the original file is preserved as ``scorecard.legacy.json``
and a new ``scorecard.json`` in the current schema is written next to it. Safe
to re-run; the legacy file is only written on first port.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

LEGACY_NAME = "scorecard.legacy.json"
CURRENT_NAME = "scorecard.json"

_LATENCY_BUCKET_MAP = {
    "retrieval_ms": "retrieval_per_query",
    "generation_ms": "generation_per_query",
    "total_answer_ms": "answer_total_per_query",
    "ingestion_ms": "ingestion_per_session",
}

_FOOTPRINT_BUCKET_MAP = {
    "units_retrieved": "units_per_query",
    "tokens_retrieved": "tokens_per_query",
}

_STAT_KEYS = ("mean", "p50", "p95", "max")


def port_scorecard(legacy: dict[str, Any]) -> dict[str, Any]:
    n_questions = legacy.get("n_questions")
    out: dict[str, Any] = {
        "benchmark": legacy.get("benchmark"),
        "n_questions": n_questions,
        "n_cases": legacy.get("n_conversations"),
        "quality": _port_quality(legacy),
        "latency_ms": _port_latency(legacy, n_questions=n_questions),
        "retrieval_footprint": _port_footprint(legacy, n_questions=n_questions),
        "_schema_ported_from": "agent-memory (predecessor repo)",
    }
    return out


def _port_quality(legacy: dict[str, Any]) -> dict[str, Any]:
    per_cat_acc = legacy.get("per_category") or {}
    per_cat_f1 = legacy.get("per_category_f1") or {}
    per_category: dict[str, dict[str, Any]] = {}
    for cat in sorted(set(per_cat_acc) | set(per_cat_f1)):
        acc_row = per_cat_acc.get(cat) or {}
        f1_row = per_cat_f1.get(cat) or {}
        per_category[cat] = {
            "accuracy": acc_row.get("mean"),
            "token_f1": f1_row.get("mean"),
            "count": acc_row.get("n") or f1_row.get("n"),
        }
    return {
        "overall_accuracy": legacy.get("overall_accuracy"),
        "macro_accuracy": legacy.get("macro_accuracy"),
        "overall_token_f1": legacy.get("overall_f1"),
        "per_category": per_category,
        "replicate_mean": None,
        "replicate_std": None,
        "judge_std_by_question": legacy.get("judge_std_by_question") or [],
    }


def _port_latency(legacy: dict[str, Any], *, n_questions: int | None) -> dict[str, Any]:
    legacy_lat = legacy.get("latency_ms") or {}
    out: dict[str, Any] = {}
    for src_key, dst_key in _LATENCY_BUCKET_MAP.items():
        bucket = legacy_lat.get(src_key)
        if not isinstance(bucket, dict):
            continue
        out[dst_key] = {k: bucket.get(k) for k in _STAT_KEYS if k in bucket}
        if n_questions is not None:
            out[dst_key]["n"] = n_questions
    return out


def _port_footprint(legacy: dict[str, Any], *, n_questions: int | None) -> dict[str, Any]:
    legacy_lat = legacy.get("latency_ms") or {}
    out: dict[str, Any] = {}
    for src_key, dst_key in _FOOTPRINT_BUCKET_MAP.items():
        bucket = legacy_lat.get(src_key)
        if not isinstance(bucket, dict):
            continue
        out[dst_key] = {k: bucket.get(k) for k in _STAT_KEYS if k in bucket}
        if n_questions is not None:
            out[dst_key]["n"] = n_questions
    return out


def _resolve_input(path: Path) -> Path:
    if path.is_dir():
        return path / CURRENT_NAME
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Run directory or scorecard.json file(s) to port in-place.",
    )
    args = parser.parse_args(argv)

    rc = 0
    for raw in args.paths:
        target = _resolve_input(raw)
        if not target.is_file():
            print(f"skip: {raw} (no scorecard.json)", file=sys.stderr)
            rc = 1
            continue

        legacy_path = target.with_name(LEGACY_NAME)
        legacy = json.loads(target.read_text(encoding="utf-8"))
        if "quality" in legacy and "overall_accuracy" not in legacy:
            print(f"skip: {target} already in current schema")
            continue
        if not legacy_path.exists():
            legacy_path.write_text(
                json.dumps(legacy, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        ported = port_scorecard(legacy)
        target.write_text(
            json.dumps(ported, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"ported: {target}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
