"""Manifest + QARecord + RunDir serialization round-trips.

These guard the on-disk ``answers.json`` schema. Any additions to QARecord
or RunManifest fields should come with a test added here (or at minimum a
``_dict_to_record`` drop-unknown test that proves forward-compat).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from agent_memory_benchmark.runner.manifest import (
    QARecord,
    RunDir,
    RunManifest,
    build_run_directory_name,
    load_run_file,
    sanitize_path_component,
    save_meta_json,
    save_run_file,
)


def _manifest(benchmark: str = "longmemeval") -> RunManifest:
    return RunManifest(
        benchmark=benchmark,
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
        judge_prompt_fingerprint="fp" * 32,
        dataset_name=benchmark,
        dataset_split="s",
        dataset_path=None,
        dataset_descriptor_hash="ds" * 32,
        hf_revision_sha="0" * 40,
        replicate_idx=0,
        replicate_seed=None,
        benchmark_git_sha="0123456789abcdef0123456789abcdef01234567",
        benchmark_git_branch="feat/runner",
        benchmark_git_dirty=False,
        benchmark_version="0.1.0",
        protocol_version="0.1",
        tag=None,
        cli_argv=["amb", "run", "longmemeval"],
        timestamp_utc="2026-04-20T00:00:00Z",
    )


def _record(key: str, gold: str = "hello", generated: str = "hello world") -> QARecord:
    return QARecord(
        key=key,
        benchmark="longmemeval",
        case_id="case1",
        question="What did I say?",
        gold=gold,
        generated=generated,
        question_id=key.split("::")[0],
        question_type="single-session-user",
        qa_index=0,
        judge_runs=[{"correct": True, "raw": "yes"}],
    )


def test_save_and_load_answers_roundtrip(tmp_path: Path) -> None:
    manifest = _manifest()
    records = [_record("q1::0"), _record("q2::0", gold="2", generated="2")]
    path = tmp_path / "answers.json"

    save_run_file(path, manifest, records)
    loaded_manifest, record_map = load_run_file(path)

    assert loaded_manifest == manifest
    assert set(record_map) == {"q1::0", "q2::0"}
    assert record_map["q1::0"].gold == "hello"
    assert record_map["q2::0"].generated == "2"


def test_load_run_file_drops_unknown_fields(tmp_path: Path) -> None:
    """Forward-compat: extra keys on disk must not break loads."""

    path = tmp_path / "answers.json"
    payload = {
        "meta": {
            **asdict(_manifest()),
            "some_future_field": 42,
        },
        "records": [
            {**asdict(_record("q1::0")), "future_metric_ms": 1.23},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    manifest, record_map = load_run_file(path)
    assert manifest.benchmark == "longmemeval"
    assert record_map["q1::0"].gold == "hello"


def test_save_meta_json_writes_manifest_as_json(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / "meta.json"
    save_meta_json(path, manifest)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["benchmark"] == "longmemeval"
    assert data["judge_prompt_fingerprint"] == "fp" * 32


def test_build_run_directory_name_sanitizes_and_orders() -> None:
    from datetime import datetime

    ts = datetime(2026, 4, 20, 10, 30, 15)
    name = build_run_directory_name(
        benchmark="longmemeval",
        memory_system_id="my/system id",
        answer_model_spec="ollama:llama3.1:8b",
        timestamp=ts,
        tag="exp-1",
    )
    assert name.startswith("2026-04-20_103015_longmemeval_my-system-id_ollama-llama3.1-8b")
    assert name.endswith("_exp-1")


def test_sanitize_path_component_collapses_unsafe_chars() -> None:
    assert sanitize_path_component("hello/world:tag") == "hello-world-tag"
    assert sanitize_path_component("...") == "unnamed"


def test_run_dir_properties_compose_paths(tmp_path: Path) -> None:
    rd = RunDir(tmp_path / "run_abc")
    assert rd.answers_path == tmp_path / "run_abc" / "answers.json"
    assert rd.meta_path == tmp_path / "run_abc" / "meta.json"
    assert rd.scorecard_json == tmp_path / "run_abc" / "scorecard.json"
    assert rd.scorecard_md == tmp_path / "run_abc" / "scorecard.md"


def test_save_run_file_creates_parent_dir(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "answers.json"
    save_run_file(nested, _manifest(), [_record("q::0")])
    assert nested.is_file()


@pytest.mark.parametrize(
    "value",
    [
        "",
        "///",
        ":::",
    ],
)
def test_sanitize_path_component_never_empty(value: str) -> None:
    assert sanitize_path_component(value) == "unnamed"
