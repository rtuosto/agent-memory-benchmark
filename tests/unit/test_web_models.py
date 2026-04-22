"""Tests for model + memory-adapter choice helpers."""

from __future__ import annotations

import subprocess

from agent_memory_benchmark.web import models


def test_parse_ollama_list_skips_header_and_blank_lines() -> None:
    sample = (
        "NAME                    ID              SIZE    MODIFIED\n"
        "llama3.1:8b             46e0c10c039e    4.9 GB  3 days ago\n"
        "\n"
        "qwen2.5:14b             1234567890ab    9.0 GB  1 week ago\n"
    )
    assert models._parse_ollama_list(sample) == ["llama3.1:8b", "qwen2.5:14b"]


def test_parse_ollama_list_empty_is_empty() -> None:
    assert models._parse_ollama_list("") == []
    assert models._parse_ollama_list("NAME  ID  SIZE  MODIFIED\n") == []


def test_fetch_ollama_models_returns_empty_when_cli_missing(monkeypatch) -> None:
    monkeypatch.setattr(models.shutil, "which", lambda name: None)
    assert models._fetch_ollama_models() == ()


def test_fetch_ollama_models_returns_empty_on_nonzero_exit(monkeypatch) -> None:
    monkeypatch.setattr(models.shutil, "which", lambda name: "/usr/bin/ollama")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=1, stdout="", stderr="")

    monkeypatch.setattr(models.subprocess, "run", fake_run)
    assert models._fetch_ollama_models() == ()


def test_fetch_ollama_models_prefixes_with_ollama(monkeypatch) -> None:
    monkeypatch.setattr(models.shutil, "which", lambda name: "/usr/bin/ollama")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=(
                "NAME                    ID              SIZE    MODIFIED\n"
                "llama3.1:8b             46e0c10c039e    4.9 GB  3 days ago\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(models.subprocess, "run", fake_run)
    assert models._fetch_ollama_models() == ("ollama:llama3.1:8b",)


def test_ollama_models_caches_between_calls(monkeypatch) -> None:
    calls = {"n": 0}

    def fake_fetch() -> tuple[str, ...]:
        calls["n"] += 1
        return ("ollama:fake:1b",)

    # Force a clean cache first.
    models._ollama_cache.fetched_at = 0.0
    models._ollama_cache.models = ()

    monkeypatch.setattr(models, "_fetch_ollama_models", fake_fetch)
    first = models.ollama_models()
    second = models.ollama_models()
    assert first == second == ("ollama:fake:1b",)
    assert calls["n"] == 1, "second call should have hit the cache"


def test_ollama_models_force_refresh(monkeypatch) -> None:
    models._ollama_cache.fetched_at = 0.0
    models._ollama_cache.models = ()

    counter = {"n": 0}

    def fake_fetch() -> tuple[str, ...]:
        counter["n"] += 1
        return (f"ollama:v{counter['n']}",)

    monkeypatch.setattr(models, "_fetch_ollama_models", fake_fetch)
    models.ollama_models()
    second = models.ollama_models(force_refresh=True)
    assert counter["n"] == 2
    assert second == ("ollama:v2",)


def test_available_models_combines_ollama_and_openai(monkeypatch) -> None:
    monkeypatch.setattr(models, "ollama_models", lambda: ("ollama:llama3.1:8b",))
    result = models.available_models()
    assert result[0] == "ollama:llama3.1:8b"
    assert "openai:gpt-4o" in result
    assert "openai:gpt-4o-mini" in result
    # Dedupe is a no-op here but should not reorder existing entries.
    monkeypatch.setattr(models, "ollama_models", lambda: ("openai:gpt-4o",))
    dedup = models.available_models()
    assert dedup.count("openai:gpt-4o") == 1


def test_memory_adapter_presets_returns_list() -> None:
    presets = models.memory_adapter_presets()
    assert "full-context" in presets
    assert any(p.startswith("python:") for p in presets)
