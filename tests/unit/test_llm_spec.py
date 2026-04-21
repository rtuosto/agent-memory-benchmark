"""Tests for :func:`agent_memory_benchmark.llm.parse_spec` and friends."""

from __future__ import annotations

import pytest

from agent_memory_benchmark.llm import (
    InvalidSpecError,
    ProviderUnavailableError,
    build_provider,
    parse_spec,
)


class TestParseSpec:
    def test_openai_dated_model(self) -> None:
        assert parse_spec("openai:gpt-4o-mini-2024-07-18") == (
            "openai",
            "gpt-4o-mini-2024-07-18",
        )

    def test_ollama_model_containing_colon(self) -> None:
        """Ollama tags legitimately contain a colon; split on the first only."""
        assert parse_spec("ollama:llama3.1:8b") == ("ollama", "llama3.1:8b")

    def test_ollama_model_with_digest_pin(self) -> None:
        assert parse_spec("ollama:llama3.1:8b@sha256:abc") == (
            "ollama",
            "llama3.1:8b@sha256:abc",
        )

    def test_provider_is_lowercased(self) -> None:
        assert parse_spec("Ollama:llama3") == ("ollama", "llama3")
        assert parse_spec("OPENAI:gpt-4o") == ("openai", "gpt-4o")

    def test_whitespace_is_stripped(self) -> None:
        assert parse_spec("  ollama  :  llama3  ") == ("ollama", "llama3")

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "llama3",
            ":model",
            "provider:",
            "   :   ",
            "provider:  ",
        ],
    )
    def test_rejects_malformed(self, bad: str) -> None:
        with pytest.raises(InvalidSpecError):
            parse_spec(bad)

    def test_rejects_non_string(self) -> None:
        with pytest.raises(InvalidSpecError):
            parse_spec(None)  # type: ignore[arg-type]


class TestBuildProvider:
    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(InvalidSpecError, match="Unknown provider"):
            build_provider("gemini:gemini-pro")

    def test_openai_without_sdk_raises_provider_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Importing ``openai`` inside the provider must surface a clean error."""
        import builtins
        import sys

        real_import = builtins.__import__

        def _fake_import(
            name: str,
            globals: object = None,
            locals: object = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            # Only intercept the *absolute* import of the top-level ``openai``
            # package; leave the relative ``from .openai import ...`` alone.
            if name == "openai" and level == 0:
                raise ModuleNotFoundError("No module named 'openai'")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        # Purge any cached import so the next `from openai import ...` re-hits __import__.
        monkeypatch.delitem(sys.modules, "openai", raising=False)

        with pytest.raises(ProviderUnavailableError):
            build_provider("openai:gpt-4o-mini-2024-07-18", openai_api_key="dummy")

    def test_ollama_builds_without_env(self) -> None:
        """Ollama provider should construct purely from the spec (no key needed)."""
        provider = build_provider("ollama:llama3.1:8b", ollama_base_url="http://x:1")
        assert provider.spec == "ollama:llama3.1:8b"
        assert provider.model == "llama3.1:8b"

    def test_ollama_num_ctx_forwarded(self) -> None:
        """``ollama_num_ctx`` must reach the OllamaProvider constructor."""

        provider = build_provider(
            "ollama:llama3.1:8b",
            ollama_base_url="http://x:1",
            ollama_num_ctx=131072,
        )
        # Accessing ``_num_ctx`` is a test-only peek — the public API is the
        # value Ollama receives in ``options.num_ctx`` on ``chat(...)``.
        assert provider._num_ctx == 131072  # noqa: SLF001

    def test_ollama_num_ctx_none_uses_default(self) -> None:
        """Omitting ``ollama_num_ctx`` keeps the OllamaProvider default (8192)."""

        provider = build_provider("ollama:llama3.1:8b", ollama_base_url="http://x:1")
        assert provider._num_ctx == 8192  # noqa: SLF001
