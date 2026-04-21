"""Byte-exact regression for cache key derivation.

These tests LOCK the hash algorithm. Any change to:

- the ``\\x1e`` separator,
- the ordering of parts,
- the string format of numeric fields (``f"{temp:.6f}"``, ``str(runs)``),
- the UTF-8 encoding discipline,

MUST bump the golden digests below AND produce a migration note. The
digests were computed against the reference algorithm the predecessor
benchmark uses so cross-repo scorecard comparability is preserved for
the inputs both codebases share.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_memory_benchmark.cache.keys import (
    SEPARATOR,
    answer_key,
    dataset_descriptor_hash,
    dataset_file_hash,
    hash_bytes,
    hash_parts,
    hash_text,
    ingestion_key,
    judge_key,
    prompt_fingerprint,
)


class TestSeparator:
    def test_is_ascii_record_separator(self) -> None:
        assert SEPARATOR == b"\x1e"
        assert len(SEPARATOR) == 1


class TestHashParts:
    def test_golden_two_parts(self) -> None:
        # sha256(b"a\x1eb\x1e").hexdigest()
        assert (
            hash_parts(("a", "b"))
            == "619c1d904388d8e52dc7ebd549c96d3cdff34266b44917a14cdf06919cea1648"
        )

    def test_empty_tuple_is_empty_sha256(self) -> None:
        assert hash_parts(()) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_trailing_separator_framing_matters(self) -> None:
        """(``"a"``, ``""``) and (``"a"``,) must produce different digests.

        The trailing separator after each part is what distinguishes them,
        which is the whole point of the framing.
        """

        assert hash_parts(("a", "")) != hash_parts(("a",))

    def test_utf8_multibyte_is_stable(self) -> None:
        # Russian + Japanese + emoji — exercises multi-byte UTF-8.
        assert hash_parts(("привет",)) == hash_parts(("привет",))
        assert hash_parts(("こんにちは",)) != hash_parts(("hello",))


class TestHashBytesAndText:
    def test_hash_bytes_matches_stdlib(self) -> None:
        assert hash_bytes(b"") == hash_parts(())
        assert (
            hash_text("hello\n")
            == "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
        )

    def test_prompt_fingerprint_is_hash_of_utf8_bytes(self) -> None:
        assert (
            prompt_fingerprint("")
            == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )


class TestDatasetHashes:
    def test_file_hash(self, tmp_path: Path) -> None:
        p = tmp_path / "data.json"
        p.write_bytes(b"payload")
        assert dataset_file_hash(p) == hash_bytes(b"payload")

    def test_descriptor_hash_is_hash_parts(self) -> None:
        parts = ("longmemeval", "s", "deadbeef")
        assert dataset_descriptor_hash(parts) == hash_parts(parts)


class TestIngestionKey:
    def test_golden(self) -> None:
        assert (
            ingestion_key("engram", "v1", "abc123", "case_01")
            == "1a8e345bcf0dabef7c11229142b5fe5dae7c1c03df1be9e18b650b79ca7a1654"
        )


class TestAnswerKey:
    def test_default_replicate_omits_suffix_for_predecessor_compat(self) -> None:
        """replicate_idx=0 must produce the predecessor's 6-part digest."""

        assert (
            answer_key(
                "engram",
                "v1",
                "abc123",
                "openai:gpt-4o-mini-2024-07-18",
                "q_001",
                "What did I buy?",
            )
            == "21b28fb4d668e07117bca55fdec7144d4899ff06ee12cb8018f0b88605dbfc0f"
        )

    def test_replicate_greater_than_zero_appends_suffix(self) -> None:
        assert (
            answer_key(
                "engram",
                "v1",
                "abc123",
                "openai:gpt-4o-mini-2024-07-18",
                "q_001",
                "What did I buy?",
                replicate_idx=2,
            )
            == "776cceeb206cc06ac8d2adfd8d23c2e4fe6767db1741586a2348b05a0637f795"
        )

    def test_replicate_distinguishes_runs(self) -> None:
        base = answer_key("engram", "v1", "h", "spec", "qk", "q", replicate_idx=0)
        one = answer_key("engram", "v1", "h", "spec", "qk", "q", replicate_idx=1)
        two = answer_key("engram", "v1", "h", "spec", "qk", "q", replicate_idx=2)
        assert base != one != two
        assert base != two

    def test_negative_replicate_rejected(self) -> None:
        with pytest.raises(ValueError):
            answer_key("engram", "v1", "h", "spec", "qk", "q", replicate_idx=-1)


class TestJudgeKey:
    def test_golden_with_type_and_id(self) -> None:
        assert (
            judge_key(
                "longmemeval",
                "openai:gpt-4o-2024-08-06",
                0.0,
                10,
                "fp_deadbeef",
                "Q?",
                "gold",
                "gen",
                question_type="single-session-user",
                question_id="q_001",
            )
            == "0d0d0a903391d129888a52c16400f430c3ff9d493def6de5aa51639950dda100"
        )

    def test_golden_without_type_and_id_preserves_pipe_framing(self) -> None:
        assert (
            judge_key(
                "locomo",
                "openai:gpt-4o-mini-2024-07-18",
                0.5,
                1,
                "fp_cafe",
                "Q?",
                "gold",
                "gen",
            )
            == "919b6c2300c1fb8f6d951aa6f34d214d88ea97284d32ccd0c1f56c74b6b2bd69"
        )

    def test_temperature_format_is_six_decimal_places(self) -> None:
        """0 and 0.0 and 0.000000 must produce the same digest."""

        k0 = judge_key("b", "m", 0, 1, "fp", "q", "g", "r")
        k_float = judge_key("b", "m", 0.0, 1, "fp", "q", "g", "r")
        k_six = judge_key("b", "m", 0.000000, 1, "fp", "q", "g", "r")
        assert k0 == k_float == k_six

    def test_temperature_variants_differ(self) -> None:
        k_low = judge_key("b", "m", 0.0, 1, "fp", "q", "g", "r")
        k_mid = judge_key("b", "m", 0.5, 1, "fp", "q", "g", "r")
        assert k_low != k_mid


class TestPaths:
    def test_ingestion_state_path_layout(self, tmp_path: Path) -> None:
        from agent_memory_benchmark.cache.keys import ingestion_state_path

        p = ingestion_state_path(tmp_path, "engram", "abcd")
        assert p == tmp_path / "ingestion" / "engram" / "abcd" / "state.json"

    def test_ingestion_state_path_sanitizes_slashes(self, tmp_path: Path) -> None:
        from agent_memory_benchmark.cache.keys import ingestion_state_path

        p = ingestion_state_path(tmp_path, "vendor/product", "k")
        assert p == tmp_path / "ingestion" / "vendor_product" / "k" / "state.json"

    def test_answer_and_judge_paths(self, tmp_path: Path) -> None:
        from agent_memory_benchmark.cache.keys import answer_cache_path, judge_cache_path

        assert answer_cache_path(tmp_path, "abc") == tmp_path / "answers" / "abc.json"
        assert judge_cache_path(tmp_path, "abc") == tmp_path / "judge" / "abc.json"
