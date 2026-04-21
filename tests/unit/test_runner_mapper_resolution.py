"""Tests for ``runner._resolve_callable`` — CLI spec → Python callable.

The resolver is the shared implementation behind ``--session-mapper`` and
``--result-mapper``. Same grammar as ``--memory python:pkg.mod:Class``,
same error shape.
"""

from __future__ import annotations

import pytest

from agent_memory_benchmark.runner import _resolve_callable


def _sample_mapper(value: object) -> object:
    """Declared at module scope so importlib can resolve it from a spec."""

    return value


def test_resolve_callable_returns_none_for_none() -> None:
    assert _resolve_callable(None, flag="--session-mapper") is None


def test_resolve_callable_resolves_module_function() -> None:
    fn = _resolve_callable(
        "tests.unit.test_runner_mapper_resolution:_sample_mapper",
        flag="--session-mapper",
    )
    assert fn is _sample_mapper
    assert fn("hello") == "hello"


def test_resolve_callable_rejects_missing_colon() -> None:
    with pytest.raises(ValueError, match=r"--session-mapper must be 'pkg.module:function'"):
        _resolve_callable("bad_spec", flag="--session-mapper")


def test_resolve_callable_rejects_empty_module() -> None:
    with pytest.raises(ValueError, match="both module path and callable name"):
        _resolve_callable(":func", flag="--result-mapper")


def test_resolve_callable_rejects_empty_function_name() -> None:
    with pytest.raises(ValueError, match="both module path and callable name"):
        _resolve_callable("pkg.mod:", flag="--result-mapper")


def test_resolve_callable_reports_import_failure() -> None:
    with pytest.raises(ValueError, match="could not import 'nonexistent.pkg'"):
        _resolve_callable("nonexistent.pkg:fn", flag="--session-mapper")


def test_resolve_callable_reports_missing_attribute() -> None:
    with pytest.raises(ValueError, match="has no attribute 'missing_fn'"):
        _resolve_callable(
            "tests.unit.test_runner_mapper_resolution:missing_fn",
            flag="--result-mapper",
        )


def test_resolve_callable_rejects_noncallable_attribute() -> None:
    # Our own test module exports a constant; resolve to something that exists
    # but isn't callable by targeting the `pytest` module's __name__ attr.
    with pytest.raises(ValueError, match="not callable"):
        _resolve_callable("pytest:__name__", flag="--session-mapper")


def test_resolve_callable_flag_name_appears_in_every_error() -> None:
    """Users should see which flag's value was invalid."""

    cases = [
        ("bad", "must be"),
        (":f", "both module"),
        ("pkg:", "both module"),
        ("nonexistent:x", "could not import"),
    ]
    for spec, _ in cases:
        with pytest.raises(ValueError, match="--my-flag"):
            _resolve_callable(spec, flag="--my-flag")
