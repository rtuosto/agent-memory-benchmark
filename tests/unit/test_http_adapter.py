"""Tests for :class:`HttpAdapter`.

Uses ``respx`` to mock the REST contract defined by ``openapi.yaml`` /
``docs/http-api.md`` so the tests are hermetic — no running service
required.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from agent_memory_benchmark.adapters.http_adapter import HttpAdapter, HttpAdapterError
from agent_memory_benchmark.types import Session, Turn

BASE = "http://mem.test:8000"


def _identity_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "memory_system_id": "example-mem",
        "memory_version": "1.2.3",
        "protocol_version": "0.1",
        "supports_persistence": False,
    }
    payload.update(overrides)
    return payload


def _session() -> Session:
    return Session(
        session_index=1,
        turns=(
            Turn(turn_id="t1", speaker="user", text="hi"),
            Turn(turn_id="t2", speaker="assistant", text="hello", timestamp="2024-01-01"),
        ),
        session_time="2024-01-01",
        session_id="s1",
    )


async def test_construction_does_not_hit_the_wire() -> None:
    """Constructor MUST NOT perform network I/O — assembly is sync-safe."""

    with respx.mock(assert_all_called=False) as router:
        identity = router.get(f"{BASE}/v1/identity").respond(json=_identity_payload())
        HttpAdapter(BASE)
        assert not identity.called


async def test_open_populates_identity_and_persistence_flag() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{BASE}/v1/identity").respond(
            json=_identity_payload(memory_version="9.9.9", supports_persistence=True),
        )
        adapter = await HttpAdapter.connect(BASE)
    assert adapter.memory_system_id == "example-mem"
    assert adapter.memory_version == "9.9.9"
    assert adapter.supports_persistence is True


async def test_open_rejects_missing_required_fields() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{BASE}/v1/identity").respond(json={"memory_system_id": "x"})
        with pytest.raises(HttpAdapterError, match="missing required field"):
            await HttpAdapter.connect(BASE)


async def test_open_rejects_non_2xx_identity() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{BASE}/v1/identity").respond(status_code=500, text="boom")
        with pytest.raises(HttpAdapterError, match="HTTP 500"):
            await HttpAdapter.connect(BASE)


async def test_operations_before_open_error_out() -> None:
    adapter = HttpAdapter(BASE)
    with pytest.raises(HttpAdapterError, match="must be opened"):
        await adapter.ingest_session(_session(), "case_1")
    with pytest.raises(HttpAdapterError, match="must be opened"):
        await adapter.answer_question("q", "case_1")
    with pytest.raises(HttpAdapterError, match="must be opened"):
        await adapter.reset()


async def test_ingest_sends_session_json() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{BASE}/v1/identity").respond(json=_identity_payload())
        ingest = router.post(f"{BASE}/v1/ingest").respond(json={"ok": True})
        adapter = await HttpAdapter.connect(BASE)
        await adapter.ingest_session(_session(), "case_1")

    body = json.loads(ingest.calls.last.request.content)
    assert body["case_id"] == "case_1"
    session = body["session"]
    assert session["session_index"] == 1
    assert session["session_id"] == "s1"
    assert session["session_time"] == "2024-01-01"
    assert [t["turn_id"] for t in session["turns"]] == ["t1", "t2"]
    # Nullable fields must be present as JSON null, not omitted — the REST
    # contract advertises them explicitly.
    assert session["turns"][0]["timestamp"] is None
    assert session["turns"][0]["image_caption"] is None


async def test_ingest_trailing_slash_in_base_url_is_stripped() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{BASE}/v1/identity").respond(json=_identity_payload())
        ingest = router.post(f"{BASE}/v1/ingest").respond(status_code=204)
        adapter = await HttpAdapter.connect(f"{BASE}///")
        await adapter.ingest_session(_session(), "case_1")
    assert ingest.called


async def test_ingest_204_no_content_is_accepted() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{BASE}/v1/identity").respond(json=_identity_payload())
        router.post(f"{BASE}/v1/ingest").respond(status_code=204)
        adapter = await HttpAdapter.connect(BASE)
        await adapter.ingest_session(_session(), "case_1")


async def test_ingest_error_raises_with_status_code() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{BASE}/v1/identity").respond(json=_identity_payload())
        router.post(f"{BASE}/v1/ingest").respond(status_code=400, text="bad case")
        adapter = await HttpAdapter.connect(BASE)
        with pytest.raises(HttpAdapterError) as exc_info:
            await adapter.ingest_session(_session(), "case_1")
    assert exc_info.value.status_code == 400
    assert "bad case" in str(exc_info.value)


async def test_answer_returns_answer_result_with_retrieved_units() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{BASE}/v1/identity").respond(json=_identity_payload())
        router.post(f"{BASE}/v1/answer").respond(
            json={
                "answer": "Berlin",
                "retrieval_time_ms": 12.5,
                "generation_time_ms": 200.0,
                "units_retrieved": 2,
                "tokens_retrieved": 100,
                "retrieved": [
                    {
                        "unit_id": "u1",
                        "text": "moved to Berlin",
                        "score": 0.9,
                        "source_turn_ids": ["t1"],
                    },
                    {"unit_id": "u2", "text": "Germany is nice", "score": None},
                ],
            },
        )
        adapter = await HttpAdapter.connect(BASE)
        result = await adapter.answer_question("where?", "case_1")

    assert result.answer == "Berlin"
    assert result.retrieval_time_ms == pytest.approx(12.5)
    assert result.generation_time_ms == pytest.approx(200.0)
    assert result.units_retrieved == 2
    assert result.tokens_retrieved == 100
    assert len(result.retrieved) == 2
    assert result.retrieved[0].unit_id == "u1"
    assert result.retrieved[0].text == "moved to Berlin"
    assert result.retrieved[0].score == pytest.approx(0.9)
    assert result.retrieved[0].source_turn_ids == ("t1",)
    # Missing score coerces to None, missing source_turn_ids to ().
    assert result.retrieved[1].score is None
    assert result.retrieved[1].source_turn_ids == ()


async def test_answer_tolerates_minimal_payload() -> None:
    """A service that only sets `answer` must still produce a valid result."""

    with respx.mock(assert_all_called=True) as router:
        router.get(f"{BASE}/v1/identity").respond(json=_identity_payload())
        router.post(f"{BASE}/v1/answer").respond(json={"answer": "42"})
        adapter = await HttpAdapter.connect(BASE)
        result = await adapter.answer_question("q", "case_1")
    assert result.answer == "42"
    assert result.retrieval_time_ms == 0.0
    assert result.generation_time_ms == 0.0
    assert result.units_retrieved == 0
    assert result.tokens_retrieved == 0
    assert result.retrieved == ()


async def test_answer_rejects_missing_answer_field() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{BASE}/v1/identity").respond(json=_identity_payload())
        router.post(f"{BASE}/v1/answer").respond(json={"retrieval_time_ms": 1.0})
        adapter = await HttpAdapter.connect(BASE)
        with pytest.raises(HttpAdapterError, match="'answer' field"):
            await adapter.answer_question("q", "case_1")


async def test_answer_rejects_non_string_answer() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{BASE}/v1/identity").respond(json=_identity_payload())
        router.post(f"{BASE}/v1/answer").respond(json={"answer": 42})
        adapter = await HttpAdapter.connect(BASE)
        with pytest.raises(HttpAdapterError, match="must be a string"):
            await adapter.answer_question("q", "case_1")


async def test_reset_posts_empty_body() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{BASE}/v1/identity").respond(json=_identity_payload())
        reset = router.post(f"{BASE}/v1/reset").respond(status_code=204)
        adapter = await HttpAdapter.connect(BASE)
        await adapter.reset()
    body = json.loads(reset.calls.last.request.content or b"{}")
    assert body == {}


async def test_transport_error_maps_to_adapter_error() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{BASE}/v1/identity").respond(json=_identity_payload())
        router.post(f"{BASE}/v1/answer").mock(
            side_effect=httpx.ConnectError("refused"),
        )
        adapter = await HttpAdapter.connect(BASE)
        with pytest.raises(HttpAdapterError, match="Transport error"):
            await adapter.answer_question("q", "case_1")


async def test_headers_are_sent_on_every_request() -> None:
    headers = {"Authorization": "Bearer TOKEN-123", "X-Run-Id": "amb-42"}
    with respx.mock(assert_all_called=True) as router:
        identity = router.get(f"{BASE}/v1/identity").respond(json=_identity_payload())
        ingest = router.post(f"{BASE}/v1/ingest").respond(status_code=204)
        adapter = await HttpAdapter.connect(BASE, headers=headers)
        await adapter.ingest_session(_session(), "case_1")

    for req in (identity.calls.last.request, ingest.calls.last.request):
        assert req.headers["authorization"] == "Bearer TOKEN-123"
        assert req.headers["x-run-id"] == "amb-42"


async def test_non_json_response_body_raises_clear_error() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{BASE}/v1/identity").respond(json=_identity_payload())
        router.post(f"{BASE}/v1/answer").respond(
            status_code=200,
            content=b"not json",
            headers={"content-type": "text/plain"},
        )
        adapter = await HttpAdapter.connect(BASE)
        with pytest.raises(HttpAdapterError, match="non-JSON body"):
            await adapter.answer_question("q", "case_1")


async def test_persistence_refused_when_service_does_not_support_it(
    tmp_path: Path,
) -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{BASE}/v1/identity").respond(
            json=_identity_payload(supports_persistence=False),
        )
        adapter = await HttpAdapter.connect(BASE)
        with pytest.raises(NotImplementedError, match="supports_persistence=false"):
            await adapter.save_state(tmp_path)
        with pytest.raises(NotImplementedError, match="supports_persistence=false"):
            await adapter.load_state(tmp_path)


async def test_save_and_load_state_roundtrip(tmp_path: Path) -> None:
    blob = b"\x00\x01\x02opaque-state"
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{BASE}/v1/identity").respond(
            json=_identity_payload(supports_persistence=True),
        )
        router.get(f"{BASE}/v1/state").respond(
            status_code=200,
            content=blob,
            headers={"content-type": "application/octet-stream"},
        )
        put = router.put(f"{BASE}/v1/state").respond(status_code=204)

        adapter = await HttpAdapter.connect(BASE)
        assert adapter.supports_persistence is True

        save_dir = tmp_path / "state-out"
        await adapter.save_state(save_dir)
        assert (save_dir / "state.bin").read_bytes() == blob

        await adapter.load_state(save_dir)
        assert put.called
        assert put.calls.last.request.content == blob
        assert put.calls.last.request.headers["content-type"] == "application/octet-stream"


async def test_context_manager_opens_and_closes() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{BASE}/v1/identity").respond(json=_identity_payload())
        adapter = HttpAdapter(BASE)
        async with adapter as opened:
            assert opened is adapter
            assert opened.memory_system_id == "example-mem"


async def test_empty_base_url_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        HttpAdapter("")


async def test_close_is_idempotent_on_owned_client() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{BASE}/v1/identity").respond(json=_identity_payload())
        adapter = await HttpAdapter.connect(BASE)
        await adapter.close()
        await adapter.close()  # second close must not raise


async def test_external_client_is_not_closed_by_adapter() -> None:
    async with httpx.AsyncClient() as client:
        with respx.mock(assert_all_called=True) as router:
            router.get(f"{BASE}/v1/identity").respond(json=_identity_payload())
            adapter = await HttpAdapter.connect(BASE, client=client)
            await adapter.close()
        # Client still usable after adapter.close().
        assert not client.is_closed
