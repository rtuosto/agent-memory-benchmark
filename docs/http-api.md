# HTTP adapter contract

The benchmark ships an `HttpAdapter` that speaks a small REST protocol so a
memory system can be benchmarked over the network with **zero Python-level
coupling**. The machine-readable source of truth is
[`openapi.yaml`](../openapi.yaml); this document is the narrative reference
for service authors.

Run a benchmark against an HTTP service with:

```bash
amb run longmemeval \
    --memory http://localhost:8000 \
    --memory-header Authorization="Bearer ${TOKEN}" \
    --answer-model ollama:llama3.1:8b \
    --judge-model ollama:llama3.1:70b \
    --split s --limit 5
```

## Required endpoints

### `GET /v1/identity`

Returns the memory system's identity. The benchmark calls this once at run
start — **before** the first `ingest` — so `memory_system_id` and
`memory_version` flow into cache keys and the run manifest.

```json
{
  "memory_system_id": "example-mem",
  "memory_version": "1.4.2",
  "protocol_version": "0.1",
  "supports_persistence": false
}
```

`memory_version` MUST be bumped whenever ingestion semantics or stored-state
shape change. The benchmark's ingestion cache keys on `memory_version`; a
stale cache auto-invalidates once the service reports a new value.

`supports_persistence` (default `false`) controls whether the benchmark
exercises `GET /v1/state` and `PUT /v1/state`. Services that don't support
state snapshotting should simply omit those endpoints and return
`supports_persistence: false`.

### `POST /v1/ingest`

Ingest one `Session` for a given case. Sessions arrive in chronological
order (`session_index` strictly increasing within a `case_id`) and the
benchmark does not interleave cases — every session for a `case_id` lands
before any `answer_question` for that case.

Request body:

```json
{
  "case_id": "longmemeval_s_000",
  "session": {
    "session_index": 1,
    "session_time": "2024-01-15",
    "session_id": "sess_1",
    "turns": [
      {
        "turn_id": "sess_1:0",
        "speaker": "user",
        "text": "I moved to Berlin last spring.",
        "timestamp": null,
        "image_caption": null
      }
    ]
  }
}
```

Response: `200 {"ok": true}` or `204 No Content`. Any 4xx/5xx is treated as
a hard failure and aborts the run.

### `POST /v1/answer`

Produce one answer. The service performs its own retrieval and generation;
the benchmark independently measures wall time, so the `*_time_ms` fields
in the response are informational self-reports (drift between self-reports
and runner-measured time is logged per-query as a telemetry signal).

Request body:

```json
{
  "case_id": "longmemeval_s_000",
  "question": "When did I move to Berlin?"
}
```

Response body:

```json
{
  "answer": "Last spring.",
  "retrieval_time_ms": 12.3,
  "generation_time_ms": 456.7,
  "units_retrieved": 3,
  "tokens_retrieved": 512,
  "retrieved": [
    {
      "unit_id": "chunk_0042",
      "text": "I moved to Berlin last spring.",
      "score": 0.94,
      "source_turn_ids": ["sess_1:0"]
    }
  ]
}
```

**`retrieved[*].text` is load-bearing for evidence KPIs.** The benchmark
attributes retrieval to evidence turns via SQuAD-normalized token-multiset
overlap on `text`. If you return an empty `retrieved` array, evidence
metrics will stay at zero for that question — that's correct for a null
baseline but usually means a real memory system is under-reporting what it
sent to the answerer.

`source_turn_ids` is optional provenance: it's stored on the per-question
record for diagnostics but NOT consulted by the scorer.

### `POST /v1/reset`

Drop all in-memory state. Called between case groups and by the benchmark's
cleanup paths. After a successful reset the service MUST behave as if no
`ingest` had ever happened. Request body is an empty JSON object; response
`200 {"ok": true}` or `204 No Content`.

## Optional endpoints (persistence)

Only implement these if you advertise `supports_persistence: true` in
`/v1/identity`. They let the benchmark cache ingested state and skip
re-ingestion across runs.

### `GET /v1/state`

Return internal state as **opaque bytes**. The benchmark writes the body
verbatim as `state.bin` inside the ingestion-cache directory; the body
format is entirely the service's choice.

### `PUT /v1/state`

Restore state from bytes previously returned by `GET /v1/state`. Request
body is `application/octet-stream`; response is `200 {"ok": true}` or
`204 No Content`.

Services that don't implement these endpoints should return
`supports_persistence: false` from `/v1/identity`; the benchmark will fall
back to re-ingesting on every run.

## Error payloads

Any endpoint may return a JSON error on 4xx/5xx:

```json
{"error": "case_id not found", "detail": "ingest_session never called for case_id=foo"}
```

The benchmark surfaces the HTTP status code plus the first ~500 bytes of
the response body in the run's error output.

## Headers and authentication

Every request the benchmark sends carries whatever headers were configured
at CLI time via `--memory-header NAME=VALUE` (repeatable). Typical use:

```
--memory-header Authorization="Bearer ${TOKEN}"
--memory-header X-Request-ID=amb-run-42
```

The benchmark does not inject any headers by itself — `Content-Type:
application/json` is set by the HTTP client for POST requests automatically.

## Timing expectations

The benchmark's default request timeout is 600 s, matching the Ollama
provider. Services that do heavy cold-cache work on the first `ingest` or
`answer` call should still return within that window; if your service needs
warm-up, do it out-of-band before pointing the benchmark at it.

## Versioning

The benchmark pins `protocol_version: "0.1"` for v0.1 of this contract. An
incompatible change to any request/response shape will bump the major
version and the benchmark's `HttpAdapter` will refuse to connect to a
service that advertises a mismatched protocol.

## Reference

The authoritative schema is [`openapi.yaml`](../openapi.yaml); keep this
document and that file in sync. If you're writing a service from scratch,
copy the example payloads above, implement the four required endpoints,
and run `amb run longmemeval --memory http://...` against a tiny subset
(`--limit 3`) to smoke-test the integration.
