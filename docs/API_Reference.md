# API Reference

## HTTP v1

The HTTP adapter is a local FastAPI server. It exposes no authentication,
tenant, storage, or hosted-inference behavior.

| Method | Path | Request | Success | Rejection |
| --- | --- | --- | --- | --- |
| `POST` | `/v1/evaluations` | Native v1 evaluation envelope | `200` native completed result | `400` native rejected result; `413` when `REQUEST_TOO_LARGE` is caused by the 256 KiB limit, `400` when caused by more than 64 questions |
| `POST` | `/v1/replays` | `{ "contract_version": "v1", "trace": <inline result trace> }` | `200` native completed result | `400` invalid replay body; `409` native rejected result with `REPLAY_CONFIGURATION_UNAVAILABLE` |
| `POST` | `/v1/jev/evaluations` | Documented Jev-shaped input map | `200` Jev adapter result | `400` adapter result whose `error` is a native structural code (input validation) or `JEV_ADAPTER_UNMAPPABLE_RESULT` (unmappable result) |

Request and response fields are normative in `docs/CONTRACT.md`; the generated
OpenAPI document must reference the Stage 1 JSON Schema definitions.

## MCP v1

The local stdio FastMCP server exposes three thin tools:

| Tool | Input | Output |
| --- | --- | --- |
| `local_judge_evaluate` | Native v1 evaluation envelope | Native result, including rejected results |
| `local_judge_replay` | Self-contained native replay envelope | Native result, including configuration-unavailable rejection |
| `local_judge_evaluate_jev` | Documented Jev-shaped input map | Jev adapter result, including the `local_judge.traces` extension |

Tool-level execution errors are reserved for server startup or protocol failure.
Contract validation and model outcomes are returned as typed result objects.
Each tool receives the request wrapped as `{"request": <request or raw JSON
text>}` and returns the typed result object as JSON text.

### Request forms

The `request` argument accepts either form:

- **Object** — the request as a structured object. Ordinary requests use this
  form; it behaves exactly as before this union existed.
- **Raw JSON text** — the request serialized to a JSON string. The string's
  bytes go straight to the shared evaluator, which owns parsing, the 256 KiB
  limit, and structural rejection.

Deeply nested requests **must** use the raw-text form. The stdio transport
parses the whole JSON-RPC message before a tool runs, so a deeply nested
object can exceed that parser's recursion limit before local-judge sees it;
the message is then dropped with no response (ADR 0005). As text, the outer
parser sees only a string, and the shared evaluator returns the typed
`MALFORMED_JSON` result without a model call.

Consumer example — deep request as raw text:

```python
import json

from fastmcp import Client

envelope = {"contract_version": "v1", "state": deep_state, "model": "qwen3:8b",
            "policy": {"version": "p"}, "inference": {},
            "questions": {"q": {"type": "noul", "instructions": "x"}}}

async with Client(server) as client:
    result = await client.call_tool("local_judge_evaluate", {"request": json.dumps(envelope)})
```

`local_judge_replay` and `local_judge_evaluate_jev` take their envelopes in
the same two forms and keep their existing typed result shapes.
