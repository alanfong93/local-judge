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
