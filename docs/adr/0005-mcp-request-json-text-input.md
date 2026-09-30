# ADR 0005: MCP request objects and raw JSON text

## Status

Accepted — 2026-09-29. The tribunal on 2026-09-29 recommended this direction
(cross-review mean 8.4/10, gate 7.0), and the owner authorized applying it to
paused PR #42 and completing its remaining implementation/review cycle:
"Alan authorized one additional bounded fix/review cycle; deeply nested MCP
input must return a typed MALFORMED_JSON rejection." The ruling was given in
session and is mirrored on issue #45.

## Context

The MCP stdio transport parses the outer JSON-RPC message before dispatching a
tool call. A deeply nested request supplied as an object can exceed that
parser's recursion limit before local-judge receives it, so the shared
evaluator cannot return its typed `MALFORMED_JSON` result. The HTTP and library
paths already pass raw request bytes to the shared evaluator, which owns the
256 KiB limit and structural rejection behavior.

## Decision

Each MCP tool accepts either its current structured request object or a raw
JSON string containing that request. Object inputs retain their existing
behavior. String inputs are UTF-8 encoded and passed directly to the same
shared evaluator, allowing its parser to return a typed `MALFORMED_JSON`
result for deep or malformed JSON. MCP callers with deeply nested payloads
must send the envelope as JSON text inside the `request` argument; the
ordinary object form remains supported for typical requests.

## Alternatives considered

- **Replace or wrap the stdio parser and dispatcher:** rejected for this change;
  it would add a second protocol implementation around three thin tools and
  would need to preserve request IDs and MCP error semantics.
- **Return only a protocol-level parse error or document a nesting limit:**
  rejected because neither satisfies the required typed local-judge rejection.
- **Make the argument string-only:** rejected because it would break existing
  object-form MCP callers unnecessarily.

## Consequences

- Existing object-form callers remain supported; the MCP input schema also
  admits a string.
- Deep MCP payloads must be serialized by the caller and supplied as raw JSON
  text. The shared evaluator still enforces the existing byte and structural
  limits.
- The API reference and stdio regression tests must cover both forms, including
  a real deep-input MCP call returning a typed result without invoking a model.

## Decision addendum: MCP serialization failures

Accepted by Alan during Plan-Loop on 2026-09-30, issue #46.

When an MCP request value reaches the tool wrapper but cannot be serialized or
encoded as UTF-8 JSON, the wrapper returns the tool-specific typed
`MALFORMED_JSON` result before invoking the supplied handler. The error object
is `{code: "MALFORMED_JSON", path: "", message: "the MCP request could not be
serialized as UTF-8 JSON"}`. Native and replay tools use the rejection envelope
with `contract_version: null`, `model: null`, `status: "rejected"`, and empty
`results`; the Jev tool uses `answers: null` and `local_judge: null`. The input
and exception text are never echoed. This intentionally changes the custom
handler callback contract for serialization failures: those handlers are not
called.

The conversion boundary is limited to `json.dumps(..., ensure_ascii=False)`
and UTF-8 encoding for object arguments, or UTF-8 encoding for raw-text
arguments. Only `RecursionError`, `TypeError`, and `ValueError` from those
conversions (including `UnicodeEncodeError`) are normalized. Existing
`json.dumps` options and successfully serialized values, including non-finite
numbers, remain unchanged; handler exceptions and output-serialization errors
are not caught by this branch.

A successfully encoded raw JSON string, including an empty string, continues
to the handler unchanged. Custom handlers retain responsibility for the result
of such input. Encoded but malformed raw JSON also continues to the handler.
The fail-closed default Jev handler returns its typed malformed-input envelope
for empty or malformed raw text. Handler exceptions and result serialization
failures remain tool-level errors. This addendum does not change pre-dispatch
FastMCP JSON-RPC parsing; deeply nested MCP objects must still use the raw-text
request form above, and successfully serialized values retain the existing
`json.dumps` behavior.
