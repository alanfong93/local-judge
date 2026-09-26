# ADR 0003: OpenAI-Compatible Chat Completions as the Endpoint Port Protocol

## Status

Accepted (issue #37 design, 2026-09-26)

## Context

The model-port boundary had exactly one adapter: Ollama, pinned to literal
loopback addresses. Alan's automation estate runs models behind an Open WebUI
container, and other OpenAI-compatible endpoints exist (vLLM, llama.cpp
server, LiteLLM, most hosted providers). The port needed to reach models
served through an endpoint without adding per-provider clients, weakening the
Ollama local-only guarantee, or letting request data configure inference.
An endpoint may receive the supplied state and prompt and may charge for
inference, so pointing local-judge at one must remain an explicit,
deployment-owned choice.

## Decision

1. The endpoint model port speaks the OpenAI-compatible Chat Completions
   protocol — the de-facto cross-provider standard — instead of one adapter
   per provider or vendor SDK.
2. Support is limited to the standard non-streaming chat-completions
   request/response shape: one bounded POST per attempt, no streaming, no
   automatic retries, no redirect following, no tools, embeddings, or image
   input. Provider management stays outside the project.
3. Profiles are deployment-owned configuration — model ID, exact API base
   URL prefix, optional bearer key, response-format mode, and declared
   inference-setting capabilities. Endpoint, model, and key values never
   come from request data.
4. URL joining is exact: `base_url` is the API prefix exactly as configured,
   trailing slashes removed, `/chat/completions` appended, and nothing is
   inferred or appended (`/v1` is never added automatically). Open WebUI is
   configured with `/api`; conventional providers with `/v1`; a bare host is
   valid only when its route is `/chat/completions`.
5. Ollama remains the local adapter with unchanged loopback-only validation.
   The endpoint adapter permits local and remote endpoints; docs disclose
   that a configured endpoint may receive judged data and incur charges.

## Alternatives rejected

- **Per-provider adapters** (native Anthropic, Google, Mistral, … protocols):
  one client per provider multiplies maintenance and security surface for
  providers that almost all already expose the OpenAI-compatible shape; a new
  provider protocol would need its own ADR.
- **Vendor SDKs** (for example the `openai` Python package): adds a runtime
  dependency to a zero-dependency core; the bounded stdlib HTTP transport is
  already written, shared with Ollama, and tested for the exact
  one-attempt-no-redirect contract.
- **Streaming (SSE) support**: the native contract needs one complete JSON
  value per sample; streaming adds parser state and partial-output failure
  modes that judgments do not benefit from.
- **Inferring the path prefix** (auto-appending `/v1`, probing the server):
  magic URL rewriting silently sends prompts and keys to a different route or
  host than the operator configured; exact joining keeps misconfiguration
  visible in the configuration itself.
- **Remote endpoint as the default model source**: would silently replace the
  local-data guarantee the product stands on. Ollama stays the default; the
  endpoint is opt-in per deployment.

## Consequences

- Open WebUI and other OpenAI-compatible endpoints work through one profile
  with no new dependency.
- Providers without chat-completions compatibility remain unsupported until
  a dedicated ADR covers them.
- `response_format` support varies across endpoints and models, so the mode
  is an explicit profile field (`json_schema` default, `json_object`
  fallback) with no silent downgrade; local typed-output validation remains
  authoritative either way.
- A configured endpoint may receive judged data and may bill for it; the
  README and product docs must keep disclosing this so the local-first
  promise stays honest.
