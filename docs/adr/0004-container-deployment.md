# ADR 0004: Single-image HTTP and MCP stdio container

## Status

Accepted by Alan (2026-09-26; issue #39).

## Context

local-judge has HTTP and MCP access adapters, but they do not provide a
standalone service composition: `api.create_app` requires evaluator
handlers, and the default MCP command intentionally fails closed without
a configured runner. A Docker image therefore needs to compose the existing
core, model port, endpoint configuration, and handlers before starting either
access mode.

Alan runs Open WebUI separately at host port 3040 and wants local-judge to use
it without joining its Docker network. The HTTP API has no authentication. Alan
explicitly chose to publish its port on all host interfaces for host, n8n, and
LAN clients.

## Decision

1. Ship one local-judge image with HTTP as its default mode and an explicit
   MCP stdio mode. Both modes load the same deployment configuration and use
   the same native, replay, and Jev evaluator handlers.
2. Configure one OpenAI-compatible API prefix and a comma-separated model-ID
   allowlist, with an optional outbound bearer key and a selected response
   format mode. Configuration is deployment-owned; request bodies cannot
   change the endpoint, credentials, or allowlist.
3. Keep Open WebUI external. The container connects to its published host
   endpoint at `http://host.docker.internal:3040/api`; Compose does not join
   the Open WebUI or n8n networks and does not bundle any model server.
4. Bind HTTP to `0.0.0.0:8000` inside the image and publish `8000:8000` on all
   host interfaces. This is intentionally unauthenticated and LAN-reachable.
   The standard Python library server remains loopback-only.
5. Package the current prompt, output-schema, aggregation, and configured
   model-profile artifacts needed for replay. If the recorded versions are
   unavailable, return `REPLAY_CONFIGURATION_UNAVAILABLE` before an inference
   call.

## Alternatives considered

- **Separate HTTP and MCP images:** rejected; both adapters share the same
  evaluator and deployment config, so a single image avoids version drift.
- **Join the Open WebUI or n8n Docker networks:** rejected; host-published
  ports are the user's chosen connectivity boundary.
- **Bind the host port to loopback only:** rejected for this deployment;
  n8n's container and LAN clients need to reach the host-published HTTP port.
- **Add authentication as part of this task:** rejected as a separate public
  API/security feature. The unauthenticated LAN exposure is an explicit user
  choice, not a secure default for untrusted networks.
- **Bundle Ollama, Open WebUI, or model weights:** rejected; existing model
  services remain separate and are accessed through the configured endpoint.

## Consequences

- Compose starts the HTTP mode, while `docker compose run --rm -i local-judge
  mcp` runs MCP over stdio from the same image.
- Port 8000 is reachable from the LAN and has no authentication. The operator
  must only run this configuration on a trusted network; an untrusted-network
  deployment requires a separate authentication/security decision.
- API keys are supplied through local `.env` configuration, ignored by Git,
  and never baked into the image or copied into result traces.
- Docker Desktop resolves `host.docker.internal`; Compose also supplies the
  `host-gateway` mapping for engines that need it.
- The image exposes the same documented HTTP and MCP operations as the
  existing package; this adds no new API route or MCP tool.
