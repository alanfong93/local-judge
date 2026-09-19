# local-judge

**A local decision engine that speaks the Jev wire format. Send state and typed questions; get back typed answers, probabilities, and confidence — from your own models, on your own machine.**

Large decision models (like TypeSafe's Jev) showed that software doesn't need a text generator for filter / verification / triage steps — it needs fast, structured judgments. But their inference runs on someone else's servers, costs money per token, and works best in English.

`local-judge` implements the same request/response shape over local models (Ollama) — as a documented subset — so most Jev-targeted tooling works here by changing the base URL, while every byte stays on your machine and costs nothing.

**Status: design stage. Nothing is built yet.**

---

## The port, not the phone

The interface is the standard: state + typed questions (`choice` / `score` / `noul`, each with instructions and criteria) → answers constrained to your menu, with probabilities and a confidence value. You don't design a power bank for one phone; you fit the standard port and any device works. Same here: consumers validate the port, they don't shape it.

```
                     ┌──────────────────────────────┐
                     │   local-judge (one core)     │
                     │  POST /v1/systemone-shaped   │
                     │  state + questions →         │
                     │  menu-enforced answers +     │
                     │  probabilities + confidence  │
                     └───────┬─────────┬───────┬────┘
                        HTTP│       MCP│    lib│
                            │         │       │
                     n8n workflows  agents  Python projects
                                  (OpenCode)  (ops-guard)
```

## Design decisions

1. **Jev-compatible request/response subset** — the `choice` and `noul` question types follow the public Jev wire shape; `score` arrives with its first consumer. Field-level differences are documented, never silently divergent, and the compatibility contract gets pinned by published fixtures.
2. **Menu enforcement in code, not prompting.** The engine physically cannot return an answer outside the options you supplied.
3. **Agreement, not calibrated confidence.** The native result exposes `agreement` — vote share across repeated samples — and never presents it as a calibrated probability. A Jev-compat adapter maps it onto the `confidence` field with a documented weaker guarantee, so a consumer cannot mistake one for the other.
4. **Three faces, one core**: HTTP for workflow tools (n8n), MCP for AI agents, importable library for Python projects.
5. **Fail closed**: model returns garbage → fallback, never an invented answer; model server down → error, not a guess.
6. **Consumer-agnostic scope**: v0 ships `noul` + `choice`; `score` lands with its first real consumer.

## Scope

**In:** the wire-compatible evaluation endpoint, menu enforcement, sample-based confidence, the HTTP / MCP / library faces, a spec test suite.

**Out, for now:** text generation, model training/fine-tuning, speed or calibration parity with hosted decision models, multi-tenancy, a hosted service.

## Not affiliated

local-judge is an independent implementation of a public request/response shape, for people who want judgments from local models. It is not affiliated with, endorsed by, or connected to TypeSafe AI.

## Licence

Not yet chosen.
