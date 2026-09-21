# ADR 0001: Native Contract Uses Jev-Shaped Maps and Caller-Declared Policy

## Status

Accepted

## Context

`local-judge` promises a documented Jev-compatible subset, but it runs local
models and must not imply Jev calibration or security properties it cannot
establish.

The initial planning discussion used an ordered `questions[]` array. The
authoritative TypeSafe API documentation instead defines `questions` as a map
from caller-selected IDs to typed questions and returns `answers` under those
same IDs. The user selected the map shape after this conflict was found.

The project also needs a boundary between caller-supplied task policy and
untrusted state. The project has no identity system, registry, multi-tenancy,
or authorization requirement from which to authenticate policy provenance.

## Decision

1. Native v1 uses a nonempty `questions` map and keyed result map. Map order is
   not a public execution or response-order guarantee.
2. Native v1 accepts state only as a JSON string, object, or array. The engine
   treats every value within state as evidence data, not instruction authority.
3. Instructions, criteria, type configuration, menus, rubrics, and
   `policy.version` are caller-declared policy. Traces label their provenance
   `caller-declared-unverified`.
4. The engine validates the envelope and typed policy objects strictly, but it
   deliberately permits arbitrary JSON content inside state and structured
   instructions or criteria where the contract permits them.
5. The native result exposes empirical vote shares and agreement. It does not
   call them calibrated probabilities or confidence. A Jev adapter may map
   agreement to the Jev `confidence` field only with an explicit disclosure.
6. Native per-question failures are an intentional divergence from Jev's
   request-level validation errors. The adapter refuses unmappable partial
   results rather than inventing Jev-shaped answers.

## Consequences

- Consumers can use the documented Jev request keying model without an
  array-to-map translation.
- State-containing prompt injection remains an adversarial quality problem to
  measure, not an attack that the structure alone claims to detect or prevent.
- Each caller remains responsible for building complete candidate menus,
  selecting actions, authorization, and retaining traces if it needs durable
  audit storage.
- The contract must keep the native and adapter semantics visibly separate.

## Sources

- https://docs.typesafe.ai/api.md (read 2026-09-21)
- https://docs.typesafe.ai/primitives/choice.md (read 2026-09-21)
- https://docs.typesafe.ai/primitives/score.md (read 2026-09-21)
- https://docs.typesafe.ai/primitives/noul.md (read 2026-09-21)
