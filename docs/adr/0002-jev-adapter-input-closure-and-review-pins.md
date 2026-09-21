# ADR 0002: Jev Adapter Input Closure, Corpus Pairing Fields, and Review-Pinned Rules

## Status

Accepted (tribunal ruling under Alan's AFK delegation, 2026-09-22; seats: DeepSeek v4 Pro, GPT 5.6-terra, GLM 5.3, Claude Sonnet 5 — Grok 4.6 not convened, xAI spending limit)

## Context

The PR #20 review (both passes, 17 confirmed findings) proved that issue #6's
done-when — no later Stage 1 issue may decide a contract rule — was threatened
in seven rule areas. The upstream Jev documentation leaves unknown-field
behavior for its request envelope unspecified, so closure is a local policy
choice, not a parity claim. Issue #21 pinned these rules before #7 freezes
JSON Schemas.

## Decision

1. The Jev adapter input is a closed object with exactly `state`, `model`,
   and `questions`. The adapter validates it with the native structural codes
   before conversion. Input rejections keep the adapter result shape with
   `answers` and `local_judge` null and the structural code in `error`;
   `JEV_ADAPTER_UNMAPPABLE_RESULT` stays reserved for unmappable results.
2. The corpus record gains `inference` (optional, request shape),
   `matched_case_id` (required non-null for adversarial and metamorphic cases,
   must reference an existing case in the same corpus version), and
   `metamorphic_relation` (four-value enum matching the relations the
   acceptance gate already declares).
3. Replay-body validation reuses the structural codes with JSON Pointer paths
   into the replay body; replay does not verify trace integrity
   (`canonical_request_hash` is a correlation aid, consistent with the
   trace-is-not-an-audit-proof rule).
4. `path` is a JSON Pointer to the offending field when one exists, and the
   empty string for whole-request and runtime causes. Every response object is
   closed. Score `legend` keys are the decimal level-index strings. The
   acceptance metrics and their denominators are pinned in the contract; the
   0.1/0.05 invariance tolerances are owned design constants (0.05 is half the
   pinned Noul bucket width), recalibratable via corpus version without a
   contract break. A complete error-code registry table with classes lives in
   the contract; HTTP status mappings stay in the API reference. Adapter
   profiles may not set `sample_count` below 2 (agreement is null at one
   sample, and the adapter refuses results without agreement).

## Alternatives rejected

- **Lenient adapter input** (ignore or strip unknown fields): silent wrongness
  is what the honesty non-claims exist to prevent; a strip rule can never be
  tightened without breaking callers who relied on it.
- **One adapter code for input and result failures**: makes
  `JEV_ADAPTER_UNMAPPABLE_RESULT` mean two caller-actionable things (fix your
  request vs your result did not map).
- **Corpus linkage by convention only**: defers a contract decision past its
  deadline; gates 4 and 5 become incomputable.
- **Full corpus JSON Schema in the contract**: freezes harness detail the
  evidence gate does not need; v2 risk without v2 benefit.
- **Verifying `canonical_request_hash` at replay**: an implicit audit-integrity
  claim the contract's own vocabulary disclaims; adds a failure mode and a
  code for behavior the engine never promised.
- **Splitting `REQUEST_TOO_LARGE` into two codes**: the single code is
  face-agnostic; the size/count distinction is an HTTP-mapping concern that
  belongs to the API reference.

## Consequences

- #7 can encode the adapter input schema, replay validation, the error
  registry, and the corpus record without deciding any contract rule.
- If Jev later publishes an input-validation spec, divergence is handled by a
  contract version bump, not silent realignment.
- The invariance tolerances must be exercised by a metamorphic pilot before
  Stage 5 evidence relies on them; tightening is a corpus-version change.
