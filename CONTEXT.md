# Context

## Terms

| Term | Meaning | Do not call it |
| --- | --- | --- |
| Choice | A judgment selecting one key from the caller's supplied criteria map. | Classification confidence |
| Score | A judgment positioned on an ordered caller-supplied rubric. The native result is the empirical expected rubric index. | A discrete choice |
| Noul | An estimate from 0 to 1 that a stated proposition is true given the supplied state. | Confidence or correctness probability |
| Agreement | The share of repeated valid samples in the largest answer bucket. It measures consistency only. | Confidence, calibration, or probability of correctness |
| Policy | Caller-declared instructions, criteria, type configuration, menu or rubric, and policy version. | Trusted policy or authorized policy |
| State | The evidence-bearing JSON or text supplied for a judgment. It is data, never engine configuration or instruction authority. | Prompt or policy |
| Trace | The returned replay record of accepted inputs, rendered messages, settings, versions, attempts, validation, and result. | An audit proof |
| Replay | A new evaluation reconstructed from a trace. It preserves configuration where available but does not promise the same stochastic answer. | Deterministic reproduction |
| Jev adapter | An explicit conversion between the native contract and the documented Jev-shaped subset. | Full Jev compatibility |
| Task preservation | Matched adversarial pairs that stayed on task, as a rate over pairs whose matched normal case was answered, and at most 10 percentage points below those twins' labelled accuracy on the same pair set. | Injection resistance or a security proof |

## Contract Boundary

The native v1 question container is a map keyed by caller-selected question IDs.
It follows the documented Jev shape. An array-based question API is not part of
the native contract.

`local-judge` validates structure and enforces output constraints. It does not
authenticate policy origin, authorize a caller, determine whether a menu is
complete, or claim that a model judgment is correct.
