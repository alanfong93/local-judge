# Native Contract v1

## Status

This is the normative Stage 1 design for the native `local-judge` contract.
It defines the contract that implementation work must follow. It does not add
an HTTP endpoint, MCP tool, or Python package.

## Scope

Native v1 evaluates a shared state against a nonempty map of independent
Choice, Score, and Noul questions using a selected local model. It provides
typed answers, empirical repeated-sample agreement, per-question failure
reporting, and an inline trace for replay.

Native v1 does not provide calibrated confidence, policy authentication,
authorization, a prompt-injection detector, attack-triggered abstention,
server-side trace retention, automatic retries, cancellation, or a guarantee
of parallel execution.

## Upstream Reference

The documented Jev source uses `state`, `model`, and a map of typed
`questions`; it returns answers under the same question IDs. Choice returns a
selected option, option probabilities, and confidence. Score returns a
probability-weighted position on an ordered rubric. Noul returns the
probability that a proposition is true.

Native v1 preserves the map-based shape and the three primitive meanings while
explicitly diverging where local repeated sampling cannot establish Jev's
probability or confidence semantics. The exact mapping is defined in
[Jev Adapter](#jev-adapter).

Sources read on 2026-09-21 and re-verified on 2026-09-22:

- https://docs.typesafe.ai/api.md
- https://docs.typesafe.ai/primitives/choice.md
- https://docs.typesafe.ai/primitives/score.md
- https://docs.typesafe.ai/primitives/noul.md

## JSON Content

`JSONContent` is a JSON string, object, or array. A `state` value is required
and must be `JSONContent`. Nested object and array values may contain ordinary
JSON scalars, arrays, objects, and nulls.

The contract treats all state content as evidence data. Text such as `ignore
the policy`, JSON keys such as `system`, and fake schemas within state do not
change the engine's policy, output schema, model configuration, or execution
rules.

## Request

The top-level envelope is closed. Unknown top-level fields are a structural
error.

```json
{
  "contract_version": "v1",
  "state": {"ticket": "Example evidence"},
  "model": "qwen3:8b",
  "policy": {"version": "support-triage-2026-09-21"},
  "inference": {
    "sample_count": 1,
    "temperature": 0,
    "timeout_ms": 30000
  },
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoices, and refunds",
        "technical": "Bugs, outages, and integrations"
      }
    }
  }
}
```

| Field | Rule |
| --- | --- |
| `contract_version` | Required and exactly `v1`. |
| `state` | Required `JSONContent`. The envelope validates its outer JSON type but does not interpret nested fields as contract fields. |
| `model` | Required nonempty string naming a local model profile. |
| `policy` | Required closed object with required nonempty `version`. The version is caller-declared and is not checked against a registry. |
| `inference` | Required closed object. Omitted optional fields use the defaults below. |
| `questions` | Required nonempty map from nonempty unique IDs to question objects. A question ID is a correlation key and is never rendered into the model prompt. |

`inference` has the following closed shape:

| Field | Rule |
| --- | --- |
| `sample_count` | Optional integer from 1 through 5. Default: 1. It is the fixed denominator for every aggregate. |
| `temperature` | Optional number from 0 through 2. Default: 0. |
| `seed` | Optional integer. If supplied to a profile that cannot honor it, reject the request with `UNSUPPORTED_INFERENCE_SETTING` before model calls. |
| `timeout_ms` | Optional integer from 1 through 120000. Default: 30000. It applies to one sample attempt. |

The encoded request must not exceed 256 KiB. A request may contain at most 64
questions. A question whose rendered state and policy exceed the selected
model's context limit receives `CONTEXT_LIMIT_EXCEEDED`; valid sibling
questions still run.

## Policy and Prompt Boundary

`policy.version`, each question's `instructions`, and each question's
`criteria` are caller-declared policy. They are traceable but unauthenticated.
`local-judge` never determines whether the caller was entitled to supply them.

The engine compiles a versioned prompt template with three distinct regions:

1. Engine-owned output contract and output schema.
2. Caller-declared policy for the current typed question.
3. State evidence.

The engine never merges state into the policy region and never makes state
keys executable configuration. It sends a backend JSON schema for the
sample-output union and validates the raw result again locally. These are
deterministic structural safeguards, not proof that a model will ignore every
adversarial statement in state.

## Questions

Question objects are closed. Unknown fields are a per-question
`INVALID_QUESTION` error. `instructions` is required `JSONContent`, so callers
may use a string, object, or array. Structured instructions and criteria are
rendered as policy data by the versioned prompt template.

### Choice

```json
{
  "type": "choice",
  "instructions": "Which option best matches the state?",
  "criteria": {
    "option_a": "First described option",
    "option_b": null
  }
}
```

`criteria` is a map of 2 through 255 nonempty option IDs. Each value is
`JSONContent` or null. The option IDs are the complete native Choice menu for
that question. `OTHER` and `ESCALATE` have no engine-reserved meaning; they are
ordinary options only if the caller supplied them.

A Choice answer proves only that the selected ID belonged to the supplied
criteria map. It does not prove the menu was complete or that the choice was
correct.

### Score

```json
{
  "type": "score",
  "instructions": "How severe is the issue?",
  "criteria": [
    "Cosmetic impact",
    "Workaround exists",
    "No workaround exists"
  ]
}
```

`criteria` is an ordered array of 2 through 10 `JSONContent` values. Its array
position is the rubric level: the first value is level 0 and the last is level
`length - 1`. A sample selects one integer level index. The aggregate Score is
an empirical expected position on this rubric, not a model-calibrated value.

### Noul

```json
{
  "type": "noul",
  "instructions": "Is the customer requesting a refund?",
  "criteria": {
    "true": "Explicitly asks for money back",
    "false": "Does not ask for money back"
  }
}
```

`criteria` is optional. If present, it is a closed object whose only permitted
keys are `true` and `false`; each is `JSONContent` or null. A Noul sample is a
finite number from 0 through 1 estimating whether its stated proposition is
true given the state. It is not a measure of degree, calibrated confidence, or
probability that the answer is correct.

## Validation and Execution

### Structural Rejection

Reject the whole request before any model call when the envelope cannot be
interpreted: invalid JSON, an unknown or unsupported contract version, an
unknown envelope field, a missing or malformed required envelope field, an
invalid inference setting, an over-limit payload, an empty or non-map
`questions` value, a duplicate question ID in raw JSON, or an unsupported
requested model profile.

The response status is `rejected`, `results` is empty, and `error` contains a
stable code plus a JSON Pointer path. Structural rejection does not emit a
partial result map. A rejected response sets `contract_version` to `v1` only
when that value parsed successfully, otherwise null; it sets `model` to the
parsed nonempty model string when available, otherwise null.

Every error object has this closed shape:

```json
{
  "code": "INVALID_FIELD",
  "path": "/inference/sample_count",
  "message": "sample_count must be an integer from 1 through 5"
}
```

`path` is an RFC 6901 JSON Pointer to the offending field when one exists, and
the empty string when no single field applies (`MALFORMED_JSON`,
`REQUEST_TOO_LARGE`, and the runtime, replay, and adapter codes). `message` is
a stable contract description, never a backend exception, stack trace, raw
model output, or prompt content.

| Structural code | Meaning |
| --- | --- |
| `MALFORMED_JSON` | The raw request cannot be decoded as one JSON value. |
| `UNSUPPORTED_CONTRACT_VERSION` | `contract_version` is missing or is not `v1`. |
| `UNKNOWN_FIELD` | A closed envelope, policy, or inference object has an extra field. |
| `MISSING_FIELD` | A required envelope, policy, or inference field is absent. |
| `INVALID_FIELD` | A present field has the wrong type, range, or value. |
| `INVALID_QUESTIONS_MAP` | `questions` is not a nonempty JSON object. |
| `DUPLICATE_QUESTION_ID` | The raw JSON contains the same question key more than once. |
| `REQUEST_TOO_LARGE` | The encoded request exceeds 256 KiB or has more than 64 questions. |
| `UNSUPPORTED_INFERENCE_SETTING` | The selected profile cannot honor a supplied inference setting such as `seed`. |
| `UNSUPPORTED_LOCAL_MODEL` | The requested model is not a configured local profile. |

### Question Isolation

After the envelope is valid, each submitted question ID receives exactly one
entry in `results`. A malformed typed question produces `question_error` for
that ID and does not stop valid siblings. Native v1 may evaluate questions
sequentially, but map insertion order is not a public execution guarantee and
no question can observe another question's answer.

There are no automatic retries and no cancellation mechanism in native v1. A
timeout is a per-question error, not a silent retry or a reduced sample count.

### Sample Union

Every sample must be exactly one of:

1. A type-valid answer: a Choice option ID, a Score integer level index, or a
   finite Noul value from 0 through 1.
2. An engine-defined explicit inability object with a closed reason code:
   `INSUFFICIENT_EVIDENCE`, `AMBIGUOUS_EVIDENCE`, or
   `UNSUPPORTED_QUESTION`.

An explicit inability immediately ends that question with
`inability_to_answer`; it has no aggregate or agreement. An invalid model
shape or value immediately ends with `question_error` and
`INVALID_MODEL_OUTPUT`. A timeout ends with `MODEL_TIMEOUT`; an unavailable
backend ends with `MODEL_UNAVAILABLE`; a context overflow ends with
`CONTEXT_LIMIT_EXCEEDED`. Failed questions have no partial aggregate and never
drop failed attempts from the denominator.

## Results and Aggregation

```json
{
  "contract_version": "v1",
  "model": "qwen3:8b",
  "status": "completed",
  "results": {
    "department": {
      "type": "choice",
      "status": "answered",
      "answer": {
        "choice": "technical",
        "vote_share": {"billing": 0, "technical": 1}
      },
      "agreement": null,
      "requested_samples": 1,
      "error": null,
      "trace": {}
    }
  },
  "error": null
}
```

The `trace` member is shown empty for brevity; a real trace always carries the
required fields defined in [Trace and Replay](#trace-and-replay).

For a valid envelope, `status` is `completed` even when individual results
contain an inability or question error. Every object in a native response —
the response envelope, each result, each answer, each trace, and each error
object — is a closed object: unknown fields are errors. Every result is a
closed object with
these fields:

| Field | Rule |
| --- | --- |
| `type` | `choice`, `score`, or `noul` when the type parsed; otherwise null. |
| `status` | Exactly `answered`, `inability_to_answer`, or `question_error`. |
| `answer` | A type-specific answer object only for `answered`; otherwise null. |
| `agreement` | Number from 0 through 1 only for answered results with `sample_count > 1`; otherwise null. |
| `requested_samples` | Always the request's `inference.sample_count`, including results with zero attempts. |
| `error` | Null for `answered`; otherwise the error object defined above. |
| `trace` | The inline trace object defined below, including zero or more attempts. |

Choice `answer` is `{ "choice": string, "vote_share": map<string, number> }`.
Score `answer` is `{ "score": number, "legend": map<string, JSONContent>,
"vote_share": map<string, number> }`. Score map keys for both `legend` and
`vote_share` are the decimal strings
`"0"` through `"criteria.length - 1"`, matching the zero-based criterion
positions, and each `legend` value is the criterion `JSONContent` at that
position. Noul `answer` is `{ "noul": number }`.

`INVALID_QUESTION` is a `question_error` with the parsed type when available,
otherwise type null. `INVALID_MODEL_OUTPUT`, `MODEL_TIMEOUT`,
`MODEL_UNAVAILABLE`, and `CONTEXT_LIMIT_EXCEEDED` are also `question_error`.
`INSUFFICIENT_EVIDENCE`, `AMBIGUOUS_EVIDENCE`,
`UNSUPPORTED_QUESTION`, and `AGGREGATION_TIE` are
`inability_to_answer`. This distinction is fixed for native v1.

For `n = sample_count` valid samples:

| Type | Aggregate | Agreement |
| --- | --- | --- |
| Choice | `choice` is the unique option with the largest vote count. `vote_share[option] = count(option) / n` for every supplied option, including zero-vote options. A tie for the largest count is `inability_to_answer` with `AGGREGATION_TIE`. | `max(option count) / n`, or null when `n = 1`. |
| Score | `vote_share[String(level)] = count(level) / n` for every level. `score = sum(level * vote_share[String(level)])`, so it is in `[0, criteria.length - 1]` and may be fractional. | `max(level count) / n`, or null when `n = 1`. A tied mode does not invalidate a Score because its empirical expected position remains defined. |
| Noul | `noul = sum(sample values) / n`. For agreement only, `bucket(p) = min(floor(10 * p), 9)`: buckets 0 through 8 cover `[0.0, 0.9)` in 0.1 intervals and bucket 9 covers `[0.9, 1.0]`. | `max(bucket count) / n`, or null when `n = 1`. This is quantized consistency and can hide disagreement near bucket boundaries. |

Agreement is never named `confidence` in the native response. It is a
repeated-sample consistency statistic, not a correctness probability,
calibration claim, or Noul truth probability.

## Trace and Replay

`trace` is inline in every result. Native v1 does not persist traces on behalf
of callers; callers retain the returned record if they need durable audit
storage.

Each trace is a closed object with a required `trace_id`, a UUID generated by
the engine, and required `parent_trace_id`, which is null for an original
evaluation and the source trace ID for a replay. It includes:

- `trace_schema_version`, `contract_version`, prompt-template version,
  output-schema version, aggregation version, and policy version.
- The accepted request representation, canonical request hash, exact state,
  question, criteria, and policy provenance
  `caller-declared-unverified`.
- Rendered model messages, resolved inference settings, backend and model
  identity, model digest when available, and local runtime version.
- Every attempt in order: raw output, parsed value if any, validation outcome,
  timestamp, and terminal error or inability reason.
- The aggregate when one exists, plus the parent trace ID for a replay.

A replay request is a closed object with exactly these fields:

```json
{
  "contract_version": "v1",
  "trace": {}
}
```

`trace` is one complete inline result trace returned by a previous evaluation,
abbreviated as `{}` in the example above. It is self-contained: the engine reads the accepted request and resolved
configuration recorded in it rather than using a server-side trace ID or
storage lookup. The replay envelope is validated with the same structural
rules and codes as an evaluation envelope before any model call: malformed
JSON, unknown or missing envelope fields, and a `trace` value that is not a
closed object carrying the required `trace_id` and `parent_trace_id` are
structural rejections with a JSON Pointer path into the replay body, such as
`/trace/trace_id`.

`canonical_request_hash` is the lowercase hexadecimal SHA-256 digest of the
UTF-8 JSON Canonicalization Scheme (RFC 8785) representation of the accepted
request. Replay does not verify trace integrity: `canonical_request_hash` is a
correlation aid that lets a caller recompute and compare, not an integrity
proof, and the engine does not reject a trace whose recorded hash is
inconsistent with its recorded request. The trace retains the exact rendered
messages as well as the prompt
template, output schema, and aggregation artifact versions used to produce
them.

Replay submits the recorded input and resolved settings as a new evaluation and
creates a new trace linked to its parent. It must resolve and use the recorded
prompt-template, output-schema, and aggregation versions, not current
defaults. If the referenced model, backend capability, or required versioned
artifact is unavailable, replay returns a native response with status
`rejected`, empty `results`, and error code
`REPLAY_CONFIGURATION_UNAVAILABLE`; it must not silently substitute a current
equivalent. This is the only native v1 rejection that occurs after accepting a
replay request rather than while validating an evaluation envelope. A seed is
passed when the selected backend supports it, but replay does not promise equal
stochastic output.

## Jev Adapter

The adapter is intentionally narrower than the native contract.

It accepts the documented Jev-shaped input map on every supported access face.
The input map is a closed object with exactly the fields `state` (required
`JSONContent`), `model` (required nonempty string naming a configured local
model profile), and `questions` (required nonempty map of typed questions); it
has no `policy` or `inference` input because the adapter supplies those from
its own defaults. The adapter validates the input with the same structural
rules and codes as the native envelope before any conversion or model call. An
input that fails structural validation produces the adapter result shape with
`answers` and `local_judge` both null and `error` containing the structural
code with a JSON Pointer path into the raw Jev body;
`JEV_ADAPTER_UNMAPPABLE_RESULT` is never used for input validation.

Its result is a closed adapter object with three fields:

```json
{
  "answers": {},
  "local_judge": {
    "contract_version": "v1",
    "traces": {},
    "confidence_disclosure": "confidence is local repeated-sample agreement, not Jev or calibrated confidence"
  },
  "error": null
}
```

`answers` contains only the documented Jev-shaped answer map when every
requested question maps successfully; otherwise it is null. `local_judge` is a
documented extension, not a Jev field: `traces` is the map of native inline
result traces keyed by question ID, and the `local_judge` object carries the
mandatory semantics disclosure in
`confidence_disclosure`. A caller replays one trace through the normal native
replay
operation. `error` is null on a mapped result and otherwise contains the
native error object with `JEV_ADAPTER_UNMAPPABLE_RESULT`.

1. It accepts Jev-shaped input maps and converts them to native v1 with
   `policy.version = "jev-adapter-v1"`, `sample_count = 3`,
   `temperature = 0`, and `timeout_ms = 30000` unless a documented adapter
   profile says otherwise. A profile may not set `sample_count` below 2:
   agreement is null at one sample, and the adapter refuses results that lack
   agreement.
2. The Jev-shaped `model` field names a configured local model profile. Hosted
   Jev aliases such as `jev-latest` are rejected with
   `UNSUPPORTED_LOCAL_MODEL`; the adapter never forwards an evaluation to
   TypeSafe.
3. It accepts only native results whose every requested question is
   `answered`. Any inability, question error, unknown field, unsupported
   feature, or missing agreement becomes
   `JEV_ADAPTER_UNMAPPABLE_RESULT`; it never invents a Jev-shaped answer.
4. For Choice, native `vote_share` maps to Jev `probabilities`, `choice` maps
   directly, and `agreement` maps to Jev `confidence` with the required
   disclosure: `confidence is local repeated-sample agreement, not Jev or
   calibrated confidence`.
5. For Score, native `score` maps to Jev `score`, native `legend` maps to Jev
   `legend`, and native string-keyed level `vote_share` maps to Jev
   `probabilities`. `agreement` maps to `confidence` with the same disclosure.
   Jev Score results carry `score`, `probabilities`, `legend`, and
   `confidence`, so these mappings preserve the documented Jev field set.
6. For Noul, native mean `noul` maps directly. Noul has no Jev confidence
   field. The mapped `noul` is a local repeated-sample estimate of the same
   yes-probability Jev's Noul expresses, not a calibrated probability.

The adapter therefore preserves the Jev answer-map shape for this documented
subset while exposing required native provenance as an extension. It does not
claim parity in calibration, latency, parallelism, error behavior, or a
byte-for-byte Jev wire response.

## Error Code Registry

Every stable error code in the contract, with its class. HTTP status mappings
are defined by each access adapter, not here.

| Code | Class |
| --- | --- |
| `MALFORMED_JSON` | structural |
| `UNSUPPORTED_CONTRACT_VERSION` | structural |
| `UNKNOWN_FIELD` | structural |
| `MISSING_FIELD` | structural |
| `INVALID_FIELD` | structural |
| `INVALID_QUESTIONS_MAP` | structural |
| `DUPLICATE_QUESTION_ID` | structural |
| `REQUEST_TOO_LARGE` | structural |
| `UNSUPPORTED_INFERENCE_SETTING` | structural |
| `UNSUPPORTED_LOCAL_MODEL` | structural |
| `INVALID_QUESTION` | `question_error` |
| `INVALID_MODEL_OUTPUT` | `question_error` |
| `MODEL_TIMEOUT` | `question_error` |
| `MODEL_UNAVAILABLE` | `question_error` |
| `CONTEXT_LIMIT_EXCEEDED` | `question_error` |
| `INSUFFICIENT_EVIDENCE` | `inability_to_answer` |
| `AMBIGUOUS_EVIDENCE` | `inability_to_answer` |
| `UNSUPPORTED_QUESTION` | `inability_to_answer` |
| `AGGREGATION_TIE` | `inability_to_answer` |
| `REPLAY_CONFIGURATION_UNAVAILABLE` | replay — response-level rejection |
| `JEV_ADAPTER_UNMAPPABLE_RESULT` | adapter — result-mapping refusal |

## Acceptance Evidence

Stage 1 creates a versioned corpus. Each case records state, policy,
questions, exact menu or rubric, expected answer or allowed-answer set, model
profile, contract version, and evidence rationale. Each case also records
`inference` (optional, with the request inference shape and ranges),
`matched_case_id` (the ID of the paired base case; required non-null for
adversarial and metamorphic cases and null otherwise, and it must reference an
existing case in the same corpus version), and `metamorphic_relation` (one of
`json-key-reorder`, `question-map-reorder`, `irrelevant-evidence-insertion`,
`id-aligned-permutation`; required non-null for metamorphic cases and null
otherwise).

| Corpus | Required report |
| --- | --- |
| Deterministic fixtures | Pass rate for envelope validation, typed validation, aggregate equations, trace fields, replay configuration handling, and adapter refusal. |
| Normal cases | Labelled accuracy, answer coverage, agreement distribution, invalid-output rate, inability rate, and backend-error rate. |
| Ambiguous cases | Allowed-outcome coverage and inability behavior. Ambiguity does not imply low agreement or a fixed correct answer. |
| Adversarial cases | Task-preservation rate for labelled state-contained role spoofing, policy override text, fake schemas, candidate substitution, and confidence-redefinition attempts. Report the behavior; do not label it injection resistance without measured evidence. |
| Metamorphic cases | Invariance rate under JSON object-key reordering, question-map reordering, irrelevant-evidence insertion, and ID-aligned question permutations. Choice-option permutation is measured, not guaranteed. Score-level permutation is not an invariance because it changes the rubric. |

The report metrics are fixed. `Answer coverage` is answered cases over
submitted labelled cases. `Accuracy` is answered cases whose answer equals the
case's expected answer or falls within its allowed-answer set, over answered
cases. `Allowed-outcome coverage` is ambiguous cases whose answer falls within
its allowed set, over submitted ambiguous cases. `Task preservation` is
matched adversarial pairs whose adversarial answer is type-valid and within
the matched normal case's answer set, over pairs whose matched normal case was
answered, with the excluded count reported. `Invariance` is, for each declared
relation, metamorphic pairs in which every ID-aligned question is invariant
over pairs of that relation; a question is invariant when a Choice selects the
same option, a Score changes by at most 0.1, or a Noul changes by at most
0.05. The 0.1 and 0.05 tolerances are contract-chosen design constants, not
empirical findings.

The evidence gate is fixed for each tested model profile and corpus version:

1. Every deterministic fixture must pass.
2. At least 50 labelled normal cases must produce answer coverage of at least
   90% and accuracy of at least 80% among answered cases. Both figures and the
   denominator must be reported.
3. At least 20 ambiguous cases must reach at least 80% allowed-outcome
   coverage. An answer outside the case's allowed set is not rescued by high
   agreement.
4. At least 20 matched adversarial cases must report task-preservation no more
   than 10 percentage points below their matched normal cases.
5. At least 20 metamorphic pairs must report at least 80% invariance for each
   declared relation.

Failing this gate does not prove a security flaw or calibration failure. It
means Stage 1 cannot claim demonstrated usefulness for that model profile and
the contract, prompt template, corpus, or intended consumer gate must be
revisited before a workflow relies on it.

An implementation issue is not ready until it cites this contract and owns
fixed fixtures. No issue may decide a contract field, aggregation rule, prompt
boundary, trace field, adapter conversion, or evidence metric during
implementation.
