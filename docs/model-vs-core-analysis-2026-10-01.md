# Isolating model, serving interface, and core design

Analysis and diagnostic probes by OpenCode/gpt-6.1-sol, 2026-10-01. These selected-case diagnostics are not the acceptance gate and do not replace the Jev comparison.

## New observations

Two false Noul cases from the pilot were selected: a customer asks to fix a bug and explicitly does not want a refund; another asks how long refunds take but says they are only asking about policy. Expected semantic result is near zero in both.

| Direct Gemma condition | Fix / no refund | Refund-policy inquiry |
| --- | --- | --- |
| Exact prompt-2 and existing schema, Ollama `/v1/chat/completions` | Insufficient evidence | Insufficient evidence |
| Exact prompt-2 and existing schema, Ollama `/api/chat` | Insufficient evidence | Insufficient evidence |
| Simplified messages, same scalar-or-inability schema | Insufficient evidence | Insufficient evidence |
| Same simplified messages, no schema constraint | Zero inside Markdown code fence | `{"json": 0}` |
| Separate simplified prompt, numeric-only schema | `0` | `0` |
| Separate simplified prompt, object value-or-inability schema | `{"value": 0}` | `{"value": 1}` (wrong) |

Each condition was observed once, at temperature zero. The numeric-only and object-union conditions also changed prompt wording; they are not isolated schema-only interventions. Both Ollama endpoints can share the same constrained decoder. Criteria were retained in the first four conditions, but not in the final two short diagnostic prompts.

## Interpretation

The downstream local-judge core is not necessary to reproduce these refusals: they occur in direct model-server calls before local validation or aggregation. Unconstrained output shows the model can express a semantically correct answer on these cases, albeit in invalid wire shapes. This narrows investigation to the prompt, serving implementation, constrained output format, and model's behavior under those conditions. It does not identify the model weights alone as the cause.

The object-union experiment is not a demonstrated improvement: reducing refusal produced an incorrect affirmative answer on one negative case. The numeric-only experiment removes the inability path entirely and must remain diagnostic. Do not improve apparent coverage by forcing answers when abstention is warranted.

There is no evidence from these probes that HTTP/MCP/library access or deterministic validation needs redesign. Prompt composition, model-port schemas, and the choice to fail an aggregate when any sample abstains are also design decisions and must be assessed separately from transport correctness.

## Bounded next experiment

1. **Core control:** inject frozen raw responses through a scripted model port (correct answer, out-of-range output, malformed JSON, each inability code, transport error). Confirm native validation and aggregation against the documented contract, including mixed answered/inability samples. Extend existing meaningful tests only where coverage is missing.
2. **Interface comparison:** Gemma and Grok × current and simplified prompts × scalar-or-inability and object-or-inability output formats. Hold question, criteria, evidence, sample count, and inability policy fixed. Across formats change only necessary output-shape instructions; describe that arm as a format-plus-instruction change. Provider differences mean the comparison is of model/serving stacks, not weights alone.
3. **Evaluation discipline:** freeze a labelled mixed-case set before running; include positives, negatives, genuinely unclear inputs, misleading instructions, and relevant languages. Repeat each condition, vary condition order, and retain exact requests, raw outputs, model/configuration identity, and latency. Repeats assess variability and are not new independent cases.
4. **Decision:** judge wrong answered results, unnecessary abstentions, output validity, coverage and latency separately. Choose at most one candidate and test on untouched real-workflow data. Jev remains a separate native-service comparator because its API does not expose the same prompt/decoder controls.

Illustrative budget: 24 cases × 8 conditions × 3 repeats = 576 attempts. This is a proposed budget, not an experiment executed here, not a production-readiness gate, and not permission to run unlimited paid calls. Use a small paired subset for native/OpenAI endpoint parity instead of doubling that budget.

**Current conclusion:** model-serving-interface behavior is implicated; a core rewrite and a model-only attribution are both premature. Given Jev's measured pilot advantage, any further local diagnostic should be bounded rather than an open-ended prompt-tuning effort.

## Independent review: Astra and Grok 4.7

Both reviewers received the reported results, relevant code excerpts, and proposed experiment; these were reviews of supplied evidence, not independent reproductions. Astra was consulted through a read-only subagent with inline evidence after its initial attempt could not access the filesystem. Grok 4.7 reviewed through OpenRouter. Recorded by OpenCode/gpt-6.1-sol.

**Agreement:** neither model-only attribution nor a core rewrite is justified; selected cases, single observations, prompt/schema changes, shared endpoint decoding, and incomplete historical result retention limit causal inference. A format that suppresses refusal can increase wrong answers. Both recommend a staged experiment rather than spending 576 calls up front.

**Difference:** Astra accepts that direct calls reproduce the selected failures without downstream parsing/aggregation, while Grok wants captured outbound request parity before granting that inference. The probes regenerated messages using the current compiler rather than capturing actual outbound HTTP bytes, so the stronger parity claim remains untested. Grok prefers a targeted schema/criteria diagnostic first; Astra prefers screening all eight configurations on a smaller frozen subset. These are competing experimental sequences, not independent verification of the same result.

Astra also identifies a separate deterministic discrepancy: `SamplingOrchestrator.run_question` makes the whole requested sample loop before `executor.run` classifies inability. That differs from the contract's immediate termination language. It affects sample count, latency, and cost; it cannot explain these single-sample direct refusals. Verify expected call counts with a scripted port before changing this behavior.

Revised proposed order: (1) capture and replay actual outbound payloads, and stub-test validation, aggregation, and call counts; (2) keep Gemma fixed and vary one diagnostic treatment at a time with complete criteria retained; removing criteria or removing abstention are explicitly diagnostic controls, never proposed production improvements; (3) compare at most two retained candidates with Grok on untouched labelled cases. Preserve every run and score unnecessary refusal, wrong answers, and invalid wire separately. The reviewer proposals differ on the next screening design; no further paid experiment was executed as part of this review.

## Executed follow-up: captured requests and one-factor controls

The authorized follow-up used `docs/eval/isolate_core.py`; exact requests and raw outputs are retained in `docs/eval/core-isolation-20261001T080329Z.results.json`. This was a host Python deployment-runtime test using the actual endpoint adapter and `UrllibHttpTransport`, with an injected opener that records urllib's serialized request body and headers before dispatch. It was not a network packet capture or a capture from the running Docker process. The capturing opener differs from the default opener in redirect handling; no redirects were observed or investigated. Captured headers contain only `Content-type`, no authentication or cookies.

Configuration: `gemma4:e4b` digest `c6eb396dbd5992bbe3f5cdb947e8bbc0ee413d7c17e2beaae69f5d569cf982eb`, Ollama `0.34.4`, local endpoint `http://127.0.0.1:11434/v1/chat/completions`, temperature 0, one sample, 120000 ms timeout, prompt-2/schema-1. No seed was supplied. Model/server identity was read after the experiment, not independently pinned before every call.

Four cases: the two false-refund cases above, an explicit true refund request, and an empty-ticket evidence control. The full question and criteria were retained. All four live runtime attempts returned `INSUFFICIENT_EVIDENCE`; independently replaying each captured request with the **identical serialized body and captured headers** returned the same reason. Thus these failures were reproduced outside parsing/aggregation without reconstructing their payloads.

Four treatment arms were randomized with a fixed shuffle seed; each case/arm was called twice:

| Arm | Clear cases (3 cases × 2 repeats) | Empty evidence (2 repeats) |
| --- | --- | --- |
| Current messages and scalar-or-inability schema | 6 unnecessary insufficient-evidence refusals | 2 insufficient-evidence refusals |
| Simplified messages only; same question, criteria, state and schema | 6 ambiguous-evidence refusals | 2 ambiguous-evidence refusals |
| Remove schema only; original messages unchanged | 4 invalid outputs and 2 insufficient-evidence refusals | 2 valid insufficient-evidence refusals |
| Number-only schema; original messages unchanged | 6 expected numeric answers | 2 numeric zeros; inability cannot be expressed |

The number-only intervention changes only the schema. It demonstrates that schema selection affects Gemma's observed output in this serving configuration. It does **not** identify whether that comes from model abstention preference, the grammar/constrained decoder, or their interaction. It cannot be promoted as a solution because it removes the contract's inability alternative. The empty-case zero is not evidence of reliable uncertainty handling. Object-union schemas were not part of this controlled run.

Total: **40 local model calls** (4 captures + 4 exact replays + 32 ablation attempts), four selected cases, not 40 independent examples. No Grok benchmark calls were made: neither tested alternative preserving the inability path yielded a viable candidate to advance.

Ten fixed-response core controls passed: valid numbers, out-of-range number, malformed JSON, boolean, all three inability reasons, timeout and unavailable transport. The eleventh control confirmed the separate termination discrepancy: first-sample inability followed by two numeric outputs consumes **three calls** and produces inability. The contract's immediate-termination expectation would consume one. This is a real core orchestration issue to fix separately, but not the cause of the single-sample failures. The existing suite still passed **281 tests**, with its pre-existing cleanup warning; that suite did not flag the call-count discrepancy.

**Conclusion:** the current Noul schema/model-serving interaction is a reproducible bottleneck. The semantic capability is present on these examples when the allowable output set changes, but no tested contract-preserving alternative solved it. Keep the typed core rather than rewrite it, track the early-termination issue separately, and stop this diagnostic budget here. Astra's completion review agreed with that bounded stop and cautioned against claiming the candidate space is exhausted. Jev remains the stronger measured option for the next actual-workflow trial.

## Early-termination correction

OpenCode/gpt-6.1-sol subsequently corrected the orchestrator: it now stops a question's sample loop after classification records any terminal error or inability. Earlier attempts remain in the trace, the result retains the originally requested sample count, failed questions produce no partial aggregate, and valid sibling questions continue. Regression tests cover all three inability reasons, invalid model output, prior-attempt retention, and sibling isolation. This changes call count and latency, not the interpretation of an inability or the model's judgment quality. The retained isolation artifact above records the pre-fix behavior; its diagnostic script now asserts one call for that control.
