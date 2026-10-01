# Jev API vs local-judge: decision pilot

Tested 2026-10-01 by OpenCode/gpt-6.1-sol. Purpose: decide where to invest further effort, not validate local-judge by treating Jev as ground truth.

## Results

The same 19 synthetic, developer-labelled cases were sent to TypeSafe `jev-1.13.0` and the running local-judge container with `gemma4:e4b`, `prompt-2`, one sample and temperature 0. Cases and expected answers were fixed before either backend's responses were inspected. Jev's native API does not expose the identical sampling settings; this compares the products as used, not controlled identical inference internals.

| Measure | Jev API | local-judge / Gemma |
| --- | --- | --- |
| Expected answers returned | 19/19 | 15/19 |
| Choice, including no-issue `other` | 6/6 | 6/6 |
| Score | 5/5 | 5/5 |
| Noul, explicit money-back requests | 5/5 | 1/5 |
| Misleading-instruction Choice cases | 3/3 | 3/3 |
| Unnecessary inabilities | 0 | 4 |
| Incorrect answered judgments | 0 | 0 |
| Median observed request duration | 0.360 seconds | 2.255 seconds |
| Observed duration range | 0.258–0.486 seconds | 1.427–3.815 seconds |

The four local failures were model-produced `INSUFFICIENT_EVIDENCE` objects on clear Noul inputs, not malformed JSON or transport failures. Jev Noul values were 0.98, 0.04, 0.09, 0.99, and 0.03. The pilot's fixed scoring rules require true Nouls >=0.8, false Nouls <=0.2, and Scores within 0.25 of their labelled level. These are pilot thresholds, not calibration evidence.

The scored Jev run reported **6,611 input tokens and 527 output tokens**. At the documented price of $0.042 per million input tokens, with output free, estimated inference cost is **$0.000277662**. This is a calculation, not a verified invoice; it excludes setup probes, Cloudflare charges, and local electricity/hardware costs. Pricing source: <https://docs.typesafe.ai/models.md>.

## Reproduction and artifacts

- Script: `docs/eval/compare_jev.py`; requires process environment `LLM_GATEWAY_TOKEN` and the running local container on port 8000.
- Retained raw scored responses, request states, labels, timings, and token usage: `docs/eval/jev-comparison-pilot-1.results.json`. No credentials are included.
- Jev was called through `https://llm.alanfong.uk/jev/v1/systemone`, not local-judge's Jev-shaped compatibility adapter.
- Gateway initially blocked Python's default user agent with HTTP 403. An explicit `local-judge-evaluation/1.0` user agent succeeded. Those initial remote access failures are not scored as Jev inference failures.
- In the initial local run, all five Noul cases declined (14/19 overall); in the scored repeat one answered (15/19). The first raw run was overwritten by the script; its aggregate console output remains session evidence, not an independently retained result artifact.

## Gateway finding and correction

The live `llm-gateway` Worker already stored `JEV_API_KEY` and `MODAL_API_KEY`, along with the four other listed secrets. Its deployed code only routed Explabs, Bitdeer and APMIX. Two entries were added in `D:\projects\llm-gateway\index.js`: the Jev upstream `https://api.typesafe.ai/v1` and its `JEV_API_KEY` binding. Deployment used Cloudflare's `keep_bindings: ["secret_text"]`; all six secret names were confirmed retained, without reading their values.

After deployment, authenticated Jev model listing and actual evaluations returned 200; unauthenticated Jev model listing returned 401. Explabs and APMIX model listing returned 200. Bitdeer returned 401, which was not investigated and cannot be attributed to this deployment without a pre-change baseline. MODAL has a stored key but no route was added. Existing code forwards requests without automatic retries.

## Recommendation

**Prioritize the Jev API for a bounded trial on one actual workflow, and pause further open-ended local prompt tuning.** Jev showed a clear answer-coverage and latency advantage on this set. This is enough to choose the next experiment, not enough to replace every consumer or abandon local-judge.

The unanswered local results are safe refusals, not wrong decisions, but would interrupt useful automation. Local-judge remains valuable if data locality and independence from a hosted API are requirements. It remains running on `127.0.0.1:8000`; this comparison did not switch it to a remote backend or change its inference code.

Limits: one call per case per backend, a small synthetic corpus, five Noul examples sharing one refund question, no real workflow data, no calibration study, no full acceptance-gate run, and no proof of broad injection resistance. Next compare both in shadow mode on pre-labelled real-workflow inputs, including genuinely ambiguous cases, and measure incorrect automatic decisions separately from abstentions and backend failures. TypeSafe's known limitations remain relevant: <https://docs.typesafe.ai/model-jaggedness/jev-1.13.md>.
