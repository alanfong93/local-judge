# Who
Alan's automation estate — n8n workflows, AI agents (OpenCode), and Python projects (ops-guard) that need structured judgments without sending their data to an external inference service.

# What
A local, consumer-independent decision engine that answers typed questions about supplied state using Ollama.

# Problem
Workflows, agents, and Python applications need structured judgments without maintaining separate judgment logic for each consumer or sending the judged data to a paid external service.

# How
Accept state, instructions, criteria, and typed questions; obtain judgments from local models; enforce each question type's output constraints; and return structured results with honestly labelled uncertainty information.

# Required capabilities and constraints

- Support **Choice, Score, and Noul as distinct question types**, preserving their respective semantics and validity constraints: Choice selects one option from a defined set, Score places the state on an ordered rubric, and Noul estimates whether a statement is true. Menu enforcement applies where the question supplies a menu.
- Offer a documented Jev-compatible subset, with unsupported features and semantic differences explicit. Compatibility is limited to that subset.
- Be accessible through HTTP, MCP, and an importable Python library.
- Keep inference and request data local, without per-request hosted inference API charges.
- Report repeated-sample agreement as **agreement**, not calibrated confidence or probability of correctness. Noul's estimated probability that a statement is true is distinct from both agreement and the probability that an answer is correct. Any compatibility-field mapping must disclose those distinctions.
- Reject invalid outputs or report inability to answer rather than inventing a valid-looking judgment. Typed validity is not evidence that an answer is correct.
- Make each judgment traceable to its supplied state, typed question, instructions, criteria, exact candidate menu where applicable, model identity, relevant inference settings, and result or failure. Preserve applicable rubric, schema, and threshold versions for replay; replay does not promise identical stochastic results.

# Done when

Choice, Score, and Noul work according to their documented semantics through all three access routes; supported Jev-subset requests behave as documented; invalid outputs cannot escape validation; local operation, failure reporting, agreement labelling, and traceability are demonstrated; and representative normal, ambiguous, and adversarial judgments have evidence of usefulness beyond response-shape compliance.

# Not this project

General-purpose text generation. Model training or fine-tuning. Full Jev parity. Hosted-model speed or calibration parity. A hosted service. Multi-tenancy.
