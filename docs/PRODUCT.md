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
- Report repeated-sample agreement as **agreement**, not calibrated confidence or probability of correctness. Any compatibility-field mapping must disclose that distinction.
- Reject invalid outputs or report inability to answer rather than inventing a valid-looking judgment. Typed validity is not evidence that an answer is correct.

# Done when

Choice, Score, and Noul work according to their documented semantics through all three access routes; supported Jev-subset requests behave as documented; invalid outputs cannot escape validation; local operation, failure reporting, and agreement labelling are demonstrated; and representative judgments have evidence of usefulness beyond response-shape compliance.

# Not this project

General-purpose text generation. Model training or fine-tuning. Full Jev parity. Hosted-model speed or calibration parity. A hosted service. Multi-tenancy.
