# Who

Alan's automation estate — n8n workflows, AI agents (OpenCode), and Python projects (ops-guard) that need fast, cheap, structured judgments and will not send data to a paid external API.

# Must be able to

1. Send a state plus typed questions (`choice` / `score` / `noul`, with instructions and criteria) and get back typed answers, probabilities, and confidence — **wire-compatible with the public Jev API shape**, so Jev-targeted tooling works by changing the base URL
2. Run entirely local (Ollama) — no data leaves the machine, zero marginal cost
3. Be reached three ways: HTTP (n8n workflows), MCP (AI agents), importable library (Python projects)
4. Refuse answers outside the supplied menu — enforced in code, not by prompting
5. Report honest confidence: agreement across repeated samples, documented as pseudo-calibration rather than trained calibration

# Done when

A request written from the public Jev API docs, pointed at local-judge, returns a correctly-shaped response from a local model; returning an answer outside the supplied menu is impossible by construction; and the first real caller (Article Saver tag selection over the closed tag menu) works end-to-end through it.

# Not this project

Text generation of any kind. Model training or fine-tuning. Speed or calibration parity with hosted decision models. A hosted service. Multi-tenancy.
