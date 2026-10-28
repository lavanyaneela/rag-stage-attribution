# Security

## Reporting

Report suspected vulnerabilities by opening a GitHub security advisory on
this repository rather than a public issue. Expect an acknowledgement
within a week.

## Scope and posture

This repository is an offline measurement harness. The default path makes
no network calls, reads no credentials, and executes nothing it downloads:

- The only network touching code is the embedding capture tool, behind the
  `capture` extra, run manually, pinned to one model repository, with the
  downloaded model's hash recorded in the committed manifest.
- The live adapter is behind the `live` extra and three environment
  variables, is never imported on the default path (a test enforces this
  with the dependency made unimportable), and sends only corpus text and
  golden questions to the endpoint the operator configured.
- Fixtures are data, not code: the loaders validate structure and refuse
  contradictions, and nothing in a fixture is ever executed or templated
  into a shell.
- `stageattr` reads and writes JSON files at paths the operator passes. It
  does not fetch URLs, spawn shells, or evaluate expressions from input.

## Dependency policy

Dependencies are the measurement stack (numpy, scipy, faiss-cpu, langgraph,
langchain-core) plus dev tooling. Dependabot keeps them current; a bump
that changes a published figure fails `make verify` by design, which is the
intended behaviour, not an inconvenience to suppress.
