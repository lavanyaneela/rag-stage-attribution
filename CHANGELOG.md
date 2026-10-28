# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog, and versions are annotated git tags.

## [Unreleased]

## [0.3.0] - 2026-10-28

### Added

- README in the post-incident register, five ADRs, and the failure mode
  catalogue, every figure wired to a receipt anchor.
- Evidence pipeline: architecture SVG rendered from committed Mermaid
  source with intrinsic dimensions, a six segment terminal demo captured
  with real exit codes and replayed to a GIF, and a checker proof that
  breaks a figure on purpose.
- CI: three Python matrix with a 95% coverage floor, collected-equals-
  published test count gate, receipts re-measurement with a diff gate,
  experiment byte reproducibility check, and the awkward tables in the job
  summary. Community files and the handle and badge filler.

## [0.2.0] - 2026-10-28

### Added

- Experiments E1 to E5 with committed deterministic outputs: attribution
  accuracy and the trace ablation, the confusable class with exact
  intervals and sensitivity arms, retriever decoupling, chunking blame,
  and trace cost with flag gated overhead capture.
- The receipts mechanism: metric collection from machine readable sources
  only, and the anchored number checker with a reverse pass, duration
  sweep, exemption hygiene, and the prove-the-checker rule.

### Fixed

- The scorer stripped edge periods after a payload closing a sentence
  tokenised as its own stale form and a faithful answer scored wrong.
- The experiment resource cache keyed on a partial config and silently
  reran E2's baseline tail definition in the sensitivity arms; it now
  keys on the whole frozen config.

## [0.1.0] - 2026-10-28

### Added

- Committed handbook corpus (22 documents), 73 user-voice golden questions, and
  a parametric wrong-belief bank, with read-path consistency checks.
- Three chunking strategies with span-derived golden chunk resolution.
- Captured MiniLM embedding fixture (every strategy, every query) with a
  manifest that pins the model, the input digests, and the rewrite table.
- LangGraph pipeline (rewrite, retrieve, rerank, generate) with one trace
  record per node boundary; exact FAISS retrieval; lexical blend rerank.
- Seeded stand-in generator with injected failure modes and a citation
  contract; flag-gated Azure OpenAI live adapter shape.
- Trace-based attributor with the ADR-004 tail rule, mechanical ground truth
  labelling, Clopper-Pearson verdict statistics, and the stageattr CLI with
  exit codes 0/1/2/3.
