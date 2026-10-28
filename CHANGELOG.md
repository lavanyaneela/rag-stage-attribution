# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog, and versions are annotated git tags.

## [Unreleased]

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
