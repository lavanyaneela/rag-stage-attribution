# ADR-002: LangGraph over a plain function chain

Status: accepted.

## Context

The pipeline is four steps: rewrite, retrieve, rerank, generate. Four
function calls would compute the same answers with less machinery, and a
LangChain runnable chain would too. The temptation to pick a graph library
for novelty is real and is exactly the kind of decision a reviewer probes.

## Decision

LangGraph, for one reason that survives the probing: attribution consumes
the boundaries between stages, and a graph node is a boundary with a name.
Each node appends exactly one trace record describing what crossed it: the
query before and after rewrite, the retriever's ranked output, the rerank's
kept and cut lists, the context the generator was shown, and the answer with
its citations. The attributor reads nothing else. In a plain chain those
boundaries exist only as local variables inside whoever composed the calls,
and instrumenting them means inventing ad hoc seams; the graph makes the
seams the architecture.

## The cost, admitted

This is more machinery than a pipeline needs if nobody wants attribution.
E5 prices the part of the cost that is machine independent: 4 trace
records per question and a median of 1688 bytes per question. The wall time overhead of the graph
plus trace against bare function calls was captured once on the capture
machine, lives in docs/experiments/e5_overhead.json as a ratio, and is
asserted by a test rather than quoted in prose, because an absolute
duration is a fact about a runner. What the machinery buys is E1's ablation
delta: deny the attributor the trace and retrieval side blame becomes
unreadable, which is the opening impossibility measured.

## Alternatives rejected

Callbacks and tracing hooks on a plain chain: reproduces the graph's trace
with more framework surface and without the explicit edge list that makes
the pipeline's shape reviewable in one screen. Logging statements: a trace
that only exists in a log stream cannot be handed to an attributor as a
value, and parsing logs back into structure is the bug factory this
repository declines to build.
