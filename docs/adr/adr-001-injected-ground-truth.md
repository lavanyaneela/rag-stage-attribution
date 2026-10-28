# ADR-001: Injected ground truth rather than a real LLM

Status: accepted.

## Context

The repository measures an attribution method: given a wrong answer from a
retrieval augmented pipeline, name the stage that failed. Scoring an
attribution requires knowing the true cause, and no real model volunteers
the true cause of its failures. A real LLM also cannot run offline, and the
offline rule is what lets a reviewer re-derive every published number from a
fresh clone with no keys.

## Decision

The generator is a seeded stand-in driven by an explicit failure mode per
question: answer faithfully, answer from a committed bank of wrong beliefs
while the right chunk sits in the context, answer from the wrong chunk, or
abstain. Injection gives ground truth; the retrieval side is measured for
real against committed MiniLM embeddings, so recall figures are genuine
properties of a real embedding space over this corpus.

The separation of knowledge is the load bearing wall. The stand-in may see
the golden span, because it simulates the model. The attributor may see the
trace, the answer, the citations, and the golden label, and may never see
the injected mode or the belief bank, because the mode is the answer key it
is graded against. The wall is enforced structurally: the trace schema has
no field for the mode, a test asserts the serialized trace never contains
it, and a source inspection test asserts the attribute call inside the
runner references no mode ledger.

## What is claimed, and what is not

Claimed: how often this attribution method recovers the true failing stage
on failures whose cause is known, and which class of failure defeats it.
Not claimed: anything about how any particular model behaves, how often any
model ignores context in the wild, or that these failure rates resemble any
production system's.

## What a live model would add

An Azure OpenAI shaped adapter sits behind the live extra and three
environment variables, honours the same interface, and is never imported on
the default path (a test runs the whole measured surface with the openai
package made unimportable). Running it would measure one real model's
failure mix on this corpus, which is a different and complementary number.
It cannot add ground truth: with a real model, "retrieved and ignored" goes
back to being an inference, which is the problem this repository measures a
method against, not a problem it can define away.
