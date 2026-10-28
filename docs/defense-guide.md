# Defense guide: rag-stage-attribution

For the author's preparation before interviews. Read the README first; this
document explains how to own every decision in it, what each number means,
where the sharp questions land, and the honest answer at each edge. Figures
here use the same receipt anchors as the README, so this guide can never
drift from the measurements without failing the build.

## The one-breath summary

A wrong RAG answer cannot tell you whether retrieval failed or generation
ignored what retrieval delivered; both look identical at the output. I
built a pipeline whose failures I inject, so ground truth exists, and
measured how well a trace based attribution method recovers the true
failing stage. It recovers the cause in 99.5% of cases; the finding is
where it fails, which is the tail of the context window, and what happens
when you take the trace away, which is that accuracy falls to 61.9% and
every retrieval side failure becomes invisible.

## Why each piece exists

**Why injection instead of a real model.** Scoring attribution needs the
true cause, and real models do not volunteer the true cause of their
failures. Injection creates it. The retrieval side is real: real MiniLM
embeddings over a committed corpus, so recall numbers are genuine
properties of an embedding space, not simulations. What I give up is any
claim about how a specific model behaves; ADR-001 states that boundary and
the flag gated adapter that would let someone with an Azure OpenAI
deployment measure their own failure mix.

**Why LangGraph.** Not novelty. Attribution consumes stage boundaries, and
a graph node is a boundary with a name that leaves a record. The ablation
is my proof that the machinery earns its place: deny the attributor the
trace and it places none of the 360 retrieval-side failures. The cost is
admitted and priced: 4 trace records per question, a median of 1688 bytes
per question, and a wall time ratio captured once and asserted by a test
rather than quoted, because durations are facts about machines.

**Why a FAISS flat index.** Exact search removes an approximation confound.
When retrieval misses here, the embedding space missed, not an ANN
heuristic. At 112 paragraph chunks, speed is irrelevant.

**Why the corpus is hand built and hostile.** The first corpus was too easy
and retrieval never failed, which made half the experiments vacuous. The
committed one has distractor pairs that share vocabulary and disagree on
payloads, six pure distractor documents, and 73 golden questions rewritten
into user voice so query vocabulary drifts the way real questions do.

**Why the golden label is a span, not a chunk id.** Chunk ids are artifacts
of the chunking strategy under test. A span survives every chunker, and
golden chunks are recomputed per strategy. E4 is only possible because of
this choice.

## The numbers, and how each was measured

Every figure is re-derived by `make verify` from committed experiment
outputs; `make experiments` re-runs the sweeps themselves, 30 seeds over
all 73 questions per arm.

- **Headline, E1.** Recovers the cause in 99.5% of 1172 injected failures,
  default configuration. Per class: all 360 retrieval-side failures read
  directly off the trace; every error the method makes lives on the
  generation side.
- **The ablation, E1.** 61.9% with the trace withheld, and it marks 101
  failures as unattributable. This is the opening impossibility expressed
  as a number: without instrumentation, never-retrieved and ignored
  present identically.
- **The confusable class, E2.** Golden chunk in the context tail, ignored
  by the generator: 82.4% with the contract on, and it drops to 52.9%
  with the contract off, over the 34 tail-ignored failures, exact 95%
  interval from 35.1% up to 70.2%. Widening the tail definition grows the
  class to 46 cases and moves accuracy little: the blind spot tracks the
  definition, which is the point of publishing the sensitivity arms.
- **Decoupling, E3.** Recall sat at 83.6% at baseline; the wider window
  lifts recall at context to 90.4%, and those 6.8 points of recall bought
  3.5 points end to end, and generation's share of the remaining failures rises to
  80.8%. This is the slide for "why did fixing the retriever not fix the
  product".
- **Chunking, E4.** Paragraph chunking carries 10.2% of blame on retrieval
  misses; the fixed window strategy moves it to 8.4% by retrieving easier
  and burying payloads deeper. Chunking decides which stage looks broken.

## Release history, and why each version exists

- v0.1.0, core: corpus, fixtures, chunkers, embeddings with drift checks,
  the LangGraph pipeline, the stand-in, the attributor, the CLI. The point
  where clone-run-verify first passed.
- v0.2.0, measurement: the five experiments with committed outputs, and
  the receipts mechanism with the prove-the-checker rule. The point where
  the documents stopped being able to lie.
- v0.3.0, evidence: README, ADRs, failure modes, rendered diagram,
  captured demo with a manifest of real exit codes, CI with the
  reproducibility gates, community files.
- The delivery tag is cut only after the forensics passes complete, with
  the audit findings fixed in dedicated commits whose messages carry the
  finding, so the audit trail is the log itself.

## How the demo's claims were measured

| Claim in the demo | Where it comes from |
| --- | --- |
| verdict names a stage with intervals | `stageattr verdict` on a real run file; Clopper-Pearson bounds from SciPy |
| the refusal exits 2 | a run with too few attributable failures; the manifest records the exit code as captured |
| per failure reasons | the attributor's reason strings, printed by `stageattr attribute` |
| the checker catches a broken figure | `tools/prove_checker.py` substitutes a wrong value and must see the failure |

The GIF is a replay of captured stdout; `docs/evidence/demo-manifest.json`
lists every command with the exit code it really returned.

## Questions an interviewer will ask, and the answers

**"It recovers the cause in 99.5% of cases. Is the problem real?"** The headline is propped
up by the trace: retrieval blame is elimination, not inference. The real
problem lives in the two honest numbers next to it: 52.9% with the
contract off on the confusable class, and 61.9% with the trace withheld.
I lead with those because the method's value is exactly the gap between
them and the headline.

**"The stand-in is yours. Did you tune it to be attributable?"** The
opposite direction is documented: the first parametric answers were too
easy to detect lexically, so I rebuilt them as paraphrases of the policy
sentence with the wrong payload, which is what a model misremembering a
policy sounds like, and accuracy on the tail class fell to a coin flip
without the contract. The hardening is in the log as named fix commits
and in the difficulty tests that pin what the redesign bought; the build
itself was a continuous sprint, which the log's timestamps show honestly.

**"Why should I trust the citation contract as a mitigation?"** Only as
far as ADR-005 trusts it: it converts most tail-ignored failures into
attributable ones, and it is an artifact the faulty generator controls. An
adversarial generator that always cites what it ignored collapses it. The
honest fix is a signal the generator does not emit, attention or logit
level, which is outside the offline boundary and named as future work.

**"What breaks if I point this at my pipeline?"** Three things, stated in
the README's cannot-measure section: you have no injected ground truth, so
you get attributions without a score for them; my support inference is
lexical and a strong abstractive model defeats it; and my thresholds were
fit to this corpus. What transfers is the architecture: trace at node
boundaries, attribution as a consumer of the trace, refusal as a first
class verdict.

**"Why does the tool refuse instead of guessing?"** Because exit 1 and
exit 2 are different answers. "Both stages contribute" is a finding with
intervals behind it. "The evidence cannot carry a verdict" is a
confession, and a tool that folds the two together trains people to
ignore it. The refusal thresholds are config, not folklore.

**"What was the hardest bug?"** The quiet ones in the measurement itself.
The scorer glued a sentence closing period onto the payload and failed
faithful answers; the experiment cache keyed on a partial config and
silently reran a baseline in a sensitivity arm, and the tell was two
output rows that were byte identical. Both are pinned by tests named after
them, and both are why the receipts exist: the code was fine and the
sentences were about to be false.

## Edges to concede without being asked

- The corpus is one domain and authored by me; the support floors are fit
  to it.
- The confusable class sample is small; that is why it ships with an exact
  interval and why the zero-adjacent numbers are priced, not rounded.
- Nothing here measures a real model, on purpose, and the adapter is the
  invitation for someone who can.
