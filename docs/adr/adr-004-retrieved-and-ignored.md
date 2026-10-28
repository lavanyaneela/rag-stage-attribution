# ADR-004: What counts as "retrieved and ignored"

Status: accepted. This is the definitional document the headline hangs on.

## Context

The attribution classes are retrieval_miss, retrieval_rank, and generation.
Two of the three are read directly off the trace. The third requires a
definition, because "the generator ignored a provided chunk" presupposes an
answer to three questions: what counts as provided, what counts as used,
and who owns a failure when both stages misbehaved at once.

## Definitions

Provided. A chunk is provided when it sits in the context window the
generate node was shown, with one carve out: the last tail_slots positions
of the window (one slot by default) are the tail, where presence is weak
evidence of provision. The carve out is the honest encoding of the fact
that a chunk at the context's edge and a chunk one rank past the edge are
nearly the same event from the generator's side, and E2 exists because this
line is a choice: widen the tail to two slots and the blind spot grows with
it, which the sensitivity arms measure.

Used. The attributor infers use lexically: the share of the answer's
content words found in a chunk, with three named floors from the config. An
answer unsupported by every provided chunk reads as fabrication; an answer
better supported by a non golden chunk than the golden one, past a margin,
reads as drawing on the wrong source; a citation to a non golden chunk is
generation evidence whether or not the answer supports it, because a
citation the answer does not draw on is itself a generator fault. Partial
use is the hard case: the stand-in sometimes cites the golden chunk it
ignored, and a golden citation on a wrong answer neutralises the citation
signal rather than deciding either way.

Ownership under compound failure. Upstream dominates. If the golden chunk
never reached the generator, the failure is retrieval's whatever the
generator was injected to do, because fixing the generator could not have
fixed this answer. Ground truth encodes the same rule, so the attributor
and the labeller cannot disagree about ownership by construction, only
about facts.

## Consequence

In a non tail slot, the trace plus elimination makes generation blame
airtight: retrieval demonstrably delivered, the answer is wrong, the
generator owns it. In the tail, generation blame must clear a raised
evidence floor, and when nothing clears it the attributor says
retrieval_rank with low confidence, which is the designed blind spot
ADR-005 measures.
