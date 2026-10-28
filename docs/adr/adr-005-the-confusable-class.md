# ADR-005: The confusable class, named and priced

Status: accepted. This is the boundary document.

## The class

Ground truth is a generation fault, and the golden chunk sat in the tail of
the context window. The generator was given the right content and ignored
it, from the last slot the rerank kept. By ADR-004's own definitions this
is the place where "provided and ignored" and "barely not provided" are
nearly indistinguishable observable events, so the attributor's tail rule
converts weak evidence into retrieval_rank on purpose rather than
manufacturing confident generation verdicts it cannot back.

## The measurement

E2 measures accuracy on exactly this slice at its measured prevalence, with
exact 95% intervals, since 34 tail-ignored failures across 30 seeds cannot
carry a point estimate alone. The attributor recovers
the class at 82.4% with the contract on. Turn the contract off and it drops
to 52.9%, with an interval from 35.1% up to 70.2%: on the information the pipeline
leaves behind without forced citations, the designed blind spot performs
near a coin flip, and the interval says the sample cannot rule out worse.

Widening the tail definition to two slots grows the class to 46 cases and
moves the accuracies little, which is the point of publishing the
sensitivity arms: the blind spot tracks the definition, not a lucky
threshold.

## The extra signal that resolves it, and its limit

The citation contract is the resolving signal: a generator forced to name
the chunks it used converts most tail-ignored failures into attributable
ones, either because the lying citation points at a chunk the answer does
not draw on, or because it points at a chunk the answer does draw on, and
both are generation evidence. The limit is stated plainly: the contract
makes attribution trust an artifact the faulty component controls. The
stand-in lies randomly; an adversarial or systematically self justifying
generator could cite the golden chunk on every ignored answer, and the
class would collapse back toward the contract-off number. Attribution that
survives an adversarial generator needs a signal the generator does not
emit, such as attention or logit level attribution, which is outside this
repository's offline boundary and is the honest edge of the method.
