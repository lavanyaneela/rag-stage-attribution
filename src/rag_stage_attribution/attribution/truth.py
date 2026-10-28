"""Ground truth labelling: what actually failed, from the harness's ledger.

Truth is mechanical and uses information the attributor never sees: the
injected mode. The labelling rule encodes one substantive decision, recorded
in ADR-004: upstream failure dominates. If the golden chunk never reached the
generator, the failure is retrieval's whatever the generator was injected to
do, because fixing the generator could not have fixed this answer. Truth is
only defined for answers that scored wrong; a lucky correct answer under an
injected fault is not a failure and is not labelled.
"""

from __future__ import annotations

STAGES = ("retrieval_miss", "retrieval_rank", "generation")


def ground_truth(
    golden_ids: frozenset[str],
    retrieved_ids: list[str],
    context_ids: list[str],
    injected_mode: str,
) -> str:
    """Label a scored-wrong answer with the stage that truly failed.

    Precondition: golden_ids is non-empty. A question whose golden span no
    chunk contains is unjudgeable and is routed to the refusal path before
    truth is ever asked for.
    """
    if not golden_ids:
        raise ValueError("ground_truth called for an unjudgeable question")
    in_retrieved = any(cid in golden_ids for cid in retrieved_ids)
    in_context = any(cid in golden_ids for cid in context_ids)
    if not in_retrieved:
        return "retrieval_miss"
    if not in_context:
        return "retrieval_rank"
    # The golden chunk was in front of the generator. The only way the answer
    # is wrong from here is generator behaviour, injected or emergent.
    assert injected_mode in {"ignore", "wrong_chunk", "abstain", "faithful"}
    return "generation"
