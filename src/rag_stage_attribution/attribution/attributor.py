"""Assign blame for a wrong answer from the trace, the golden label, and text.

What the attributor may see: the per node trace, the answer, the citations,
the golden chunk ids and texts, and the chunk texts. What it may never see:
the injected mode, which is the ground truth it is being scored against, and
the parametric bank, which is the stand-in's script. The separation is
enforced by this module's signature and pinned by a test.

The decision order is elimination first, inference second. Retrieval-side
blame is read directly off the trace, which is the concrete thing node level
instrumentation buys (E1's ablation prices it). Generation-side blame is
inference from lexical support and citations, and ADR-004 names the one place
the inference goes soft: a golden chunk sitting in the tail of the context
window, where "provided and ignored" and "barely not provided" are nearly the
same observable event.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_stage_attribution.config import PipelineConfig
from rag_stage_attribution.retrieval.rerank import lexical_overlap


@dataclass(frozen=True)
class Attribution:
    stage: str  # retrieval_miss | retrieval_rank | generation | cannot_attribute
    confidence: float  # in [0, 1]; 0 when stage is cannot_attribute
    reason: str


def _support(answer: str, chunk_text: str) -> float:
    """Share of the answer's content words present in the chunk."""
    return lexical_overlap(answer, chunk_text)


def attribute(
    cfg: PipelineConfig,
    golden_ids: frozenset[str],
    chunk_texts: dict[str, str],
    answer: str,
    citations: tuple[str, ...],
    retrieved_ids: list[str] | None,
    context_ids: list[str] | None,
) -> Attribution:
    """Blame one scored-wrong answer.

    retrieved_ids and context_ids come from the trace. Passing None for both
    is the ablation arm: attribution from the answer and citations alone,
    which is what a pipeline without node instrumentation can do.
    """
    if not golden_ids:
        return Attribution("cannot_attribute", 0.0, "golden span not present in any chunk")

    if retrieved_ids is None or context_ids is None:
        return _attribute_without_trace(cfg, golden_ids, chunk_texts, answer, citations)

    if not any(cid in golden_ids for cid in retrieved_ids):
        return Attribution(
            "retrieval_miss", 0.95, "trace: golden chunk absent from retriever output"
        )
    if not any(cid in golden_ids for cid in context_ids):
        return Attribution(
            "retrieval_rank", 0.90, "trace: golden chunk retrieved but cut by the rerank"
        )

    # The golden chunk was in the context window. Where exactly?
    position = min(i for i, cid in enumerate(context_ids) if cid in golden_ids)
    in_tail = position >= cfg.k_ctx - cfg.tail_slots

    support_golden = max(
        (_support(answer, chunk_texts[cid]) for cid in context_ids if cid in golden_ids),
        default=0.0,
    )
    others = [cid for cid in context_ids if cid not in golden_ids]
    support_other = max((_support(answer, chunk_texts[cid]) for cid in others), default=0.0)
    support_max = max(support_golden, support_other)

    fabrication_margin = cfg.parametric_floor - support_max
    wrong_source_margin = support_other - support_golden

    cited_non_golden = [cid for cid in citations if cid not in golden_ids and cid in chunk_texts]
    citation_evidence = 0.0
    citation_reason = ""
    cites_only_non_golden = bool(citations) and not any(cid in golden_ids for cid in citations)
    if cites_only_non_golden and cited_non_golden:
        cited_support = max(_support(answer, chunk_texts[cid]) for cid in cited_non_golden)
        if cited_support >= cfg.support_used_floor:
            citation_evidence = cited_support
            citation_reason = "citation names a non-golden chunk that supports the answer"
        else:
            citation_evidence = cfg.support_used_floor
            citation_reason = "citation names a chunk the answer does not draw on"

    gen_evidence = max(fabrication_margin, wrong_source_margin, citation_evidence)

    if not in_tail:
        # Elimination: the trace proves retrieval delivered the golden chunk
        # to a non-tail slot, so a wrong answer is the generator's.
        confidence = min(0.95, 0.60 + gen_evidence)
        reason = citation_reason or (
            "fabrication: answer unsupported by any provided chunk"
            if fabrication_margin >= wrong_source_margin
            else "answer draws on a non-golden chunk over the golden one"
        )
        return Attribution("generation", confidence, "trace: golden in context; " + reason)

    # Tail territory, ADR-004: presence in the last slots is weak evidence of
    # provision, so generation blame must clear a raised evidence floor.
    tail_floor = 2.0 * cfg.support_margin_floor
    if citation_evidence > 0.0:
        return Attribution(
            "generation", min(0.90, 0.55 + citation_evidence), "tail: " + citation_reason
        )
    if gen_evidence >= tail_floor:
        return Attribution(
            "generation",
            min(0.85, 0.50 + gen_evidence),
            "tail: generation evidence clears the raised floor",
        )
    return Attribution(
        "retrieval_rank",
        0.40,
        "tail: golden at the context tail with weak generation evidence; "
        "treated as effectively not provided (ADR-004)",
    )


def _attribute_without_trace(
    cfg: PipelineConfig,
    golden_ids: frozenset[str],
    chunk_texts: dict[str, str],
    answer: str,
    citations: tuple[str, ...],
) -> Attribution:
    """The ablation: what attribution can do with no node instrumentation.

    Without the trace there is no way to know what the retriever returned or
    what the generator was shown, which is the impossibility this repository
    opens with. The honest outputs are a generation verdict when the answer
    text itself betrays the generator, and cannot_attribute otherwise.
    """
    support_golden = max((_support(answer, chunk_texts[cid]) for cid in golden_ids), default=0.0)
    if citations:
        if any(cid in golden_ids for cid in citations):
            return Attribution(
                "generation", 0.55, "no trace: cites the golden chunk yet answers wrongly"
            )
        cited_known = [cid for cid in citations if cid in chunk_texts]
        if cited_known:
            cited_support = max(_support(answer, chunk_texts[cid]) for cid in cited_known)
            if cited_support >= cfg.support_used_floor and support_golden < cfg.support_used_floor:
                return Attribution(
                    "generation",
                    0.50,
                    "no trace: answer rests on a cited non-golden chunk, but whether the "
                    "golden chunk was ever offered is unknowable",
                )
    if support_golden >= cfg.support_used_floor:
        return Attribution(
            "generation", 0.45, "no trace: answer draws on the golden chunk yet gets it wrong"
        )
    return Attribution(
        "cannot_attribute",
        0.0,
        "no trace: never-retrieved and retrieved-but-ignored present identically",
    )
