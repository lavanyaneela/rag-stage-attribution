"""Every branch of the blame logic, driven through a tiny synthetic corpus."""

import pytest

from rag_stage_attribution.attribution import attribute
from rag_stage_attribution.config import PipelineConfig

CFG = PipelineConfig()
GOLDEN = frozenset({"g"})
TEXTS = {
    "g": "Itemized receipts are required for any single expense above $25.",
    "a": "Guest wifi passwords rotate on a schedule displayed at reception.",
    "b": "On approved travel, receipts are required only for expenses above $75.",
    "c": "Snapshots are retained for a fixed window and then deleted.",
    "d": "Business class flights require sign off from a vice president.",
}
CTX_GOLDEN_FRONT = ["g", "a", "b", "c"]
CTX_GOLDEN_TAIL = ["a", "b", "c", "g"]
RETRIEVED_ALL = ["g", "a", "b", "c", "d"]


def test_an_empty_golden_set_cannot_be_attributed():
    result = attribute(CFG, frozenset(), TEXTS, "any", (), RETRIEVED_ALL, CTX_GOLDEN_FRONT)
    assert result.stage == "cannot_attribute"
    assert result.confidence == 0.0


def test_golden_missing_from_the_retriever_output_is_a_miss():
    result = attribute(CFG, GOLDEN, TEXTS, "wrong", (), ["a", "b", "c"], ["a", "b"], )
    assert result.stage == "retrieval_miss"
    assert "absent from retriever output" in result.reason


def test_golden_cut_by_the_rerank_is_a_rank_failure():
    result = attribute(CFG, GOLDEN, TEXTS, "wrong", (), RETRIEVED_ALL, ["a", "b", "c"])
    assert result.stage == "retrieval_rank"
    assert "cut by the rerank" in result.reason


def test_golden_in_a_front_slot_makes_a_wrong_answer_generations_by_elimination():
    answer = "The guest wifi password rotates on a schedule at reception."
    result = attribute(CFG, GOLDEN, TEXTS, answer, (), RETRIEVED_ALL, CTX_GOLDEN_FRONT)
    assert result.stage == "generation"
    assert "golden in context" in result.reason


def test_an_unsupported_answer_reads_as_fabrication():
    answer = "Common practice is forty five days for everything."
    result = attribute(CFG, GOLDEN, TEXTS, answer, (), RETRIEVED_ALL, CTX_GOLDEN_FRONT)
    assert result.stage == "generation"
    assert "unsupported" in result.reason


def test_a_citation_to_a_supporting_non_golden_chunk_is_generation_evidence():
    answer = "On approved travel, receipts are required only for expenses above $75."
    result = attribute(CFG, GOLDEN, TEXTS, answer, ("b",), RETRIEVED_ALL, CTX_GOLDEN_FRONT)
    assert result.stage == "generation"
    assert "supports the answer" in result.reason


def test_a_citation_the_answer_does_not_draw_on_is_also_generation_evidence():
    answer = "Common practice is forty five days for everything."
    result = attribute(CFG, GOLDEN, TEXTS, answer, ("c",), RETRIEVED_ALL, CTX_GOLDEN_FRONT)
    assert result.stage == "generation"
    assert "does not draw on" in result.reason


def test_tail_presence_with_weak_evidence_is_misread_as_a_rank_failure():
    # ADR-004: the designed blind spot. The answer resembles the golden chunk
    # (a misremembered policy), no citation helps, and the golden chunk sits
    # in the tail, so the attributor concludes it was effectively not provided.
    answer = "Itemized receipts are required for any single expense above $50."
    result = attribute(CFG, GOLDEN, TEXTS, answer, (), RETRIEVED_ALL, CTX_GOLDEN_TAIL)
    assert result.stage == "retrieval_rank"
    assert result.confidence == pytest.approx(0.40)
    assert "ADR-004" in result.reason


def test_tail_presence_with_a_lying_citation_is_still_caught():
    answer = "Itemized receipts are required for any single expense above $50."
    result = attribute(CFG, GOLDEN, TEXTS, answer, ("c",), RETRIEVED_ALL, CTX_GOLDEN_TAIL)
    assert result.stage == "generation"
    assert result.reason.startswith("tail:")


def test_tail_presence_with_strong_fabrication_evidence_clears_the_raised_floor():
    answer = "zz qq ww ee rr tt yy uu"  # supported by nothing at all
    result = attribute(CFG, GOLDEN, TEXTS, answer, (), RETRIEVED_ALL, CTX_GOLDEN_TAIL)
    assert result.stage == "generation"
    assert "raised floor" in result.reason


def test_a_citation_to_the_golden_chunk_neutralises_citation_evidence_in_the_tail():
    answer = "Itemized receipts are required for any single expense above $50."
    result = attribute(CFG, GOLDEN, TEXTS, answer, ("g",), RETRIEVED_ALL, CTX_GOLDEN_TAIL)
    assert result.stage == "retrieval_rank"


def test_disabling_the_tail_rule_restores_elimination():
    cfg = PipelineConfig(tail_slots=0)
    answer = "Itemized receipts are required for any single expense above $50."
    result = attribute(cfg, GOLDEN, TEXTS, answer, (), RETRIEVED_ALL, CTX_GOLDEN_TAIL)
    assert result.stage == "generation"


def test_without_a_trace_a_golden_citation_plus_a_wrong_answer_is_generation():
    result = attribute(CFG, GOLDEN, TEXTS, "wrong words entirely", ("g",), None, None)
    assert result.stage == "generation"
    assert result.reason.startswith("no trace")


def test_without_a_trace_a_supported_foreign_citation_is_generation_with_doubt():
    answer = "Guest wifi passwords rotate on a schedule displayed at reception."
    result = attribute(CFG, GOLDEN, TEXTS, answer, ("a",), None, None)
    assert result.stage == "generation"
    assert "unknowable" in result.reason


def test_without_a_trace_an_answer_drawing_on_the_golden_chunk_is_generation():
    answer = "Itemized receipts are required for any single expense above $50."
    result = attribute(CFG, GOLDEN, TEXTS, answer, (), None, None)
    assert result.stage == "generation"


def test_without_a_trace_everything_else_is_the_impossibility_itself():
    result = attribute(CFG, GOLDEN, TEXTS, "completely unrelated words", (), None, None)
    assert result.stage == "cannot_attribute"
    assert "present identically" in result.reason


def test_confidence_is_always_within_bounds():
    for ctx in (CTX_GOLDEN_FRONT, CTX_GOLDEN_TAIL, ["a", "b", "c"]):
        result = attribute(CFG, GOLDEN, TEXTS, "receipts above $50", (), RETRIEVED_ALL, ctx)
        assert 0.0 <= result.confidence <= 1.0
