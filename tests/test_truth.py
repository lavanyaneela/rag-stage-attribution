"""Ground truth is mechanical; its one judgement call is pinned here."""

import pytest

from rag_stage_attribution.attribution import STAGES, ground_truth

GOLDEN = frozenset({"doc:p01"})


def test_golden_absent_from_the_retriever_is_a_retrieval_miss():
    assert ground_truth(GOLDEN, ["a", "b"], ["a"], "faithful") == "retrieval_miss"


def test_golden_retrieved_but_cut_from_context_is_a_rank_failure():
    assert ground_truth(GOLDEN, ["a", "doc:p01"], ["a"], "faithful") == "retrieval_rank"


def test_golden_in_context_makes_any_wrong_answer_generations_fault():
    assert ground_truth(GOLDEN, ["doc:p01"], ["doc:p01"], "ignore") == "generation"


def test_upstream_failure_dominates_an_injected_generation_fault():
    # The generator was injected to ignore, but the golden chunk never reached
    # it, so fixing the generator could not have fixed this answer (ADR-004).
    assert ground_truth(GOLDEN, ["a"], ["a"], "ignore") == "retrieval_miss"


def test_an_unjudgeable_question_is_refused_loudly():
    with pytest.raises(ValueError, match="unjudgeable"):
        ground_truth(frozenset(), ["a"], ["a"], "faithful")


def test_the_stage_vocabulary_is_fixed():
    assert STAGES == ("retrieval_miss", "retrieval_rank", "generation")
