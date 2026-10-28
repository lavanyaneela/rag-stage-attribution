"""Interval arithmetic and the verdict policy: the tool's answer is law here."""

import pytest

from rag_stage_attribution.attribution import clopper_pearson, verdict
from rag_stage_attribution.config import PipelineConfig

CFG = PipelineConfig()


def test_clopper_pearson_matches_a_hand_checked_value():
    low, high = clopper_pearson(8, 10)
    assert low == pytest.approx(0.4439, abs=1e-3)
    assert high == pytest.approx(0.9748, abs=1e-3)


def test_a_measured_zero_still_has_a_non_zero_upper_bound():
    low, high = clopper_pearson(0, 18)
    assert low == 0.0
    assert high > 0.15  # receipts rule 4: price the zero at what the sample allows


def test_a_perfect_score_still_has_a_lower_bound_below_one():
    low, high = clopper_pearson(18, 18)
    assert high == 1.0
    assert low < 1.0


def test_bounds_tighten_with_more_samples():
    _, high_small = clopper_pearson(0, 10)
    _, high_large = clopper_pearson(0, 100)
    assert high_large < high_small


def test_invalid_counts_are_rejected():
    with pytest.raises(ValueError):
        clopper_pearson(5, 3)
    with pytest.raises(ValueError):
        clopper_pearson(0, 0)


def _verdict(counts, cannot=0, unjudgeable=0, failures=None):
    failures = failures if failures is not None else sum(counts.values()) + cannot
    return verdict(CFG, counts, cannot, unjudgeable, failures)


def test_a_dominant_generation_share_is_decided_with_exit_zero():
    result = _verdict({"generation": 40, "retrieval_miss": 3, "retrieval_rank": 2})
    assert result.code == 0 and result.label == "generation"
    assert result.bounds["generation"][0] > result.bounds["retrieval"][1]


def test_a_dominant_retrieval_share_is_decided_as_retrieval():
    result = _verdict({"generation": 4, "retrieval_miss": 30, "retrieval_rank": 11})
    assert result.code == 0 and result.label == "retrieval"


def test_overlapping_intervals_mean_both_and_exit_one():
    result = _verdict({"generation": 14, "retrieval_miss": 6, "retrieval_rank": 5})
    assert result.code == 1 and result.label == "both"


def test_a_near_even_split_is_both_not_a_coin_flip():
    result = _verdict({"generation": 13, "retrieval_miss": 6, "retrieval_rank": 6})
    assert result.code == 1


def test_broken_golden_coverage_refuses_before_anything_else():
    result = _verdict({"generation": 40}, unjudgeable=2)
    assert result.code == 2 and "no golden chunk" in result.detail


def test_no_failures_is_a_successful_answer_not_a_refusal():
    result = _verdict({}, failures=0)
    assert result.code == 0 and result.label == "no_failures"


def test_too_few_attributable_failures_refuse():
    result = _verdict({"generation": 5}, failures=5)
    assert result.code == 2 and "policy floor" in result.detail


def test_too_many_unattributable_failures_refuse():
    result = _verdict({"generation": 20}, cannot=8)
    assert result.code == 2 and "could not be attributed" in result.detail


def test_the_refusal_line_is_a_share_not_a_count():
    result = _verdict({"generation": 200}, cannot=8)
    assert result.code == 0  # 8 of 208 is under the 20% line


def test_shares_sum_to_one_over_attributable_failures():
    result = _verdict({"generation": 30, "retrieval_miss": 10, "retrieval_rank": 10})
    assert sum(result.shares.values()) == pytest.approx(1.0)
