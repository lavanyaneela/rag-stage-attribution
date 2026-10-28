"""The config validates itself, because every threshold is load bearing."""

import pytest

from rag_stage_attribution.config import CHUNKING_STRATEGIES, PipelineConfig


def test_the_default_config_is_valid():
    cfg = PipelineConfig()
    assert cfg.chunking in CHUNKING_STRATEGIES


def test_an_unknown_chunking_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown chunking"):
        PipelineConfig(chunking="sentences")


def test_a_context_window_larger_than_the_retriever_depth_is_rejected():
    with pytest.raises(ValueError, match="k_ctx"):
        PipelineConfig(top_k=4, k_ctx=8)


def test_a_zero_context_window_is_rejected():
    with pytest.raises(ValueError, match="k_ctx"):
        PipelineConfig(k_ctx=0)


def test_tail_slots_must_leave_at_least_one_non_tail_slot():
    with pytest.raises(ValueError, match="tail_slots"):
        PipelineConfig(k_ctx=4, tail_slots=4)


def test_tail_slots_may_be_zero_which_disables_the_tail_rule():
    assert PipelineConfig(tail_slots=0).tail_slots == 0


def test_with_returns_a_new_config_and_leaves_the_original_alone():
    base = PipelineConfig()
    other = base.with_(k_ctx=3)
    assert other.k_ctx == 3
    assert base.k_ctx == 4


def test_path_properties_point_inside_the_repository(cfg):
    assert (cfg.corpus_dir / "expense-policy.md").exists()
    assert cfg.golden_path.exists()
    assert cfg.parametric_path.exists()
    assert (cfg.embeddings_dir / "manifest.json").exists()
