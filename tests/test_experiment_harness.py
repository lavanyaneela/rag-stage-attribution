"""The experiment harness is measurement code, so its bugs are pinned too."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import _common  # noqa: E402
from _common import resources_for  # noqa: E402

from rag_stage_attribution.config import PipelineConfig  # noqa: E402


def test_the_resource_cache_distinguishes_configs_that_differ_only_in_tail_slots():
    """Pins a bug that shipped identical experiment rows.

    The cache originally keyed on a hand-picked subset of config fields that
    left tail_slots out, and run_suite reads its config off the cached
    resources. E2's widened-tail sensitivity arms therefore reran the
    baseline tail definition, and the output table contained two rows that
    were byte identical when they could not be. The fix keys the cache on
    the whole frozen config; this test would have failed before it.
    """
    a = resources_for(PipelineConfig(tail_slots=1))
    b = resources_for(PipelineConfig(tail_slots=2))
    assert a is not b
    assert a.cfg.tail_slots == 1 and b.cfg.tail_slots == 2


def test_the_cache_still_reuses_resources_for_an_identical_config():
    a = resources_for(PipelineConfig())
    b = resources_for(PipelineConfig())
    assert a is b


def test_the_cache_key_is_the_config_itself_not_a_summary_of_it():
    assert all(isinstance(key, PipelineConfig) for key in _common._RESOURCE_CACHE)
