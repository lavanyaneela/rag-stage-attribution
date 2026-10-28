"""Mode assignment must be stable, independent, and honest about its mix."""

import pytest

from rag_stage_attribution.injection import InjectionMix, draw_mode


def test_a_mix_that_does_not_sum_to_one_is_rejected():
    with pytest.raises(ValueError, match="sum to 1"):
        InjectionMix(faithful=0.9, ignore=0.9, wrong_chunk=0.0, abstain=0.0)


def test_a_negative_probability_is_rejected():
    with pytest.raises(ValueError, match="non negative"):
        InjectionMix(faithful=1.2, ignore=-0.2, wrong_chunk=0.0, abstain=0.0)


def test_mode_assignment_is_deterministic_per_question_and_seed():
    mix = InjectionMix()
    assert draw_mode(mix, "q001", 7) == draw_mode(mix, "q001", 7)


def test_different_seeds_reassign_modes():
    mix = InjectionMix()
    draws = {draw_mode(mix, "q001", seed) for seed in range(40)}
    assert len(draws) > 1


def test_a_degenerate_mix_assigns_only_its_mode():
    mix = InjectionMix(faithful=0.0, ignore=1.0, wrong_chunk=0.0, abstain=0.0)
    assert all(draw_mode(mix, f"q{i:03d}", 1) == "ignore" for i in range(30))


def test_the_empirical_mix_tracks_the_requested_mix():
    mix = InjectionMix()
    draws = [draw_mode(mix, f"q{i:03d}", seed) for i in range(73) for seed in range(30)]
    share = draws.count("faithful") / len(draws)
    assert abs(share - mix.faithful) < 0.05


def test_the_mix_round_trips_through_as_dict():
    mix = InjectionMix(faithful=0.4, ignore=0.4, wrong_chunk=0.1, abstain=0.1)
    assert InjectionMix(**mix.as_dict()) == mix
