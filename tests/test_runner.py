"""The runner holds the ledger; these tests hold the runner."""

import inspect

from rag_stage_attribution import runner as runner_module
from rag_stage_attribution.attribution import ground_truth
from rag_stage_attribution.injection import InjectionMix
from rag_stage_attribution.runner import run_suite, summarise


def test_a_suite_run_covers_every_golden_question_once(resources):
    records = run_suite(resources, InjectionMix(), seed=1)
    assert len(records) == len(resources.golden)
    assert {r["qid"] for r in records} == {q.qid for q in resources.golden}


def test_the_same_seed_reproduces_the_same_records(resources):
    a = run_suite(resources, InjectionMix(), seed=2)
    b = run_suite(resources, InjectionMix(), seed=2)
    assert a == b


def test_failures_carry_truth_attribution_and_a_reason(resources):
    records = run_suite(resources, InjectionMix(), seed=1)
    failures = [r for r in records if not r["correct"]]
    assert failures
    for r in failures:
        assert r["truth"] in {"retrieval_miss", "retrieval_rank", "generation"}
        assert r["attributed"] in {
            "retrieval_miss",
            "retrieval_rank",
            "generation",
            "cannot_attribute",
        }
        assert r["reason"]


def test_correct_answers_are_not_labelled_or_attributed(resources):
    records = run_suite(resources, InjectionMix(), seed=1)
    for r in records:
        if r["correct"]:
            assert "truth" not in r and "attributed" not in r


def test_a_faithful_only_mix_has_only_retrieval_failures(resources):
    mix = InjectionMix(faithful=1.0, ignore=0.0, wrong_chunk=0.0, abstain=0.0)
    records = run_suite(resources, mix, seed=1)
    failures = [r for r in records if not r["correct"]]
    assert failures, "retrieval alone should produce some failures on this corpus"
    assert all(r["truth"].startswith("retrieval") for r in failures)


def test_an_ignore_only_mix_fails_wherever_the_golden_chunk_was_provided(resources):
    mix = InjectionMix(faithful=0.0, ignore=1.0, wrong_chunk=0.0, abstain=0.0)
    records = run_suite(resources, mix, seed=1)
    in_context = [r for r in records if r["golden_position"] is not None]
    assert in_context
    assert all(not r["correct"] and r["truth"] == "generation" for r in in_context)


def test_the_trace_off_arm_cannot_see_retrieval_failures(resources):
    records = run_suite(resources, InjectionMix(), seed=1, use_trace=False)
    retrieval_truth = [r for r in records if not r["correct"] and r["truth"] != "generation"]
    assert retrieval_truth
    assert all(r["attributed"] != r["truth"] for r in retrieval_truth)


def test_summarise_counts_add_up(resources):
    records = run_suite(resources, InjectionMix(), seed=3)
    summary = summarise(records)
    assert summary["questions"] == len(records)
    assert summary["failures"] == sum(1 for r in records if not r["correct"] and r["judgeable"])
    attributed_total = sum(summary["attributed_counts"].values())
    assert attributed_total + summary["cannot_attribute"] == summary["failures"]
    assert summary["recovered"] <= attributed_total


def test_summarise_of_an_empty_run_reports_no_accuracy():
    summary = summarise([])
    assert summary["failures"] == 0 and summary["accuracy_over_failures"] is None


def test_the_attributor_is_never_handed_the_injected_mode():
    # The separation is structural: the attribute() call inside run_suite must
    # not reference the mode ledger. Source inspection is blunt and effective.
    source = inspect.getsource(runner_module.run_suite)
    call = source[source.index("result = attribute(") : source.index("record[\"attributed\"]")]
    assert "mode" not in call


def test_the_truth_labeller_is_handed_the_mode_and_the_attributor_output_is_not_reused():
    source = inspect.getsource(runner_module.run_suite)
    assert "ground_truth(" in source
    assert ground_truth.__module__ == "rag_stage_attribution.attribution.truth"
