"""A checker nobody has seen fail is a checker nobody should believe."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import check_numbers  # noqa: E402
from check_numbers import (  # noqa: E402
    anchor_pattern,
    check_anchor_containment,
    forward_pass,
    load_metrics,
    reverse_pass,
    run_checks,
)


def test_the_committed_metrics_file_exists_and_every_metric_has_anchors():
    metrics = load_metrics()
    assert len(metrics) >= 25
    for name, metric in metrics.items():
        assert metric["rendered"], name
        assert metric["anchors"], name


def test_breaking_a_readme_figure_makes_the_checker_fail():
    # The prove-the-checker rule: substitute a wrong value into a real anchor
    # and the forward pass must convict exactly that sentence.
    metrics = load_metrics()
    anchor = metrics["attribution_accuracy_pct"]["anchors"][0]
    bad = "Intro. " + anchor.replace("{}", "12.3") + " of cases."
    problems = forward_pass(metrics, {"README.md": bad})
    assert any("stale value for attribution_accuracy_pct" in p for p in problems)


def test_a_correct_anchored_figure_passes_the_forward_pass():
    metrics = load_metrics()
    anchor = metrics["attribution_accuracy_pct"]["anchors"][0]
    good = anchor.replace("{}", metrics["attribution_accuracy_pct"]["rendered"])
    assert forward_pass(metrics, {"README.md": good}) == []


def test_the_sweep_convicts_a_stale_figure_inside_a_fenced_block():
    metrics = load_metrics()
    anchor = metrics["tests_total"]["anchors"][0]
    doc = "```\n" + anchor.replace("{}", "9999") + "\n```\n"
    problems = forward_pass(metrics, {"docs/experiments/report.md": doc})
    assert any("stale value for tests_total" in p for p in problems)


def test_an_unknown_number_in_readme_prose_is_flagged_by_the_reverse_pass():
    metrics = load_metrics()
    problems = reverse_pass(metrics, "The corpus has 4242 llamas.")
    assert any("4242" in p for p in problems)


def test_numbers_inside_code_spans_are_exempt_from_the_reverse_pass():
    metrics = load_metrics()
    problems = reverse_pass(metrics, "Run `stageattr run --seed 4242` to begin.")
    assert not any("4242" in p for p in problems)


def test_a_single_character_metric_value_is_not_whitelisted_bare():
    # trace_records_per_question renders as one character; a bare occurrence
    # in prose must be flagged because only its anchor legitimises it.
    metrics = load_metrics()
    rendered = metrics["trace_records_per_question"]["rendered"]
    assert len(rendered) == 1
    problems = reverse_pass(metrics, f"There are {rendered} llamas here.")
    assert any(f"number {rendered!r}" in p for p in problems)


def test_the_same_value_under_its_anchor_passes():
    metrics = load_metrics()
    anchor = metrics["trace_records_per_question"]["anchors"][0]
    text = anchor.replace("{}", metrics["trace_records_per_question"]["rendered"])
    assert reverse_pass(metrics, text) == []


def test_exit_code_prose_is_exempt_by_context_not_by_digit():
    metrics = load_metrics()
    assert reverse_pass(metrics, "The tool returns exit code 2 when it refuses.") == []
    problems = reverse_pass(metrics, "We shipped 2 llamas.")
    assert any("'2'" in p for p in problems)


def test_no_anchor_is_contained_in_another_metrics_anchor():
    assert check_anchor_containment(load_metrics()) == []


def test_every_context_exemption_is_used_by_the_reverse_passed_documents():
    # Rule: an unused exemption is a hole. The hygiene check runs against the
    # documents the reverse pass covers; an exemption none of them exercises
    # is dead weight waiting to hide a claim.
    texts = [
        (ROOT / "README.md").read_text(encoding="utf-8"),
        (ROOT / "docs" / "defense-guide.md").read_text(encoding="utf-8"),
    ]
    assert check_numbers.exemption_hygiene(texts) == []


def test_the_reverse_pass_covers_the_defense_guide_too():
    # Pins audit finding 5: the guide promised it could not drift while only
    # the README was reverse passed. An invented figure in the guide must now
    # be a build failure, not a coincidence.
    metrics = load_metrics()
    problems = check_numbers.run_checks(
        {"docs/defense-guide.md": "The guide invents 4242 llamas here."}
    )
    del metrics
    assert any("4242" in p and "defense-guide" in p for p in problems)


def test_no_exemption_pattern_equals_a_rendered_metric_value():
    metrics = load_metrics()
    rendered = {m["rendered"] for m in metrics.values()}
    for pattern in check_numbers.CONTEXT_EXEMPTIONS.values():
        assert pattern not in rendered


def test_the_full_check_passes_on_the_committed_documents():
    assert run_checks() == []


def test_anchor_patterns_capture_decimals_and_percent_values():
    pattern = anchor_pattern("recovers the cause in {}%")
    match = pattern.search("it recovers the cause in 99.5% of cases")
    assert match and match.group(1) == "99.5"


def test_the_duration_sweep_covers_every_tracked_markdown_file():
    problems = check_numbers.duration_sweep()
    assert problems == []


def test_metrics_json_is_regenerated_deterministically():
    # Guarded on the count files existing so this test can never recursively
    # spawn the suite it is part of. On the very first `make verify` of a
    # fresh clone they do not exist yet, so the byte-equality check runs
    # from the second verify onward; CI's git-diff gate covers the first.
    import pytest

    if not (ROOT / "junit.xml").exists() or not (ROOT / "coverage.json").exists():
        pytest.skip("run through make verify, where the count files exist")
    import subprocess

    before = (ROOT / "docs" / "metrics.json").read_bytes()
    subprocess.run(
        [sys.executable, "tools/collect_metrics.py"], cwd=ROOT, check=True, capture_output=True
    )
    after = (ROOT / "docs" / "metrics.json").read_bytes()
    assert before == after


def test_the_trace_recovers_every_retrieval_side_failure_in_the_committed_capture():
    """Pins audit pass 2 finding N2: a fix commit reworded the README and
    orphaned the anchors on the 360-of-360 claim, leaving a figure that was
    true only by inertia. The equality now has a tripwire: if a change makes
    the attributor miss a single retrieval-side failure, regenerating the
    experiments breaks this test before the README can go stale."""
    e1 = json.loads(
        (ROOT / "docs" / "experiments" / "e1_results.json").read_text(encoding="utf-8")
    )
    default_arm = e1["arms"][0]
    for cls in ("retrieval_miss", "retrieval_rank"):
        bucket = default_arm["per_class"][cls]
        assert bucket["recovered"] == bucket["n"], cls
    metrics = load_metrics()
    recovered = metrics["retrieval_recovered_n"]["rendered"]
    assert recovered == metrics["retrieval_failures_n"]["rendered"]


def test_the_coverage_denominator_excludes_exactly_the_two_network_gated_modules():
    """Pins audit pass 3's word-count finding: the README says 'the two
    network gated modules' are excluded from the coverage denominator, and a
    word is invisible to the number checker. The omit list is pinned here so
    a third exclusion cannot slip in and quietly falsify both the sentence
    and the published coverage figure's meaning."""
    import tomllib

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    omit = sorted(pyproject["tool"]["coverage"]["run"]["omit"])
    assert omit == [
        "src/rag_stage_attribution/adapters/azure_openai.py",
        "src/rag_stage_attribution/embeddings/capture.py",
    ]
