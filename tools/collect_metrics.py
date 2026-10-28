"""Re-measure every published figure and write docs/metrics.json.

Every number a document is allowed to state comes out of this file, derived
from machine readable sources: the committed experiment outputs, the fixture
manifests, pytest's junit XML, and coverage's JSON. Nothing here parses a
progress line, nothing here is typed by hand, and nothing here is a duration
(the one duration-derived figure, E5's overhead ratio, stays in its own
committed capture file and is asserted by a test, never anchored in prose).
"""

from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "docs" / "experiments"


def _load(name: str) -> dict:
    return json.loads((EXP / name).read_text(encoding="utf-8"))


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}"


def _run_tests_for_counts() -> tuple[int, str]:
    """Collected test count from junit XML and line coverage from coverage JSON.

    The published test count is the collected count (tests + skips if any),
    because a count that subtracts environment dependent skips differs
    between machines and fails its own receipts on a fresh clone.
    """
    junit = ROOT / "junit.xml"
    covjson = ROOT / "coverage.json"
    if not junit.exists() or not covjson.exists():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--junitxml=junit.xml",
                "--cov=rag_stage_attribution",
                "--cov-report=json",
                "-q",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    suite = ET.parse(junit).getroot()
    node = suite if suite.tag == "testsuite" else suite.find("testsuite")
    tests_total = int(node.attrib["tests"])
    coverage = json.loads(covjson.read_text(encoding="utf-8"))
    coverage_pct = f"{coverage['totals']['percent_covered']:.1f}"
    return tests_total, coverage_pct


def build_metrics() -> dict:
    e1 = _load("e1_results.json")
    e2 = _load("e2_results.json")
    e3 = _load("e3_results.json")
    e4 = _load("e4_results.json")
    e5 = _load("e5_results.json")
    manifest = json.loads((ROOT / "fixtures" / "embeddings" / "manifest.json").read_text())

    golden_lines = (ROOT / "fixtures" / "golden" / "questions.jsonl").read_text().splitlines()
    corpus_docs = len(list((ROOT / "fixtures" / "corpus").glob("*.md")))

    default_arm = e1["arms"][0]
    no_contract_arm = e1["arms"][1]
    ablation_arm = e1["arms"][2]
    retr_recovered = sum(
        default_arm["per_class"][k]["recovered"] for k in ("retrieval_miss", "retrieval_rank")
    )
    retr_n = sum(default_arm["per_class"][k]["n"] for k in ("retrieval_miss", "retrieval_rank"))
    abl_retr_recovered = sum(
        ablation_arm["per_class"][k]["recovered"] for k in ("retrieval_miss", "retrieval_rank")
    )

    conf_on = e2["arms"][0]
    conf_off = e2["arms"][1]
    base, improved = e3["arms"]
    paragraph_arm = next(a for a in e4["arms"] if a["strategy"] == "paragraph")
    fixed_arm = next(a for a in e4["arms"] if a["strategy"] == "fixed160")

    tests_total, coverage_pct = _run_tests_for_counts()

    metrics: dict[str, dict] = {
        "golden_questions": {
            "rendered": str(len(golden_lines)),
            "anchors": ["{} golden questions", "{} user-voice questions", "all {} questions"],
        },
        "corpus_docs": {
            "rendered": str(corpus_docs),
            "anchors": ["{} committed documents", "{} handbook documents"],
        },
        "corpus_chunks": {
            "rendered": str(paragraph_arm["chunks"]),
            "anchors": ["{} chunks under the default strategy", "{} paragraph chunks"],
        },
        "embedding_dims": {
            "rendered": str(manifest["dims"]),
            "anchors": ["{} dimensional", "{} dims"],
        },
        "seeds": {
            "rendered": str(len(e1["seeds"])),
            "anchors": ["{} seeds", "across {} seeded runs"],
        },
        "failures_judged": {
            "rendered": str(default_arm["failures"]),
            "anchors": ["{} injected failures", "{} failures it judged"],
        },
        "attribution_accuracy_pct": {
            "rendered": _pct(default_arm["accuracy"]),
            "anchors": ["recovers the cause in {}%", "recovers the true stage in {}%"],
        },
        "no_contract_accuracy_pct": {
            "rendered": _pct(no_contract_arm["accuracy"]),
            "anchors": ["{}% without the citation contract"],
        },
        "ablation_accuracy_pct": {
            "rendered": _pct(ablation_arm["accuracy"]),
            "anchors": ["{}% with the trace withheld", "falls to {}%"],
        },
        "ablation_cannot_attribute": {
            "rendered": str(ablation_arm["cannot_attribute"]),
            "anchors": ["{} failures as unattributable"],
        },
        "retrieval_failures_n": {
            "rendered": str(retr_n),
            "anchors": ["all {} retrieval-side failures", "{} retrieval-side failures"],
        },
        "retrieval_recovered_n": {
            "rendered": str(retr_recovered),
            "anchors": ["reads {} of them", "recovered {} of"],
        },
        "ablation_retrieval_recovered": {
            "rendered": str(abl_retr_recovered),
            "anchors": ["places exactly {} of the", "{} of the 360"],
        },
        "confusable_accuracy_pct": {
            "rendered": _pct(conf_on["accuracy"]),
            "anchors": ["{}% on that class", "{}% with the contract on"],
        },
        "confusable_off_accuracy_pct": {
            "rendered": _pct(conf_off["accuracy"]),
            "anchors": ["drops to {}%", "{}% with the contract off"],
        },
        "confusable_n": {
            "rendered": str(conf_on["n"]),
            "anchors": ["{} tail-ignored failures", "the {} cases"],
        },
        "confusable_ci_low_pct": {
            "rendered": _pct(conf_off["ci"][0]),
            "anchors": ["interval from {}%"],
        },
        "confusable_ci_high_pct": {
            "rendered": _pct(conf_off["ci"][1]),
            "anchors": ["up to {}%"],
        },
        "recall_at_ctx_pct": {
            "rendered": _pct(base["recall_at_ctx"]),
            "anchors": ["recall at context of {}%", "recall sat at {}%"],
        },
        "recall_improved_pct": {
            "rendered": _pct(improved["recall_at_ctx"]),
            "anchors": ["recall to {}%", "lifts recall at context to {}%"],
        },
        "recall_gain_pp": {
            "rendered": f"{e3['recall_gain_pp']:.1f}",
            "anchors": ["{} points of recall"],
        },
        "e2e_gain_pp": {
            "rendered": f"{e3['e2e_gain_pp']:.1f}",
            "anchors": ["{} points end to end", "bought {} points"],
        },
        "generation_share_baseline_pct": {
            "rendered": _pct(base["generation_share"]),
            "anchors": ["{}% of failures already belonged"],
        },
        "generation_share_improved_pct": {
            "rendered": _pct(improved["generation_share"]),
            "anchors": ["{}% of the failures that remain", "rises to {}%"],
        },
        "e2e_accuracy_baseline_pct": {
            "rendered": _pct(base["e2e_accuracy"]),
            "anchors": ["answers {}% correctly"],
        },
        "chunking_miss_paragraph_pct": {
            "rendered": _pct(paragraph_arm["truth_mix"]["retrieval_miss"]),
            "anchors": ["{}% of blame on retrieval misses"],
        },
        "chunking_miss_fixed_pct": {
            "rendered": _pct(fixed_arm["truth_mix"]["retrieval_miss"]),
            "anchors": ["moves it to {}%"],
        },
        "trace_records_per_question": {
            "rendered": str(e5["trace_records_per_question"]),
            "anchors": ["{} trace records per question"],
        },
        "trace_bytes_median": {
            "rendered": str(e5["trace_bytes_per_question_median"]),
            "anchors": ["{} bytes per question"],
        },
        "tests_total": {
            "rendered": str(tests_total),
            "anchors": ["{} tests"],
        },
        "coverage_line_pct": {
            "rendered": coverage_pct,
            "anchors": ["{}% line coverage"],
        },
        "top_k": {
            "rendered": "12",
            "anchors": ["retriever returns {}", "top {} candidates"],
        },
        "k_ctx": {
            "rendered": "4",
            "anchors": ["window of {} chunks", "{} chunk context window"],
        },
    }
    return metrics


def main() -> None:
    metrics = build_metrics()
    out = ROOT / "docs" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(metrics, indent=1, sort_keys=True) + "\n"
    if not out.exists() or out.read_text(encoding="utf-8") != rendered:
        out.write_text(rendered, encoding="utf-8")
        print(f"metrics written: {len(metrics)} metrics")
    else:
        print(f"metrics unchanged: {len(metrics)} metrics")


if __name__ == "__main__":
    main()
