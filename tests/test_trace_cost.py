"""E5's duration claim lives here as an inequality, never as anchored prose.

Receipts rule: never anchor prose to a metric derived from a duration. The
committed capture file records the ratio measured once on the capture
machine; this test re-measures on whatever machine runs it and asserts only
the inequality that matters: the instrumented graph costs a bounded multiple
of the bare calls, not an unbounded one. The bound is deliberately loose
because co-tenant machines are noisy, and a tight bound here would be a
flaky test pretending to be a benchmark.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from e5_trace_cost import bare_run, graph_run  # noqa: E402

from rag_stage_attribution.generation.standin import SeededStandIn  # noqa: E402
from rag_stage_attribution.injection import InjectionMix, draw_mode  # noqa: E402


def test_the_committed_overhead_capture_is_a_ratio_with_its_run_list():
    capture = json.loads(
        (ROOT / "docs" / "experiments" / "e5_overhead.json").read_text(encoding="utf-8")
    )
    assert capture["overhead_ratio_best_of_5"] > 1.0
    assert len(capture["graph_runs_s"]) == 5 and len(capture["bare_runs_s"]) == 5


def test_the_instrumented_graph_costs_a_bounded_multiple_of_bare_calls(resources):
    cfg = resources.cfg
    standin = SeededStandIn(resources.bank, cfg.citation_contract, 1)
    mode_by_qid = {q.qid: draw_mode(InjectionMix(), q.qid, 1) for q in resources.golden}

    bare_best = min(
        _timed(bare_run, resources, standin, mode_by_qid, cfg) for _ in range(3)
    )
    graph_best = min(
        _timed(graph_run, resources, standin, mode_by_qid, cfg) for _ in range(3)
    )
    ratio = graph_best / bare_best
    assert ratio < 30.0, f"instrumentation overhead ratio {ratio:.1f} is out of hand"


def _timed(fn, *args) -> float:
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start
