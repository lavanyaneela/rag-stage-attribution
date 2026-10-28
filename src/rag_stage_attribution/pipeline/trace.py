"""Serialize traces and count what instrumentation costs, for E5.

Cost is published as counts and ratios: records per question and bytes per
question are exact and machine independent, so they can be anchored in prose.
Wall time overhead is a fact about the runner; it stays in the committed
capture at docs/experiments/e5_overhead.json and a test asserts the
inequality instead (receipts rule 6).
"""

from __future__ import annotations

import json

from rag_stage_attribution.pipeline.state import TraceRecord


def trace_to_payload(trace: list[TraceRecord]) -> list[dict]:
    return [{"node": r.node, "payload": r.payload} for r in trace]


def trace_bytes(trace: list[TraceRecord]) -> int:
    return len(json.dumps(trace_to_payload(trace), sort_keys=True).encode("utf-8"))
