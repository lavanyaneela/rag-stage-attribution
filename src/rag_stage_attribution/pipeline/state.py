"""The state that flows through the graph and the trace it accumulates.

The trace is the product. Attribution consumes nothing but these records plus
the golden label, so every field here is a field the attributor may legally
see. The injected mode is deliberately absent: it lives in the runner's
ledger, outside the trace, and a test enforces that separation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict


@dataclass(frozen=True)
class TraceRecord:
    node: str
    payload: dict[str, Any]


class PipelineState(TypedDict, total=False):
    qid: str
    question: str
    rewritten: str
    retrieved: list[tuple[str, float]]  # (chunk_id, dense score), retriever order
    context: list[tuple[str, float]]  # (chunk_id, blended score), rerank order
    answer: str
    citations: tuple[str, ...]
    trace: list[TraceRecord]
