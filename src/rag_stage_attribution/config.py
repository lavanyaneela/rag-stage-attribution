"""One frozen object carries every knob, so an experiment is its config.

Every threshold the attributor applies is a named field here rather than a
literal in a decision branch, because ADR-004 is a definitional document and a
definition that cannot be varied cannot be tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

CHUNKING_STRATEGIES = ("paragraph", "window2", "fixed160")


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for one pipeline run over the golden set."""

    # Retrieval
    chunking: str = "paragraph"
    top_k: int = 12  # retriever depth, what the retriever node returns
    k_ctx: int = 4  # context window, what the generate node is shown
    rerank_alpha: float = 0.7  # dense weight in the rerank blend

    # ADR-004: the last tail_slots positions of the context window are the
    # tail. A golden chunk sitting in the tail is the confusable territory
    # where "provided and ignored" and "barely not provided" meet, and the
    # attributor's blame rule treats tail presence as weak evidence.
    tail_slots: int = 1

    # Attributor evidence thresholds (lexical support is in [0, 1])
    support_used_floor: float = 0.28  # answer plausibly drew on this chunk
    support_margin_floor: float = 0.08  # margin below this is not evidence
    parametric_floor: float = 0.30  # answer plausibly came from memory

    # Generation contract
    citation_contract: bool = True

    # Verdict policy
    min_attributable: int = 12
    refuse_unattributable_share: float = 0.20

    # Determinism
    seed: int = 20260301

    # Paths, resolved relative to the repository root
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])

    def __post_init__(self) -> None:
        if self.chunking not in CHUNKING_STRATEGIES:
            raise ValueError(f"unknown chunking strategy: {self.chunking!r}")
        if not 1 <= self.k_ctx <= self.top_k:
            raise ValueError("k_ctx must satisfy 1 <= k_ctx <= top_k")
        if not 0 <= self.tail_slots < self.k_ctx:
            raise ValueError("tail_slots must satisfy 0 <= tail_slots < k_ctx")

    @property
    def corpus_dir(self) -> Path:
        return self.root / "fixtures" / "corpus"

    @property
    def golden_path(self) -> Path:
        return self.root / "fixtures" / "golden" / "questions.jsonl"

    @property
    def parametric_path(self) -> Path:
        return self.root / "fixtures" / "parametric" / "memory.jsonl"

    @property
    def embeddings_dir(self) -> Path:
        return self.root / "fixtures" / "embeddings"

    def with_(self, **kwargs) -> PipelineConfig:
        return replace(self, **kwargs)
