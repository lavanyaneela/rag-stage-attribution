"""Seeded assignment of failure modes, which is what makes truth knowable.

A mode is drawn per (question, seed) from an explicit mix, with a stable hash
seed so the same run is the same run forever. Independence matters: whether a
question gets a generation fault is decided without looking at how retrieval
went for it, otherwise E1's claim that drift in one stage can occur without
the other would be built into the sampler rather than measured.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from rag_stage_attribution.generation.standin import MODES


@dataclass(frozen=True)
class InjectionMix:
    """Probabilities over stand-in modes; must sum to 1."""

    faithful: float = 0.55
    ignore: float = 0.25
    wrong_chunk: float = 0.15
    abstain: float = 0.05

    def __post_init__(self) -> None:
        total = self.faithful + self.ignore + self.wrong_chunk + self.abstain
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"mix must sum to 1, got {total}")
        if min(self.faithful, self.ignore, self.wrong_chunk, self.abstain) < 0:
            raise ValueError("mix probabilities must be non negative")

    def as_dict(self) -> dict[str, float]:
        return {
            "faithful": self.faithful,
            "ignore": self.ignore,
            "wrong_chunk": self.wrong_chunk,
            "abstain": self.abstain,
        }


def draw_mode(mix: InjectionMix, qid: str, seed: int) -> str:
    rng = random.Random(f"inject:{seed}:{qid}")
    roll = rng.random()
    cumulative = 0.0
    for mode in MODES:
        cumulative += getattr(mix, mode)
        if roll < cumulative:
            return mode
    return MODES[-1]
