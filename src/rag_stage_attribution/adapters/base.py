"""The adapter boundary between the pipeline and whatever generates text.

The pipeline's generate node accepts anything with this shape. The default is
the seeded stand-in, which is what every published number is about. The live
adapter in azure_openai.py has the same shape, exists behind an explicit flag
and the 'live' extra, and is never imported on the default path; a test
asserts the whole package works with that module unimportable. ADR-001 states
what running a live model would and would not add to the measurement.
"""

from __future__ import annotations

from typing import Protocol

from rag_stage_attribution.corpus.loader import GoldenQuestion
from rag_stage_attribution.generation.standin import GenOutput


class GeneratorAdapter(Protocol):
    def generate(
        self,
        question: GoldenQuestion,
        context: list[tuple[str, str]],
    ) -> GenOutput: ...
