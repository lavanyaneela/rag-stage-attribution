"""The seeded generator stand-in and the paraphraser that gives it a voice."""

from rag_stage_attribution.generation.paraphrase import paraphrase
from rag_stage_attribution.generation.standin import ABSTAIN_TEXT, GenOutput, SeededStandIn

__all__ = ["ABSTAIN_TEXT", "GenOutput", "SeededStandIn", "paraphrase"]
