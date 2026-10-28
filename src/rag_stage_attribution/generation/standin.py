"""A generator stand-in whose failures are injected, which is the whole point.

A real LLM cannot run offline, and a measurement of attribution needs ground
truth about which stage failed. The stand-in resolves that tension: it is
driven by a seeded failure mode assigned per question, so every wrong answer
has a known cause. The stand-in is allowed to see the golden span, because it
simulates the model; the attributor is never allowed to see the injected
mode, because it is the judge. ADR-001 records this boundary in full.

Behaviour by mode:
  faithful     answers from the golden span when its chunk is in context, and
               otherwise does its honest best from the most relevant provided
               chunk, which is how a competent model fails on bad retrieval.
  ignore       answers from parametric memory (the committed wrong-belief
               bank) even when the golden chunk is sitting in the context.
  wrong_chunk  paraphrases a non-golden context chunk and cites it.
  abstain      declines, citing nothing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from rag_stage_attribution.corpus.loader import GoldenQuestion
from rag_stage_attribution.generation.paraphrase import paraphrase
from rag_stage_attribution.retrieval.rerank import lexical_overlap

ABSTAIN_TEXT = "I cannot find this in the provided context."

# How often the stand-in, when forced to cite, names the golden chunk it in
# fact ignored. A property of the simulated model, not of the method.
CITE_GOLDEN_ANYWAY = 0.3

MODES = ("faithful", "ignore", "wrong_chunk", "abstain")


@dataclass(frozen=True)
class GenOutput:
    answer: str
    citations: tuple[str, ...]  # chunk ids the generator claims it used


def _split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in text.replace("\n", " ").split(". ")]
    return [p if p.endswith(".") else p + "." for p in parts if p]


def _best_sentence(question: str, text: str) -> str:
    sentences = _split_sentences(text)
    return max(sentences, key=lambda s: lexical_overlap(question, s))


class SeededStandIn:
    """One instance per run; stateless across questions apart from its seed."""

    def __init__(self, bank: dict[str, dict], citation_contract: bool, seed: int) -> None:
        self._bank = bank
        self._contract = citation_contract
        self._seed = seed

    def _rng(self, qid: str, mode: str) -> random.Random:
        return random.Random(f"{self._seed}:{qid}:{mode}")

    def generate(
        self,
        question: GoldenQuestion,
        context: list[tuple[str, str]],  # (chunk_id, chunk_text) in rank order
        golden_ids: frozenset[str],
        mode: str,
    ) -> GenOutput:
        if mode not in MODES:
            raise ValueError(f"unknown stand-in mode {mode!r}")
        rng = self._rng(question.qid, mode)
        golden_in_ctx = [cid for cid, _ in context if cid in golden_ids]

        if mode == "abstain":
            return GenOutput(ABSTAIN_TEXT, ())

        if mode == "ignore":
            # Answers from memory sound like the policy with the wrong payload
            # in it, not like a different genre of text; that resemblance is
            # what makes ignoring a present chunk hard to detect lexically.
            belief = self._bank[question.qid]
            body = paraphrase(belief["statement"], rng, protect=belief["wrong_answer"])
            if body and body[0].isupper() and body[1:2].islower():
                body = body[0].lower() + body[1:]
            answer = f"{belief['preamble']} {body}"
            if self._contract and context:
                # A model told to cite will cite something. Sometimes it even
                # cites the golden chunk it ignored, which is the partial-use
                # headache ADR-004 names.
                if golden_in_ctx and rng.random() < CITE_GOLDEN_ANYWAY:
                    return GenOutput(answer, (golden_in_ctx[0],))
                candidates = [cid for cid, _ in context if cid not in golden_ids] or [
                    context[0][0]
                ]
                return GenOutput(answer, (rng.choice(candidates),))
            return GenOutput(answer, ())

        if mode == "wrong_chunk":
            candidates = [(cid, text) for cid, text in context if cid not in golden_ids]
            if not candidates:
                candidates = list(context)
            if not candidates:
                return GenOutput(ABSTAIN_TEXT, ())
            cid, text = candidates[rng.randrange(len(candidates))]
            sentence = _best_sentence(question.question, text)
            answer = paraphrase(sentence, rng)
            return GenOutput(answer, (cid,) if self._contract else ())

        # faithful
        if golden_in_ctx:
            answer = paraphrase(question.span, rng, protect=question.answer)
            cid = golden_in_ctx[0]
            return GenOutput(answer, (cid,) if self._contract else ())
        if not context:
            return GenOutput(ABSTAIN_TEXT, ())
        cid, text = max(context, key=lambda item: lexical_overlap(question.question, item[1]))
        sentence = _best_sentence(question.question, text)
        answer = "The closest guidance I can find: " + paraphrase(sentence, rng)
        return GenOutput(answer, (cid,) if self._contract else ())
