"""Azure OpenAI shaped live adapter, behind a flag, for anyone with access.

This file is the one place a real model could enter the pipeline. It is not
imported anywhere on the default path, it needs the 'live' extra plus three
environment variables, and no number in this repository was produced through
it. What it would add is a measurement of one real model's failure mix on
this corpus; what it cannot add is ground truth, which only injection gives.
ADR-001 is the full argument. Excluded from coverage because offline CI can
never execute it; the structural tests in tests/test_adapters.py still hold
its prompt builder to the citation contract.
"""

from __future__ import annotations

import os

from rag_stage_attribution.corpus.loader import GoldenQuestion
from rag_stage_attribution.generation.standin import GenOutput

SYSTEM_PROMPT = (
    "Answer strictly from the provided context chunks. "
    "End your answer with a line 'CITATIONS: <chunk ids you used>' naming only "
    "chunks you actually drew on, or 'CITATIONS: none' if you used none."
)

REQUIRED_ENV = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT")


def build_messages(question: GoldenQuestion, context: list[tuple[str, str]]) -> list[dict]:
    blocks = "\n\n".join(f"[{cid}]\n{text}" for cid, text in context)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{blocks}\n\nQuestion: {question.question}"},
    ]


def parse_citations(answer: str, context_ids: set[str]) -> tuple[str, tuple[str, ...]]:
    body, _, tail = answer.rpartition("CITATIONS:")
    if not body:
        return answer.strip(), ()
    cited = tuple(tok for tok in tail.replace(",", " ").split() if tok in context_ids)
    return body.strip(), cited


class AzureOpenAIAdapter:
    def __init__(self) -> None:
        missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
        if missing:
            raise RuntimeError(f"live adapter needs environment variables: {missing}")
        from openai import AzureOpenAI  # the 'live' extra

        self._client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2024-06-01",
        )
        self._deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

    def generate(self, question: GoldenQuestion, context: list[tuple[str, str]]) -> GenOutput:
        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=build_messages(question, context),
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        answer, citations = parse_citations(raw, {cid for cid, _ in context})
        return GenOutput(answer, citations)
