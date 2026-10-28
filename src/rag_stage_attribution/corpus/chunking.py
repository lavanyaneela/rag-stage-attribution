"""Three chunking strategies, because the chunker decides who gets blamed.

E4 measures how the failure mix moves when only this file's choice changes.
The golden label is a span sentence, not a chunk id, precisely so that the
same golden set stays valid under every strategy: the golden chunks for a
question are recomputed per strategy as the chunks that contain its span.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_stage_attribution.corpus.loader import CorpusError, Document, GoldenQuestion

FIXED_WINDOW_WORDS = 160
FIXED_STRIDE_WORDS = 120


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str


def _paragraph_chunks(doc: Document) -> list[Chunk]:
    return [
        Chunk(f"{doc.doc_id}:p{i:02d}", doc.doc_id, doc.title, para)
        for i, para in enumerate(doc.paragraphs)
    ]


def _window2_chunks(doc: Document) -> list[Chunk]:
    """Two paragraphs per chunk, stride one: every interior paragraph appears
    in exactly two chunks, which is the overlap that makes golden membership a
    set rather than a single id."""
    chunks = []
    for i in range(len(doc.paragraphs) - 1):
        text = doc.paragraphs[i] + "\n\n" + doc.paragraphs[i + 1]
        chunks.append(Chunk(f"{doc.doc_id}:w{i:02d}", doc.doc_id, doc.title, text))
    return chunks


def _fixed_chunks(doc: Document) -> list[Chunk]:
    """Fixed word windows that ignore paragraph boundaries. This is the
    strategy that can split a span's sentence away from its context, and E4
    exists to price that."""
    words = doc.text.split()
    chunks = []
    start = 0
    idx = 0
    while start < len(words):
        window = words[start : start + FIXED_WINDOW_WORDS]
        chunks.append(Chunk(f"{doc.doc_id}:f{idx:02d}", doc.doc_id, doc.title, " ".join(window)))
        if start + FIXED_WINDOW_WORDS >= len(words):
            break
        start += FIXED_STRIDE_WORDS
        idx += 1
    return chunks


_STRATEGIES = {
    "paragraph": _paragraph_chunks,
    "window2": _window2_chunks,
    "fixed160": _fixed_chunks,
}


def chunk_corpus(docs: list[Document], strategy: str) -> list[Chunk]:
    try:
        cut = _STRATEGIES[strategy]
    except KeyError:
        raise CorpusError(f"unknown chunking strategy {strategy!r}") from None
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(cut(doc))
    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        raise CorpusError(f"strategy {strategy!r} produced duplicate chunk ids")
    return chunks


def golden_chunk_ids(question: GoldenQuestion, chunks: list[Chunk]) -> frozenset[str]:
    """Chunks whose text contains the golden span's answer sentence.

    Containment is checked on a whitespace-normalised form because the fixed
    window strategy rejoins words without paragraph breaks. An empty result
    means this strategy cut the span apart; the runner routes such a question
    to the refusal path, since retrieval cannot be judged against a golden
    chunk that does not exist.
    """
    normalised_span = " ".join(question.span.split())
    hits = frozenset(
        c.chunk_id for c in chunks if normalised_span in " ".join(c.text.split())
    )
    return hits
