"""Deterministic rerank: blend dense similarity with lexical overlap.

The rerank node is the boundary between what the retriever found and what the
generator is shown, which makes it the node that creates the retrieval-rank
failure class: a chunk the retriever surfaced that the rerank pushes past the
context cutoff. The blend is simple on purpose; the point is that reordering
happens at a named, traced node rather than invisibly inside a prompt builder.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9$.]+")

_STOPWORDS = frozenset(
    [
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "does",
        "for", "from", "has", "how", "in", "is", "it", "its", "may", "must",
        "no", "not", "of", "on", "or", "per", "that", "the", "their", "this",
        "to", "was", "what", "when", "who", "with",
    ]
)


def content_words(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS)


def lexical_overlap(query: str, text: str) -> float:
    """Share of the query's content words present in the text, in [0, 1].

    Asymmetric containment rather than Jaccard: a long chunk should not be
    punished for containing more than the query asked about.
    """
    q = content_words(query)
    if not q:
        return 0.0
    t = content_words(text)
    return len(q & t) / len(q)


def rerank(
    query: str,
    candidates: list[tuple[str, float]],
    chunk_texts: dict[str, str],
    alpha: float,
) -> list[tuple[str, float]]:
    """Reorder (chunk_id, cosine) pairs by alpha * cosine + (1 - alpha) * overlap.

    Ties break on the original dense order, which keeps the function a pure
    reordering of its input and makes every rank in the trace reproducible.
    """
    dense_rank = {cid: i for i, (cid, _) in enumerate(candidates)}
    blended = [
        (cid, alpha * score + (1.0 - alpha) * lexical_overlap(query, chunk_texts[cid]))
        for cid, score in candidates
    ]
    blended.sort(key=lambda pair: (-pair[1], dense_rank[pair[0]]))
    return blended
