"""Exact dense retrieval over the committed vectors.

A flat inner product index is a deliberate choice, not a shortcut: with unit
vectors it is exact cosine search, so a retrieval miss measured here is a
property of the embedding space and the corpus, never of an approximation. An
approximate index would add a second suspect to every miss, and this
repository exists to count suspects precisely.
"""

from __future__ import annotations

import faiss
import numpy as np


class FaissStore:
    def __init__(self, vectors: np.ndarray, chunk_ids: list[str]) -> None:
        if vectors.ndim != 2 or vectors.shape[0] != len(chunk_ids):
            raise ValueError("vectors and chunk_ids disagree on count")
        self._ids = list(chunk_ids)
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def __len__(self) -> int:
        return self._index.ntotal

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if query.ndim == 1:
            query = query[None, :]
        k = min(top_k, len(self._ids))
        scores, rows = self._index.search(np.ascontiguousarray(query, dtype=np.float32), k)
        return [
            (self._ids[row], float(score))
            for row, score in zip(rows[0], scores[0], strict=True)
            if row != -1
        ]
