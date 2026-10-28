"""Read the committed embedding fixture, and refuse one that drifted.

The offline rule means runtime never embeds anything: every vector a run can
possibly need was captured once and committed. That bargain only holds while
the fixture and the code agree on what the texts were, so loading verifies
the manifest's hashes of the corpus, the golden set, and the rewrite table
before handing out a single vector. A stale fixture answering fresh queries
is the quietest way this repository could lie.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rag_stage_attribution.corpus.chunking import Chunk
from rag_stage_attribution.corpus.loader import GoldenQuestion
from rag_stage_attribution.retrieval.rewrite import rewrite_table_sha


class FixtureError(RuntimeError):
    """The embedding fixture does not match the committed inputs."""


def text_digest(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


class EmbeddingStore:
    """Vectors for every chunk of every strategy and every golden query."""

    def __init__(self, embeddings_dir: Path) -> None:
        self.dir = embeddings_dir
        manifest_path = embeddings_dir / "manifest.json"
        if not manifest_path.exists():
            raise FixtureError(
                f"no embedding fixture at {embeddings_dir}; run 'make capture-embeddings'"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("rewrite_table_sha") != rewrite_table_sha():
            raise FixtureError(
                "rewrite table changed since the fixture was captured; query vectors are stale"
            )
        self._cache: dict[str, dict] = {}

    def _load(self, name: str) -> dict:
        if name not in self._cache:
            path = self.dir / f"{name}.npz"
            if not path.exists():
                raise FixtureError(f"fixture file missing: {path.name}")
            with np.load(path, allow_pickle=False) as z:
                self._cache[name] = {k: z[k] for k in z.files}
        return self._cache[name]

    def chunk_vectors(self, strategy: str, chunks: list[Chunk]) -> tuple[np.ndarray, list[str]]:
        data = self._load(f"chunks_{strategy}")
        ids = [i.decode() if isinstance(i, bytes) else str(i) for i in data["ids"]]
        if ids != [c.chunk_id for c in chunks]:
            raise FixtureError(f"chunk ids for {strategy!r} do not match the current chunker")
        expected = self.manifest["chunk_digests"][strategy]
        actual = text_digest([c.text for c in chunks])
        if expected != actual:
            raise FixtureError(f"chunk texts for {strategy!r} changed since capture")
        vectors = data["vectors"].astype(np.float32)
        _assert_unit(vectors, f"chunks_{strategy}")
        return vectors, ids

    def query_vector(self, qid: str) -> np.ndarray:
        data = self._load("queries")
        ids = [i.decode() if isinstance(i, bytes) else str(i) for i in data["ids"]]
        try:
            row = ids.index(qid)
        except ValueError:
            raise FixtureError(f"no captured query vector for {qid!r}") from None
        vec = data["vectors"][row].astype(np.float32)
        return vec

    def verify_queries(self, questions: list[GoldenQuestion]) -> None:
        from rag_stage_attribution.retrieval.rewrite import rewrite_query

        expected = self.manifest["query_digest"]
        actual = text_digest([rewrite_query(q.question) for q in questions])
        if expected != actual:
            raise FixtureError("golden questions changed since the fixture was captured")


def _assert_unit(vectors: np.ndarray, name: str) -> None:
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise FixtureError(f"{name}: vectors are not unit length; cosine scores would be wrong")
