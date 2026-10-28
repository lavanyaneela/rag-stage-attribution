"""The store must refuse a drifted fixture, because a stale vector lies quietly."""

import json
import shutil

import numpy as np
import pytest

from rag_stage_attribution.corpus import chunk_corpus
from rag_stage_attribution.embeddings import EmbeddingStore, FixtureError


@pytest.fixture()
def fixture_copy(tmp_path, cfg):
    dest = tmp_path / "embeddings"
    shutil.copytree(cfg.embeddings_dir, dest)
    return dest


def test_the_committed_fixture_loads_and_matches_every_strategy(cfg, docs):
    store = EmbeddingStore(cfg.embeddings_dir)
    for strategy in ("paragraph", "window2", "fixed160"):
        chunks = chunk_corpus(docs, strategy)
        vectors, ids = store.chunk_vectors(strategy, chunks)
        assert vectors.shape == (len(chunks), store.manifest["dims"])
        assert ids == [c.chunk_id for c in chunks]


def test_the_committed_query_vectors_match_the_golden_set(cfg, golden):
    store = EmbeddingStore(cfg.embeddings_dir)
    store.verify_queries(golden)
    vec = store.query_vector(golden[0].qid)
    assert vec.shape == (store.manifest["dims"],)


def test_a_missing_fixture_directory_names_the_capture_command(tmp_path):
    with pytest.raises(FixtureError, match="capture-embeddings"):
        EmbeddingStore(tmp_path)


def test_a_changed_rewrite_table_is_detected(fixture_copy):
    manifest = json.loads((fixture_copy / "manifest.json").read_text())
    manifest["rewrite_table_sha"] = "0000000000000000"
    (fixture_copy / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(FixtureError, match="rewrite table changed"):
        EmbeddingStore(fixture_copy)


def test_changed_chunk_text_is_detected_through_the_digest(fixture_copy, docs):
    manifest = json.loads((fixture_copy / "manifest.json").read_text())
    manifest["chunk_digests"]["paragraph"] = "feedfeedfeedfeed"
    (fixture_copy / "manifest.json").write_text(json.dumps(manifest))
    store = EmbeddingStore(fixture_copy)
    with pytest.raises(FixtureError, match="changed since capture"):
        store.chunk_vectors("paragraph", chunk_corpus(docs, "paragraph"))


def test_a_reordered_chunker_is_detected_through_the_ids(fixture_copy, docs):
    store = EmbeddingStore(fixture_copy)
    chunks = list(reversed(chunk_corpus(docs, "paragraph")))
    with pytest.raises(FixtureError, match="do not match the current chunker"):
        store.chunk_vectors("paragraph", chunks)


def test_changed_questions_are_detected_through_the_query_digest(fixture_copy, golden):
    from dataclasses import replace

    store = EmbeddingStore(fixture_copy)
    mutated = [replace(golden[0], question="something entirely new")] + list(golden[1:])
    with pytest.raises(FixtureError, match="questions changed"):
        store.verify_queries(mutated)


def test_a_query_vector_for_an_unknown_qid_is_refused(cfg):
    store = EmbeddingStore(cfg.embeddings_dir)
    with pytest.raises(FixtureError, match="no captured query vector"):
        store.query_vector("q999")


def test_non_unit_vectors_are_refused(fixture_copy, docs):
    with np.load(fixture_copy / "chunks_paragraph.npz") as z:
        ids, vectors = z["ids"], z["vectors"] * 3.0
    np.savez_compressed(fixture_copy / "chunks_paragraph.npz", ids=ids, vectors=vectors)
    # Digest still matches, which is exactly why the norm check must be separate.
    store = EmbeddingStore(fixture_copy)
    with pytest.raises(FixtureError, match="unit length"):
        store.chunk_vectors("paragraph", chunk_corpus(docs, "paragraph"))


def test_a_missing_strategy_file_is_named(fixture_copy, docs):
    (fixture_copy / "chunks_window2.npz").unlink()
    store = EmbeddingStore(fixture_copy)
    with pytest.raises(FixtureError, match="chunks_window2.npz"):
        store.chunk_vectors("window2", chunk_corpus(docs, "window2"))
