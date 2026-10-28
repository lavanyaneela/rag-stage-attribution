"""Exact search, deterministic rerank: retrieval must be boringly predictable."""

import numpy as np
import pytest

from rag_stage_attribution.retrieval import FaissStore, rerank
from rag_stage_attribution.retrieval.rerank import content_words, lexical_overlap


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def test_search_returns_exact_cosine_order():
    rng = np.random.default_rng(7)
    vectors = _unit_rows(rng.normal(size=(50, 16)).astype(np.float32))
    store = FaissStore(vectors, [f"c{i}" for i in range(50)])
    query = vectors[13]
    hits = store.search(query, 5)
    assert hits[0][0] == "c13"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_caps_k_at_the_index_size():
    vectors = _unit_rows(np.eye(3, dtype=np.float32))
    store = FaissStore(vectors, ["a", "b", "c"])
    assert len(store.search(vectors[0], 10)) == 3


def test_mismatched_ids_and_vectors_are_rejected():
    vectors = _unit_rows(np.eye(3, dtype=np.float32))
    with pytest.raises(ValueError, match="disagree"):
        FaissStore(vectors, ["a", "b"])


def test_the_store_reports_its_size():
    vectors = _unit_rows(np.eye(4, dtype=np.float32))
    assert len(FaissStore(vectors, list("abcd"))) == 4


def test_lexical_overlap_is_the_share_of_query_content_words_found():
    assert lexical_overlap("receipt limit travel", "the receipt limit is high") == pytest.approx(
        2 / 3
    )


def test_stopwords_do_not_count_as_content():
    assert content_words("what is the and of") == frozenset()


def test_overlap_of_an_all_stopword_query_is_zero():
    assert lexical_overlap("what is the", "anything at all") == 0.0


def test_rerank_at_alpha_one_preserves_dense_order():
    candidates = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    texts = {"a": "nothing", "b": "receipt limit travel words", "c": "receipt limit"}
    out = rerank("receipt limit", candidates, texts, alpha=1.0)
    assert [cid for cid, _ in out] == ["a", "b", "c"]


def test_rerank_at_alpha_zero_orders_by_overlap_alone():
    candidates = [("a", 0.9), ("b", 0.1)]
    texts = {"a": "nothing relevant here", "b": "the receipt limit is written here"}
    out = rerank("receipt limit", candidates, texts, alpha=0.0)
    assert out[0][0] == "b"


def test_rerank_breaks_ties_by_the_original_dense_order():
    candidates = [("first", 0.5), ("second", 0.5)]
    texts = {"first": "same words", "second": "same words"}
    out = rerank("same words", candidates, texts, alpha=0.7)
    assert [cid for cid, _ in out] == ["first", "second"]


def test_rerank_is_a_pure_reordering_of_its_input():
    candidates = [("a", 0.4), ("b", 0.6), ("c", 0.5)]
    texts = {k: "text" for k in "abc"}
    out = rerank("query", candidates, texts, alpha=0.7)
    assert sorted(cid for cid, _ in out) == ["a", "b", "c"]
