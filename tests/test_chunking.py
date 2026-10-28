"""The chunker decides who gets blamed, so its properties are pinned here."""

import pytest

from rag_stage_attribution.config import CHUNKING_STRATEGIES
from rag_stage_attribution.corpus import CorpusError, chunk_corpus, golden_chunk_ids
from rag_stage_attribution.corpus.chunking import FIXED_STRIDE_WORDS, FIXED_WINDOW_WORDS


@pytest.mark.parametrize("strategy", CHUNKING_STRATEGIES)
def test_every_strategy_produces_unique_chunk_ids(docs, strategy):
    chunks = chunk_corpus(docs, strategy)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("strategy", CHUNKING_STRATEGIES)
def test_every_golden_span_is_covered_by_at_least_one_chunk(docs, golden, strategy):
    chunks = chunk_corpus(docs, strategy)
    uncovered = [q.qid for q in golden if not golden_chunk_ids(q, chunks)]
    assert uncovered == []


def test_paragraph_chunking_yields_one_chunk_per_paragraph(docs):
    chunks = chunk_corpus(docs, "paragraph")
    assert len(chunks) == sum(len(d.paragraphs) for d in docs)


def test_paragraph_golden_membership_is_a_single_chunk(docs, golden):
    chunks = chunk_corpus(docs, "paragraph")
    assert all(len(golden_chunk_ids(q, chunks)) == 1 for q in golden)


def test_window2_overlap_puts_interior_paragraphs_in_two_chunks(docs):
    doc = docs[0]
    chunks = [c for c in chunk_corpus(docs, "window2") if c.doc_id == doc.doc_id]
    interior = doc.paragraphs[1]
    holders = [c for c in chunks if interior in c.text]
    assert len(holders) == 2


def test_fixed_windows_overlap_enough_to_always_contain_a_span_sentence(docs, golden):
    overlap = FIXED_WINDOW_WORDS - FIXED_STRIDE_WORDS
    longest_span_words = max(len(q.span.split()) for q in golden)
    assert longest_span_words <= overlap + 1


def test_an_unknown_strategy_is_rejected(docs):
    with pytest.raises(CorpusError, match="unknown chunking"):
        chunk_corpus(docs, "sentences")


def test_a_span_no_chunk_contains_yields_an_empty_golden_set(docs, golden):
    from dataclasses import replace

    orphan = replace(golden[0], span="a sentence that exists in no chunk at all")
    chunks = chunk_corpus(docs, "paragraph")
    assert golden_chunk_ids(orphan, chunks) == frozenset()


def test_chunk_ids_carry_their_document_and_strategy_prefix(docs):
    for strategy, marker in (("paragraph", ":p"), ("window2", ":w"), ("fixed160", ":f")):
        chunks = chunk_corpus(docs, strategy)
        assert all(marker in c.chunk_id for c in chunks)


def test_every_chunk_keeps_its_document_title(docs):
    chunks = chunk_corpus(docs, "paragraph")
    titles = {d.doc_id: d.title for d in docs}
    assert all(c.title == titles[c.doc_id] for c in chunks)
