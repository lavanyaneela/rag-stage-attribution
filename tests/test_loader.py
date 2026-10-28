"""Loading is where a contradictory fixture must die, not downstream."""

import pytest

from rag_stage_attribution.corpus import CorpusError, load_corpus, load_golden, load_parametric
from tests.conftest import rewrite_jsonl


def test_the_committed_corpus_loads_with_titles_and_paragraphs(docs):
    assert len(docs) == 22
    assert all(doc.title for doc in docs)
    assert all(len(doc.paragraphs) >= 3 for doc in docs)


def test_every_golden_span_appears_exactly_once_in_its_document(golden):
    assert len(golden) == 73


def test_a_document_without_a_title_line_is_rejected(corpus_copy):
    bad = corpus_copy / "corpus" / "expense-policy.md"
    bad.write_text("no title here\n\npara one\n\npara two\n\npara three\n")
    with pytest.raises(CorpusError, match="title"):
        load_corpus(corpus_copy / "corpus")


def test_a_document_with_too_few_paragraphs_is_rejected(corpus_copy):
    bad = corpus_copy / "corpus" / "expense-policy.md"
    bad.write_text("# Expenses\n\nonly one paragraph\n")
    with pytest.raises(CorpusError, match="paragraphs"):
        load_corpus(corpus_copy / "corpus")


def test_an_empty_corpus_directory_is_rejected(tmp_path):
    with pytest.raises(CorpusError, match="no corpus documents"):
        load_corpus(tmp_path)


def test_a_golden_span_that_is_not_in_its_document_is_rejected(corpus_copy):
    docs = load_corpus(corpus_copy / "corpus")
    path = corpus_copy / "golden" / "questions.jsonl"
    rewrite_jsonl(path, lambda rows: [{**rows[0], "span": "this sentence is nowhere"}] + rows[1:])
    with pytest.raises(CorpusError, match="occurs 0 times"):
        load_golden(path, docs)


def test_a_golden_span_duplicated_in_the_document_is_rejected(corpus_copy):
    docs_dir = corpus_copy / "corpus"
    doc = docs_dir / "expense-policy.md"
    first_span = "Itemized receipts are required for any single expense above $25."
    doc.write_text(doc.read_text() + "\n\n" + first_span + " Padding. More padding.\n")
    docs = load_corpus(docs_dir)
    with pytest.raises(CorpusError, match="occurs 2 times"):
        load_golden(corpus_copy / "golden" / "questions.jsonl", docs)


def test_a_golden_answer_that_is_not_inside_its_span_is_rejected(corpus_copy):
    docs = load_corpus(corpus_copy / "corpus")
    path = corpus_copy / "golden" / "questions.jsonl"
    rewrite_jsonl(path, lambda rows: [{**rows[0], "answer": "$999"}] + rows[1:])
    with pytest.raises(CorpusError, match="not a substring"):
        load_golden(path, docs)


def test_a_duplicate_qid_is_rejected(corpus_copy):
    docs = load_corpus(corpus_copy / "corpus")
    path = corpus_copy / "golden" / "questions.jsonl"
    rewrite_jsonl(path, lambda rows: rows + [rows[0]])
    with pytest.raises(CorpusError, match="duplicate qid"):
        load_golden(path, docs)


def test_a_golden_question_for_an_unknown_document_is_rejected(corpus_copy):
    docs = load_corpus(corpus_copy / "corpus")
    path = corpus_copy / "golden" / "questions.jsonl"
    rewrite_jsonl(path, lambda rows: [{**rows[0], "doc_id": "ghost-doc"}] + rows[1:])
    with pytest.raises(CorpusError, match="unknown doc_id"):
        load_golden(path, docs)


def test_an_empty_golden_set_is_rejected(corpus_copy):
    docs = load_corpus(corpus_copy / "corpus")
    path = corpus_copy / "golden" / "questions.jsonl"
    path.write_text("")
    with pytest.raises(CorpusError, match="empty"):
        load_golden(path, docs)


def test_the_parametric_bank_covers_every_golden_question(bank, golden):
    assert set(bank) == {q.qid for q in golden}


def test_a_parametric_statement_containing_the_true_answer_is_rejected(corpus_copy, golden):
    docs = load_corpus(corpus_copy / "corpus")
    loaded = load_golden(corpus_copy / "golden" / "questions.jsonl", docs)
    path = corpus_copy / "parametric" / "memory.jsonl"
    truth = loaded[0]

    def poison(rows):
        rows[0] = {**rows[0], "statement": truth.span, "qid": truth.qid}
        return rows

    rewrite_jsonl(path, poison)
    with pytest.raises(CorpusError, match="contains the true answer"):
        load_parametric(path, loaded)


def test_a_parametric_entry_for_an_unknown_qid_is_rejected(corpus_copy):
    docs = load_corpus(corpus_copy / "corpus")
    loaded = load_golden(corpus_copy / "golden" / "questions.jsonl", docs)
    path = corpus_copy / "parametric" / "memory.jsonl"
    rewrite_jsonl(path, lambda rows: [{**rows[0], "qid": "q999"}] + rows[1:])
    with pytest.raises(CorpusError, match="unknown qid"):
        load_parametric(path, loaded)


def test_a_parametric_bank_missing_a_question_is_rejected(corpus_copy):
    docs = load_corpus(corpus_copy / "corpus")
    loaded = load_golden(corpus_copy / "golden" / "questions.jsonl", docs)
    path = corpus_copy / "parametric" / "memory.jsonl"
    rewrite_jsonl(path, lambda rows: rows[1:])
    with pytest.raises(CorpusError, match="missing qids"):
        load_parametric(path, loaded)
