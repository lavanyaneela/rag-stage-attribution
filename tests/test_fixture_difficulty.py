"""The third war story, pinned: a corpus retrieval never fails against
measures nothing.

The first corpus was written in its own vocabulary, recall at context was
near total under every strategy, and E3 and E4 had nothing to show. The
committed golden set is user voice with real vocabulary drift plus six
pure distractor documents, and this file pins the property that redesign
bought: retrieval genuinely fails at a realistic rate, in every class the
attribution taxonomy names, under the default configuration.
"""

def test_the_corpus_is_hard_enough_that_retrieval_actually_fails(resources):
    cfg = resources.cfg
    missing = cut = tail = in_context = 0
    for q in resources.golden:
        golden = resources.golden_ids_by_qid[q.qid]
        hits = resources.store.search(resources.embeddings.query_vector(q.qid), cfg.top_k)
        from rag_stage_attribution.retrieval import rerank, rewrite_query

        ordered = [
            cid
            for cid, _ in rerank(
                rewrite_query(q.question), hits, resources.chunk_texts, cfg.rerank_alpha
            )
        ]
        position = min((ordered.index(c) for c in golden if c in ordered), default=None)
        if position is None:
            missing += 1
        elif position >= cfg.k_ctx:
            cut += 1
        else:
            in_context += 1
            if position >= cfg.k_ctx - cfg.tail_slots:
                tail += 1
    total = len(resources.golden)
    assert missing >= 2, "no retrieval misses at all: the corpus is too easy again"
    assert cut >= 4, "the rank-cut class is empty: the corpus is too easy again"
    assert tail >= 1, "nothing lands in the context tail: E2 has no subject"
    assert in_context / total < 0.95, "near-total recall: every retrieval experiment is vacuous"
    assert in_context / total > 0.5, "retrieval fails too often to study generation at all"


def test_user_voice_questions_share_little_vocabulary_with_their_spans(resources):
    from rag_stage_attribution.retrieval.rerank import content_words

    overlaps = []
    for q in resources.golden:
        question_words = content_words(q.question)
        span_words = content_words(q.span)
        if question_words:
            overlaps.append(len(question_words & span_words) / len(question_words))
    mean_overlap = sum(overlaps) / len(overlaps)
    assert mean_overlap < 0.5, f"questions mirror their spans (mean overlap {mean_overlap:.2f})"


def test_the_distractor_documents_carry_no_golden_spans(resources, docs):
    golden_doc_ids = {q.doc_id for q in resources.golden}
    distractors = [d for d in docs if d.doc_id not in golden_doc_ids]
    assert len(distractors) == 6
    for doc in distractors:
        normalised = " ".join(doc.text.split())
        for q in resources.golden:
            assert " ".join(q.span.split()) not in normalised, (doc.doc_id, q.qid)
