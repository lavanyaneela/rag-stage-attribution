"""The graph exists to make stage boundaries observable; prove it does."""

import json

from rag_stage_attribution.generation.standin import SeededStandIn
from rag_stage_attribution.pipeline import build_pipeline, run_question, trace_bytes
from rag_stage_attribution.pipeline.graph import PipelineResources
from rag_stage_attribution.pipeline.state import TraceRecord
from rag_stage_attribution.retrieval.rewrite import rewrite_query


def _pipeline(resources, seed=1, mode="faithful"):
    cfg = resources.cfg
    standin = SeededStandIn(resources.bank, cfg.citation_contract, seed)
    question_by_qid = {q.qid: q for q in resources.golden}

    def generate(question, context):
        return standin.generate(
            question, context, resources.golden_ids_by_qid[question.qid], mode
        )

    return build_pipeline(
        cfg,
        PipelineResources(
            store=resources.store,
            chunk_texts=resources.chunk_texts,
            query_vector=resources.embeddings.query_vector,
            question_by_qid=question_by_qid,
            generate=generate,
        ),
    )


def test_a_run_leaves_exactly_one_trace_record_per_node(resources):
    state = run_question(_pipeline(resources), resources.golden[0])
    assert [r.node for r in state["trace"]] == ["rewrite", "retrieve", "rerank", "generate"]


def test_the_rewrite_record_shows_the_query_before_and_after(resources, golden):
    question = next(q for q in golden if "pager" in q.question)
    state = run_question(_pipeline(resources), question)
    record = state["trace"][0]
    assert record.payload["query_in"] == question.question
    assert record.payload["query_out"] == rewrite_query(question.question)


def test_the_retrieve_record_carries_top_k_hits_in_score_order(resources, cfg):
    state = run_question(_pipeline(resources), resources.golden[0])
    hits = state["trace"][1].payload["hits"]
    assert len(hits) == cfg.top_k
    scores = [float(h["score"]) for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_trace_scores_are_fixed_width_strings(resources, cfg):
    """Pins the CI finding that motivated _trace_score: a rounded float
    serializes at value-dependent length, FAISS float32 scores wobble at
    the last digit across CPU generations, and the trace byte count (the
    only part of a score that reaches a committed file) changed between
    machines. Every serialized score must be a 6-decimal fixed width
    string, so the byte count no longer depends on which CPU measured."""
    state = run_question(_pipeline(resources), resources.golden[0])
    hits = state["trace"][1].payload["hits"]
    context = state["trace"][2].payload["context"]
    for entry in hits + context:
        score = entry["score"]
        assert isinstance(score, str)
        whole, _, frac = score.lstrip("-").partition(".")
        assert len(frac) == 6, score
        float(score)  # still parseable as a number


def test_the_rerank_record_names_both_the_context_and_the_cut(resources, cfg):
    state = run_question(_pipeline(resources), resources.golden[0])
    payload = state["trace"][2].payload
    assert len(payload["context"]) == cfg.k_ctx
    assert len(payload["cut"]) == cfg.top_k - cfg.k_ctx
    assert set(c["chunk_id"] for c in payload["context"]).isdisjoint(payload["cut"])


def test_the_generate_record_carries_the_answer_and_citations(resources):
    state = run_question(_pipeline(resources), resources.golden[0])
    payload = state["trace"][3].payload
    assert payload["answer"] == state["answer"]
    assert payload["citations"] == list(state["citations"])


def test_the_context_handed_to_generate_matches_the_rerank_record(resources):
    state = run_question(_pipeline(resources), resources.golden[0])
    rerank_ids = [c["chunk_id"] for c in state["trace"][2].payload["context"]]
    assert state["trace"][3].payload["context_ids"] == rerank_ids


def test_the_trace_never_contains_the_injected_mode(resources):
    # The attributor reads the trace; the injected mode is the ground truth it
    # is scored against. This test is the wall between them.
    state = run_question(_pipeline(resources, mode="ignore"), resources.golden[0])
    blob = json.dumps([{**{"node": r.node}, **r.payload} for r in state["trace"]])
    assert "ignore" not in blob
    assert "injected" not in blob
    assert "mode" not in blob


def test_trace_bytes_counts_the_serialised_trace(resources):
    state = run_question(_pipeline(resources), resources.golden[0])
    size = trace_bytes(state["trace"])
    assert size > 500
    assert isinstance(size, int)


def test_the_pipeline_is_deterministic_for_a_fixed_seed(resources):
    a = run_question(_pipeline(resources, seed=3), resources.golden[5])
    b = run_question(_pipeline(resources, seed=3), resources.golden[5])
    assert a["answer"] == b["answer"]
    assert a["context"] == b["context"]


def test_trace_records_are_immutable():
    record = TraceRecord("rewrite", {"query_in": "x"})
    try:
        record.node = "other"
        raised = False
    except AttributeError:
        raised = True
    assert raised
