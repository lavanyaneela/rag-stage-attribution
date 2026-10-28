"""Wire the pipeline as a LangGraph graph with one trace record per node.

The honest reason this is a graph rather than four function calls: attribution
needs the boundary between stages to be an observable place, and a graph node
is exactly that. Each node reads the state, does one job, and appends one
TraceRecord describing what crossed the boundary. ADR-002 admits the cost:
this is more machinery than a pipeline needs if nobody wants attribution.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph

from rag_stage_attribution.config import PipelineConfig
from rag_stage_attribution.corpus.loader import GoldenQuestion
from rag_stage_attribution.generation.standin import GenOutput
from rag_stage_attribution.pipeline.state import PipelineState, TraceRecord
from rag_stage_attribution.retrieval.faiss_store import FaissStore
from rag_stage_attribution.retrieval.rerank import rerank
from rag_stage_attribution.retrieval.rewrite import rewrite_query

# The generate node calls whatever honours this signature: the seeded stand-in
# by default, or the live adapter behind its flag (ADR-001).
GenerateFn = Callable[[GoldenQuestion, list[tuple[str, str]]], GenOutput]


@dataclass
class PipelineResources:
    store: FaissStore
    chunk_texts: dict[str, str]
    query_vector: Callable[[str], object]  # qid -> committed vector
    question_by_qid: dict[str, GoldenQuestion]
    generate: GenerateFn


def build_pipeline(cfg: PipelineConfig, res: PipelineResources):
    def rewrite_node(state: PipelineState) -> dict:
        rewritten = rewrite_query(state["question"])
        record = TraceRecord(
            "rewrite", {"query_in": state["question"], "query_out": rewritten}
        )
        return {"rewritten": rewritten, "trace": state.get("trace", []) + [record]}

    def retrieve_node(state: PipelineState) -> dict:
        hits = res.store.search(res.query_vector(state["qid"]), cfg.top_k)
        record = TraceRecord(
            "retrieve",
            {"top_k": cfg.top_k, "hits": [{"chunk_id": c, "score": round(s, 6)} for c, s in hits]},
        )
        return {"retrieved": hits, "trace": state["trace"] + [record]}

    def rerank_node(state: PipelineState) -> dict:
        ordered = rerank(state["rewritten"], state["retrieved"], res.chunk_texts, cfg.rerank_alpha)
        context = ordered[: cfg.k_ctx]
        record = TraceRecord(
            "rerank",
            {
                "k_ctx": cfg.k_ctx,
                "alpha": cfg.rerank_alpha,
                "context": [{"chunk_id": c, "score": round(s, 6)} for c, s in context],
                "cut": [c for c, _ in ordered[cfg.k_ctx :]],
            },
        )
        return {"context": context, "trace": state["trace"] + [record]}

    def generate_node(state: PipelineState) -> dict:
        question = res.question_by_qid[state["qid"]]
        ctx = [(cid, res.chunk_texts[cid]) for cid, _ in state["context"]]
        out = res.generate(question, ctx)
        record = TraceRecord(
            "generate",
            {
                "context_ids": [cid for cid, _ in ctx],
                "answer": out.answer,
                "citations": list(out.citations),
            },
        )
        return {
            "answer": out.answer,
            "citations": out.citations,
            "trace": state["trace"] + [record],
        }

    graph = StateGraph(PipelineState)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def run_question(compiled, question: GoldenQuestion) -> PipelineState:
    initial: PipelineState = {"qid": question.qid, "question": question.question, "trace": []}
    return compiled.invoke(initial)
