"""Run the golden set through the pipeline under an injection mix and judge it.

The runner holds the only ledger that knows the injected mode per question.
The attributor receives the trace and the golden label and nothing else; the
truth labeller receives the ledger. Keeping both calls inside one function is
what makes the separation auditable: grep this file for every argument each
side is handed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_stage_attribution.attribution.attributor import attribute
from rag_stage_attribution.attribution.truth import ground_truth
from rag_stage_attribution.config import PipelineConfig
from rag_stage_attribution.corpus import (
    chunk_corpus,
    golden_chunk_ids,
    load_corpus,
    load_golden,
    load_parametric,
)
from rag_stage_attribution.embeddings.store import EmbeddingStore
from rag_stage_attribution.generation.standin import SeededStandIn
from rag_stage_attribution.injection import InjectionMix, draw_mode
from rag_stage_attribution.pipeline.graph import PipelineResources, build_pipeline, run_question
from rag_stage_attribution.pipeline.trace import trace_bytes
from rag_stage_attribution.retrieval.faiss_store import FaissStore
from rag_stage_attribution.scoring import is_correct


@dataclass
class SuiteResources:
    """Everything a run needs, built once per (config, corpus)."""

    cfg: PipelineConfig
    golden: list
    golden_ids_by_qid: dict[str, frozenset[str]]
    chunk_texts: dict[str, str]
    store: FaissStore
    embeddings: EmbeddingStore
    bank: dict[str, dict]


def build_resources(cfg: PipelineConfig) -> SuiteResources:
    docs = load_corpus(cfg.corpus_dir)
    golden = load_golden(cfg.golden_path, docs)
    bank = load_parametric(cfg.parametric_path, golden)
    chunks = chunk_corpus(docs, cfg.chunking)
    embeddings = EmbeddingStore(cfg.embeddings_dir)
    embeddings.verify_queries(golden)
    vectors, ids = embeddings.chunk_vectors(cfg.chunking, chunks)
    return SuiteResources(
        cfg=cfg,
        golden=golden,
        golden_ids_by_qid={q.qid: golden_chunk_ids(q, chunks) for q in golden},
        chunk_texts={c.chunk_id: c.text for c in chunks},
        store=FaissStore(vectors, ids),
        embeddings=embeddings,
        bank=bank,
    )


def run_suite(
    res: SuiteResources,
    mix: InjectionMix,
    seed: int,
    use_trace: bool = True,
) -> list[dict[str, Any]]:
    """One pass over the golden set: pipeline, score, truth, attribution."""
    cfg = res.cfg
    standin = SeededStandIn(res.bank, cfg.citation_contract, seed)
    mode_by_qid = {q.qid: draw_mode(mix, q.qid, seed) for q in res.golden}
    question_by_qid = {q.qid: q for q in res.golden}

    def generate(question, context):
        return standin.generate(
            question, context, res.golden_ids_by_qid[question.qid], mode_by_qid[question.qid]
        )

    pipeline = build_pipeline(
        cfg,
        PipelineResources(
            store=res.store,
            chunk_texts=res.chunk_texts,
            query_vector=res.embeddings.query_vector,
            question_by_qid=question_by_qid,
            generate=generate,
        ),
    )

    records: list[dict[str, Any]] = []
    for question in res.golden:
        state = run_question(pipeline, question)
        golden_ids = res.golden_ids_by_qid[question.qid]
        retrieved_ids = [cid for cid, _ in state["retrieved"]]
        context_ids = [cid for cid, _ in state["context"]]
        correct = is_correct(state["answer"], question.answer)

        golden_position = min(
            (i for i, cid in enumerate(context_ids) if cid in golden_ids), default=None
        )
        record: dict[str, Any] = {
            "qid": question.qid,
            "seed": seed,
            "mode": mode_by_qid[question.qid],  # the ledger; never handed to attribute()
            "correct": correct,
            "judgeable": bool(golden_ids),
            "golden_position": golden_position,
            "golden_in_tail": golden_position is not None
            and golden_position >= cfg.k_ctx - cfg.tail_slots,
            "trace_bytes": trace_bytes(state["trace"]),
            "trace_records": len(state["trace"]),
        }
        if not correct and golden_ids:
            record["truth"] = ground_truth(
                golden_ids, retrieved_ids, context_ids, mode_by_qid[question.qid]
            )
            result = attribute(
                cfg,
                golden_ids,
                res.chunk_texts,
                state["answer"],
                state["citations"],
                retrieved_ids if use_trace else None,
                context_ids if use_trace else None,
            )
            record["attributed"] = result.stage
            record["confidence"] = round(result.confidence, 3)
            record["reason"] = result.reason
        records.append(record)
    return records


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts the verdict and the experiments both read."""
    failures = [r for r in records if not r["correct"] and r["judgeable"]]
    attributed = [r for r in failures if r.get("attributed") != "cannot_attribute"]
    correct_attr = [r for r in attributed if r["attributed"] == r["truth"]]
    counts: dict[str, int] = {}
    for r in attributed:
        counts[r["attributed"]] = counts.get(r["attributed"], 0) + 1
    truth_counts: dict[str, int] = {}
    for r in failures:
        truth_counts[r["truth"]] = truth_counts.get(r["truth"], 0) + 1
    return {
        "questions": len(records),
        "failures": len(failures),
        "unjudgeable": sum(1 for r in records if not r["judgeable"]),
        "cannot_attribute": len(failures) - len(attributed),
        "attributed_counts": counts,
        "truth_counts": truth_counts,
        "recovered": len(correct_attr),
        "accuracy_over_failures": (len(correct_attr) / len(failures)) if failures else None,
    }
