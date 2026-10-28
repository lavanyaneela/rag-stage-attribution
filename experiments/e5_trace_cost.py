"""E5: what node level instrumentation costs, and what it buys.

Costs are published as counts: trace records and bytes per question, which
are exact on every machine. Wall time overhead is a fact about the capture
machine, so measuring it is behind --measure-overhead: the default run keeps
this file byte reproducible, and the committed overhead figure is labelled
with the ratio only, never an absolute duration. What the trace buys is E1's
ablation delta; this file only prices the trace.
"""

from __future__ import annotations

import argparse
import statistics
import time

from _common import resources_for, write_results

from rag_stage_attribution.config import PipelineConfig
from rag_stage_attribution.generation.standin import SeededStandIn
from rag_stage_attribution.injection import InjectionMix, draw_mode
from rag_stage_attribution.pipeline.graph import PipelineResources, build_pipeline, run_question
from rag_stage_attribution.retrieval.rerank import rerank
from rag_stage_attribution.retrieval.rewrite import rewrite_query
from rag_stage_attribution.runner import run_suite


def bare_run(res, standin, mode_by_qid, cfg) -> None:
    """The pipeline as four plain calls: no graph, no trace records."""
    for q in res.golden:
        rewritten = rewrite_query(q.question)
        hits = res.store.search(res.embeddings.query_vector(q.qid), cfg.top_k)
        context = rerank(rewritten, hits, res.chunk_texts, cfg.rerank_alpha)[: cfg.k_ctx]
        ctx = [(cid, res.chunk_texts[cid]) for cid, _ in context]
        standin.generate(q, ctx, res.golden_ids_by_qid[q.qid], mode_by_qid[q.qid])


def graph_run(res, standin, mode_by_qid, cfg) -> None:
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
            question_by_qid={q.qid: q for q in res.golden},
            generate=generate,
        ),
    )
    for q in res.golden:
        run_question(pipeline, q)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure-overhead", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig()
    res = resources_for(cfg)
    records = run_suite(res, InjectionMix(), seed=1)
    sizes = sorted(r["trace_bytes"] for r in records)
    payload = {
        "trace_records_per_question": records[0]["trace_records"],
        "trace_bytes_per_question_median": sizes[len(sizes) // 2],
        "trace_bytes_per_question_min": sizes[0],
        "trace_bytes_per_question_max": sizes[-1],
        "questions": len(records),
    }
    write_results("e5_results.json", payload)

    if args.measure_overhead:
        mix = InjectionMix()
        standin = SeededStandIn(res.bank, cfg.citation_contract, 1)
        mode_by_qid = {q.qid: draw_mode(mix, q.qid, 1) for q in res.golden}
        bare_times, graph_times = [], []
        for _ in range(5):
            t0 = time.perf_counter()
            bare_run(res, standin, mode_by_qid, cfg)
            bare_times.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            graph_run(res, standin, mode_by_qid, cfg)
            graph_times.append(time.perf_counter() - t0)
        ratio = min(graph_times) / min(bare_times)
        write_results(
            "e5_overhead.json",
            {
                "overhead_ratio_best_of_5": round(ratio, 3),
                "note": (
                    "graph-with-trace wall time over bare-calls wall time, best of 5 on the "
                    "capture machine; the ratio is the only meaningful number here and the "
                    "spread is why (see the committed run list)"
                ),
                "graph_runs_s": [round(t, 4) for t in graph_times],
                "bare_runs_s": [round(t, 4) for t in bare_times],
                "spread_ratio": round(
                    (max(graph_times) - min(graph_times)) / statistics.median(graph_times), 3
                ),
            },
        )


if __name__ == "__main__":
    main()
