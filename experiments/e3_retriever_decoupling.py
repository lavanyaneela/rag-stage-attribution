"""E3: does improving the retriever help? Recall and outcomes, decoupled.

One knob turns: the context window widens from four chunks to six, which on
this corpus lifts recall at context to its ceiling. The generation fault
rate is injected identically in both arms, so whatever recall buys end to
end is measured against what it cannot buy: the share of failures that were
never retrieval's to begin with. Retrieval here is deterministic per
configuration, so recall is computed over questions; outcomes are computed
over question-seed runs.
"""

from __future__ import annotations

from _common import SEEDS, failures_of, resources_for, sweep, write_results

from rag_stage_attribution.config import PipelineConfig
from rag_stage_attribution.injection import InjectionMix


def arm(label: str, cfg: PipelineConfig, mix: InjectionMix) -> dict:
    res = resources_for(cfg)
    records = sweep(cfg, mix)
    # Retrieval is seed independent: golden_position is a function of config.
    by_qid = {}
    for r in records:
        by_qid.setdefault(r["qid"], r["golden_position"])
    recall_at_ctx = sum(1 for pos in by_qid.values() if pos is not None) / len(by_qid)
    correct = sum(1 for r in records if r["correct"])
    failures = failures_of(records)
    generation = sum(1 for r in failures if r["truth"] == "generation")
    return {
        "label": label,
        "k_ctx": cfg.k_ctx,
        "questions": len(res.golden),
        "recall_at_ctx": recall_at_ctx,
        "e2e_accuracy": correct / len(records),
        "failures": len(failures),
        "generation_share": generation / len(failures) if failures else 0.0,
    }


def main() -> None:
    mix = InjectionMix()
    baseline = arm("baseline retriever (k_ctx=4)", PipelineConfig(), mix)
    improved = arm("improved retriever (k_ctx=6)", PipelineConfig(k_ctx=6, tail_slots=1), mix)
    write_results(
        "e3_results.json",
        {
            "mix": mix.as_dict(),
            "seeds": list(SEEDS),
            "arms": [baseline, improved],
            "recall_gain_pp": 100.0 * (improved["recall_at_ctx"] - baseline["recall_at_ctx"]),
            "e2e_gain_pp": 100.0 * (improved["e2e_accuracy"] - baseline["e2e_accuracy"]),
        },
    )


if __name__ == "__main__":
    main()
