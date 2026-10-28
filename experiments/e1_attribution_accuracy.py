"""E1: does the attribution method work, and what does the trace buy?

Three arms over the same thirty seeds and the same default injection mix:
the default pipeline, the pipeline without the citation contract, and the
ablation where the attributor is denied the trace. The ablation is the
measured form of the opening impossibility: without node instrumentation,
never-retrieved and retrieved-but-ignored present identically, and the
accuracy collapse is the price of that.
"""

from __future__ import annotations

import json

from _common import (
    EXPERIMENTS_DIR,
    SEEDS,
    accuracy_over_failures,
    failures_of,
    sweep,
    write_results,
)

from rag_stage_attribution.config import PipelineConfig
from rag_stage_attribution.injection import InjectionMix
from rag_stage_attribution.runner import summarise


def per_class(records: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in failures_of(records):
        bucket = out.setdefault(r["truth"], {"recovered": 0, "n": 0})
        bucket["n"] += 1
        bucket["recovered"] += r.get("attributed") == r["truth"]
    return out


def main() -> None:
    mix = InjectionMix()
    arms = []

    default_cfg = PipelineConfig()
    default_records = sweep(default_cfg, mix)
    recovered, failures = accuracy_over_failures(default_records)
    arms.append(
        {
            "label": "default (contract on, trace on)",
            "accuracy": recovered / failures,
            "recovered": recovered,
            "failures": failures,
            "cannot_attribute": sum(
                1 for r in failures_of(default_records) if r["attributed"] == "cannot_attribute"
            ),
            "per_class": per_class(default_records),
        }
    )

    nc_records = sweep(PipelineConfig(citation_contract=False), mix)
    recovered, failures = accuracy_over_failures(nc_records)
    arms.append(
        {
            "label": "no citation contract",
            "accuracy": recovered / failures,
            "recovered": recovered,
            "failures": failures,
            "cannot_attribute": sum(
                1 for r in failures_of(nc_records) if r["attributed"] == "cannot_attribute"
            ),
            "per_class": per_class(nc_records),
        }
    )

    ablation_records = sweep(default_cfg, mix, use_trace=False)
    recovered, failures = accuracy_over_failures(ablation_records)
    arms.append(
        {
            "label": "no trace (ablation)",
            "accuracy": recovered / failures,
            "recovered": recovered,
            "failures": failures,
            "cannot_attribute": sum(
                1
                for r in failures_of(ablation_records)
                if r["attributed"] == "cannot_attribute"
            ),
            "per_class": per_class(ablation_records),
        }
    )

    write_results(
        "e1_results.json",
        {"mix": mix.as_dict(), "seeds": list(SEEDS), "arms": arms},
    )

    # The default run payload, kept for `stageattr verdict` and the demo.
    payload = {
        "config": {
            "chunking": default_cfg.chunking,
            "top_k": default_cfg.top_k,
            "k_ctx": default_cfg.k_ctx,
            "tail_slots": default_cfg.tail_slots,
            "citation_contract": default_cfg.citation_contract,
        },
        "mix": mix.as_dict(),
        "seeds": list(SEEDS),
        "records": default_records,
        "summary": summarise(default_records),
    }
    (EXPERIMENTS_DIR / "e1_runs.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("wrote e1_runs.json")


if __name__ == "__main__":
    main()
