"""E4: does the chunking strategy move the blame?

Same corpus, same questions, same injected generator; only the chunker
changes. The failure mix that comes out is the measured version of a claim
teams usually make from taste: chunking is not a preprocessing detail, it
decides which stage of your pipeline appears to be broken.
"""

from __future__ import annotations

from _common import SEEDS, failures_of, resources_for, sweep, write_results

from rag_stage_attribution.config import CHUNKING_STRATEGIES, PipelineConfig
from rag_stage_attribution.injection import InjectionMix


def main() -> None:
    mix = InjectionMix()
    arms = []
    for strategy in CHUNKING_STRATEGIES:
        cfg = PipelineConfig(chunking=strategy)
        res = resources_for(cfg)
        records = sweep(cfg, mix)
        failures = failures_of(records)
        mix_counts = {"retrieval_miss": 0, "retrieval_rank": 0, "generation": 0}
        for r in failures:
            mix_counts[r["truth"]] += 1
        total = len(failures)
        arms.append(
            {
                "strategy": strategy,
                "chunks": len(res.chunk_texts),
                "failures": total,
                "truth_mix": {k: v / total for k, v in mix_counts.items()},
                "truth_counts": mix_counts,
            }
        )
    write_results("e4_results.json", {"mix": mix.as_dict(), "seeds": list(SEEDS), "arms": arms})


if __name__ == "__main__":
    main()
