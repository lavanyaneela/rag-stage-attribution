"""E2: the confusable class, measured at its own prevalence.

The class: ground truth is a generation fault, and the golden chunk sat in
the tail of the context window. ADR-004's tail rule makes this the designed
blind spot, so this experiment publishes accuracy on exactly that slice with
exact binomial intervals, under the citation contract, without it, and with
the tail definition widened one slot (the rank threshold is a definition,
and a definition you cannot vary is a definition you cannot defend).
"""

from __future__ import annotations

from _common import SEEDS, failures_of, sweep, write_results

from rag_stage_attribution.attribution import clopper_pearson
from rag_stage_attribution.config import PipelineConfig
from rag_stage_attribution.injection import InjectionMix


def confusable_slice(records: list[dict]) -> tuple[int, int]:
    rows = [
        r
        for r in failures_of(records)
        if r["truth"] == "generation" and r["golden_in_tail"]
    ]
    recovered = sum(1 for r in rows if r["attributed"] == r["truth"])
    return recovered, len(rows)


def arm(label: str, cfg: PipelineConfig, mix: InjectionMix) -> dict:
    records = sweep(cfg, mix)
    recovered, n = confusable_slice(records)
    failures = len(failures_of(records))
    low, high = clopper_pearson(recovered, n)
    return {
        "label": label,
        "accuracy": recovered / n,
        "recovered": recovered,
        "n": n,
        "ci": [low, high],
        "prevalence_among_failures": n / failures,
    }


def main() -> None:
    mix = InjectionMix()
    arms = [
        arm("contract on, tail is the last slot", PipelineConfig(), mix),
        arm("contract off, tail is the last slot", PipelineConfig(citation_contract=False), mix),
        arm("contract on, tail widened to two slots", PipelineConfig(tail_slots=2), mix),
        arm(
            "contract off, tail widened to two slots",
            PipelineConfig(citation_contract=False, tail_slots=2),
            mix,
        ),
    ]
    write_results("e2_results.json", {"mix": mix.as_dict(), "seeds": list(SEEDS), "arms": arms})


if __name__ == "__main__":
    main()
