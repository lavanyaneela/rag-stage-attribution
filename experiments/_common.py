"""Shared plumbing for the experiment scripts.

Experiments are capture steps: they write committed JSON under
docs/experiments/ and `make verify` re-derives every published figure from
those files without re-running the sweeps. Everything here is seeded, so two
machines capturing the same experiment write the same bytes, with one named
exception: wall time overhead in E5, which is gated behind a flag and
documented as a fact about the capture machine.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_stage_attribution.config import PipelineConfig
from rag_stage_attribution.injection import InjectionMix
from rag_stage_attribution.runner import SuiteResources, build_resources, run_suite

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "docs" / "experiments"
SEEDS = tuple(range(1, 31))

_RESOURCE_CACHE: dict[PipelineConfig, SuiteResources] = {}


def resources_for(cfg: PipelineConfig) -> SuiteResources:
    """Cache resources per config, keyed on the WHOLE frozen config.

    The first version of this cache keyed on a hand-picked subset of fields
    and left tail_slots out, so E2's widened-tail arms silently reran the
    baseline definition: run_suite reads cfg off the cached resources, not
    off the caller's argument. The tell was two experiment rows that were
    byte identical when they could not be. A frozen dataclass is hashable;
    there is no reason to summarise it by hand, and a test now pins this.
    """
    if cfg not in _RESOURCE_CACHE:
        _RESOURCE_CACHE[cfg] = build_resources(cfg)
    return _RESOURCE_CACHE[cfg]


def sweep(cfg: PipelineConfig, mix: InjectionMix, seeds=SEEDS, use_trace=True) -> list[dict]:
    res = resources_for(cfg)
    records: list[dict] = []
    for seed in seeds:
        records.extend(run_suite(res, mix, seed=seed, use_trace=use_trace))
    return records


def failures_of(records: list[dict]) -> list[dict]:
    return [r for r in records if not r["correct"] and r["judgeable"]]


def accuracy_over_failures(records: list[dict]) -> tuple[int, int]:
    failures = failures_of(records)
    recovered = sum(1 for r in failures if r.get("attributed") == r["truth"])
    return recovered, len(failures)


def write_results(name: str, payload: dict) -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPERIMENTS_DIR / name
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")
