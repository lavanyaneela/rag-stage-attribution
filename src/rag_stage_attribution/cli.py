"""The command line interface, whose exit codes are the product.

Exit codes:
  0  attribution confident: the failing stage is named with its share, or
     every answer was correct and there was nothing to attribute
  1  both stages contribute materially; the tool worked, the answer is "both"
  2  refused: the trace cannot carry a verdict (golden coverage broken, too
     few attributable failures, or too many unattributable ones)
  3  tool error

Code 1 and code 2 are different answers on purpose: "both of you own this"
is a finding, "I cannot tell" is a confession, and a tool that folds them
together teaches its users to ignore it. Note that make folds every nonzero
recipe into its own exit 2, so automation should branch on this script, not
on a make target.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rag_stage_attribution.attribution.stats import verdict
from rag_stage_attribution.config import CHUNKING_STRATEGIES, PipelineConfig
from rag_stage_attribution.injection import InjectionMix
from rag_stage_attribution.logging import get_logger, log_event
from rag_stage_attribution.runner import build_resources, run_suite, summarise

LOGGER = get_logger("stageattr")


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    overrides = {}
    if getattr(args, "chunking", None):
        overrides["chunking"] = args.chunking
    if getattr(args, "k_ctx", None):
        overrides["k_ctx"] = args.k_ctx
    if getattr(args, "no_contract", False):
        overrides["citation_contract"] = False
    return PipelineConfig(**overrides)


def _mix_from_args(args: argparse.Namespace) -> InjectionMix:
    if getattr(args, "mix", None):
        parts = dict(item.split("=") for item in args.mix.split(","))
        return InjectionMix(**{k: float(v) for k, v in parts.items()})
    return InjectionMix()


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    mix = _mix_from_args(args)
    res = build_resources(cfg)
    all_records = []
    for seed in range(args.seed, args.seed + args.seeds):
        all_records.extend(run_suite(res, mix, seed=seed, use_trace=not args.no_trace))
    payload = {
        "config": {
            "chunking": cfg.chunking,
            "top_k": cfg.top_k,
            "k_ctx": cfg.k_ctx,
            "tail_slots": cfg.tail_slots,
            "citation_contract": cfg.citation_contract,
        },
        "mix": mix.as_dict(),
        "seeds": list(range(args.seed, args.seed + args.seeds)),
        "records": all_records,
        "summary": summarise(all_records),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    log_event(LOGGER, "run_written", path=str(out), **payload["summary"])
    return 0


def cmd_verdict(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.run).read_text(encoding="utf-8"))
    summary = payload["summary"]
    cfg = PipelineConfig(
        chunking=payload["config"]["chunking"],
        k_ctx=payload["config"]["k_ctx"],
        citation_contract=payload["config"]["citation_contract"],
    )
    result = verdict(
        cfg,
        summary["attributed_counts"],
        summary["cannot_attribute"],
        summary["unjudgeable"],
        summary["failures"],
    )
    print(f"verdict: {result.label}")
    print(f"detail:  {result.detail}")
    if result.shares:
        for stage in sorted(result.shares):
            low, high = result.bounds[stage]
            print(f"  {stage}: {result.shares[stage]:.1%} (95% CI {low:.1%} to {high:.1%})")
    return result.code


def cmd_attribute(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.run).read_text(encoding="utf-8"))
    rows = [
        r
        for r in payload["records"]
        if not r["correct"] and r["judgeable"] and (args.qid is None or r["qid"] == args.qid)
    ]
    if not rows:
        print("no failures matched")
        return 2
    for r in rows:
        print(
            f"{r['qid']} seed={r['seed']} attributed={r['attributed']} "
            f"confidence={r['confidence']:.2f} :: {r['reason']}"
        )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from rag_stage_attribution.reporting import write_report

    out = Path(args.out)
    write_report(out)
    log_event(LOGGER, "report_written", path=str(out))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stageattr", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the golden set under an injection mix")
    run.add_argument("--chunking", choices=CHUNKING_STRATEGIES)
    run.add_argument("--k-ctx", type=int, dest="k_ctx")
    run.add_argument("--mix", help="faithful=0.55,ignore=0.25,wrong_chunk=0.15,abstain=0.05")
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--seeds", type=int, default=1, help="number of consecutive seeds")
    run.add_argument("--no-contract", action="store_true", dest="no_contract")
    run.add_argument("--no-trace", action="store_true", dest="no_trace")
    run.add_argument("--out", required=True)
    run.set_defaults(fn=cmd_run)

    verd = sub.add_parser("verdict", help="exit 0/1/2 for a stored run")
    verd.add_argument("--run", required=True)
    verd.set_defaults(fn=cmd_verdict)

    attr = sub.add_parser("attribute", help="print per-failure attributions from a stored run")
    attr.add_argument("--run", required=True)
    attr.add_argument("--qid")
    attr.set_defaults(fn=cmd_attribute)

    rep = sub.add_parser("report", help="render the experiment report")
    rep.add_argument("--out", default="docs/experiments/report.md")
    rep.set_defaults(fn=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except Exception as exc:  # the broad net is the point: exit 3 means tool error
        if os.environ.get("RSA_DEBUG") == "1":
            raise
        print(f"stageattr: error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
