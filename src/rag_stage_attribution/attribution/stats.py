"""Exact binomial bounds and the verdict policy built on them.

Shares of blame are proportions of a small sample, so every published share
travels with a Clopper-Pearson interval, and "one stage is the culprit" is a
claim about non overlapping intervals rather than about two point estimates.
A measured zero is priced at the worst rate the sample supports (receipts
rule 4): the interval does that automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import beta

from rag_stage_attribution.config import PipelineConfig


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact two sided (lower, upper) bounds for a binomial proportion."""
    if not 0 <= k <= n or n <= 0:
        raise ValueError(f"need 0 <= k <= n with n > 0, got k={k} n={n}")
    alpha = 1.0 - conf
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lower, upper


@dataclass(frozen=True)
class Verdict:
    code: int  # the CLI exit code: 0 decided, 1 both, 2 refused
    label: str
    detail: str
    shares: dict[str, float]
    bounds: dict[str, tuple[float, float]]
    counts: dict[str, int]


def verdict(
    cfg: PipelineConfig,
    attributed_counts: dict[str, int],
    cannot_count: int,
    unjudgeable_count: int,
    failures_total: int,
) -> Verdict:
    """Turn a run's attribution counts into the tool's answer.

    Stage shares are computed over attributable failures, with the retrieval
    subclasses pooled against generation for the culprit question: miss and
    rank are both retrieval's to fix. Refusal is the tool saying the evidence
    cannot carry a verdict, which is a different thing from "both".
    """
    counts = dict(attributed_counts)
    counts.setdefault("retrieval_miss", 0)
    counts.setdefault("retrieval_rank", 0)
    counts.setdefault("generation", 0)

    if unjudgeable_count > 0:
        return Verdict(
            2,
            "refused",
            f"{unjudgeable_count} question(s) have no golden chunk in the corpus, so "
            "retrieval cannot be judged for them",
            {},
            {},
            counts,
        )
    if failures_total == 0:
        return Verdict(0, "no_failures", "every answer scored correct; nothing to attribute",
                       {}, {}, counts)

    attributable = sum(counts.values())
    if attributable < cfg.min_attributable:
        return Verdict(
            2,
            "refused",
            f"only {attributable} attributable failure(s); the policy floor is "
            f"{cfg.min_attributable}",
            {},
            {},
            counts,
        )
    cannot_share = cannot_count / (attributable + cannot_count)
    if cannot_share > cfg.refuse_unattributable_share:
        return Verdict(
            2,
            "refused",
            f"{cannot_share:.0%} of failures could not be attributed, above the "
            f"{cfg.refuse_unattributable_share:.0%} refusal line",
            {},
            {},
            counts,
        )

    retrieval = counts["retrieval_miss"] + counts["retrieval_rank"]
    generation = counts["generation"]
    shares = {"retrieval": retrieval / attributable, "generation": generation / attributable}
    bounds = {
        "retrieval": clopper_pearson(retrieval, attributable),
        "generation": clopper_pearson(generation, attributable),
    }
    leader, runner = sorted(shares, key=shares.get, reverse=True)
    if bounds[leader][0] > bounds[runner][1]:
        return Verdict(
            0,
            leader,
            f"{leader} owns {shares[leader]:.0%} of attributable failures and its lower "
            f"bound clears {runner}'s upper bound",
            shares,
            bounds,
            counts,
        )
    return Verdict(
        1,
        "both",
        "both stages contribute materially; the intervals overlap, so no single culprit",
        shares,
        bounds,
        counts,
    )
