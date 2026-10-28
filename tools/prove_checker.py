"""Break one README figure on purpose and show the checker failing.

A checker nobody has seen fail is a checker nobody should believe, so CI
runs this before trusting a green `check_numbers`. It takes the real README,
substitutes a wrong value into the first anchored occurrence of the headline
metric, runs the forward pass on the broken copy, and exits 0 only if the
checker convicted it. Exit 1 means the checker let a false figure through,
which fails the build louder than any stale number could.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_numbers import ROOT, anchor_pattern, forward_pass, load_metrics  # noqa: E402

TARGET_METRIC = "attribution_accuracy_pct"
WRONG_VALUE = "12.3"


def main() -> int:
    metrics = load_metrics()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    metric = metrics[TARGET_METRIC]

    broken = None
    for anchor in metric["anchors"]:
        pattern = anchor_pattern(anchor)
        if pattern.search(readme):
            broken = pattern.sub(
                lambda m: m.group(0).replace(m.group(1), WRONG_VALUE), readme, count=1
            )
            break
    if broken is None:
        print(f"prove_checker: no anchored occurrence of {TARGET_METRIC} to break")
        return 1
    if broken == readme:
        print("prove_checker: substitution changed nothing; the proof is void")
        return 1

    problems = forward_pass(metrics, {"README.md": broken})
    convicted = [p for p in problems if TARGET_METRIC in p and WRONG_VALUE in p]
    if convicted:
        print("prove_checker: the checker failed the broken figure, as it must:")
        print(f"  {convicted[0]}")
        return 0
    print("prove_checker: the checker PASSED a deliberately broken figure")
    return 1


if __name__ == "__main__":
    sys.exit(main())
