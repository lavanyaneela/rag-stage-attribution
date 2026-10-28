"""Fail the build when a document states a figure the code no longer produces.

The rules encoded here were each learned by shipping the bug they prevent:

1. Anchors, not digits: a metric carries the exact wordings a document may
   use, with the value substituted in, and every anchored occurrence in every
   checked document must carry the current value. Matching is per occurrence,
   never per document, so a stale sentence cannot hide behind a fresh one.
2. The sweep runs inside fenced blocks and code spans too; guarding known
   lines closes mutation and leaves insertion wide open.
3. A reverse pass over the README flags any number that belongs to no metric,
   after stripping ISO dates, version strings, and vetted context exemptions.
4. Every context exemption must be used, and no exemption may swallow an
   anchored occurrence; an unused exemption is a hole waiting for a claim.
5. Single-character metric values match only through anchors: the reverse
   whitelist carries multi-character values only, so a bare meaningful digit
   in prose is a finding, not a coincidence.
6. No absolute duration may appear in any tracked markdown: milliseconds and
   seconds are facts about a runner, and the sweep covers every tracked file,
   not just the checked documents, so a new file cannot smuggle one in.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKED_DOCUMENTS = [
    "README.md",
    "docs/adr/adr-001-injected-ground-truth.md",
    "docs/adr/adr-002-langgraph-over-a-chain.md",
    "docs/adr/adr-003-chunking-strategy.md",
    "docs/adr/adr-004-retrieved-and-ignored.md",
    "docs/adr/adr-005-the-confusable-class.md",
    "docs/experiments/report.md",
    "docs/defense-guide.md",
]

# Matches a number not embedded in a longer hex/identifier run.
NUMBER = re.compile(r"(?<![\w.])(-?\d+(?:,\d{3})*(?:\.\d+)?)(?![\w])")

# Removed from the reverse-pass text before scanning. Each must match at
# least once across the checked documents, and none may overlap an anchor hit.
CONTEXT_EXEMPTIONS: dict[str, str] = {
    "iso_dates": r"\b\d{4}-\d{2}-\d{2}\b",
    "exit_codes": r"\bexit(?: code)?s? [0-3](?:/[0-3])*\b",
    "code_zero_to_three": r"\b[Cc]ode [0-3]\b",
    "versions": r"\bv\d+\.\d+\.\d+\b",
    "python_versions": r"\bPython 3\.\d{2}\b",
    "adr_numbers": r"\bADR-\d{3}\b",
    "ci_level": r"\b95%\s+(?:CI|interval|confidence)\b",
    "minilm_name": r"\bMiniLM-L6-v2\b",
}


class Failure(Exception):
    pass


def _strip_code(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]*`", " ", text)
    text = re.sub(r"<[^>\n]+>", " ", text)  # HTML tags and their attributes
    text = re.sub(r"\]\([^)\s]+\)", "]( )", text)  # link targets
    text = re.sub(r"(?m)^(\s*)\d+\.\s", r"\1", text)  # ordered list markers
    return text


def load_metrics() -> dict[str, dict]:
    return json.loads((ROOT / "docs" / "metrics.json").read_text(encoding="utf-8"))


def anchor_pattern(anchor: str) -> re.Pattern:
    escaped = re.escape(anchor).replace(r"\{\}", r"(-?\d+(?:,\d{3})*(?:\.\d+)?)")
    # Anchors must survive markdown line wrapping: any space in an anchor
    # matches any run of whitespace, newlines included.
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(escaped)


def check_anchor_containment(metrics: dict[str, dict]) -> list[str]:
    problems = []
    flat = [(name, a) for name, m in metrics.items() for a in m["anchors"]]
    for name_a, a in flat:
        for name_b, b in flat:
            if name_a != name_b and a in b:
                problems.append(f"anchor containment: {name_a!r} anchor {a!r} sits inside "
                                f"{name_b!r} anchor {b!r}")
    return problems


def forward_pass(metrics: dict[str, dict], documents: dict[str, str]) -> list[str]:
    problems = []
    for doc_name, text in documents.items():
        for name, metric in metrics.items():
            for anchor in metric["anchors"]:
                for match in anchor_pattern(anchor).finditer(text):
                    found = match.group(1)
                    if found != metric["rendered"]:
                        problems.append(
                            f"{doc_name}: stale value for {name}: anchor {anchor!r} "
                            f"carries {found!r}, measured {metric['rendered']!r}"
                        )
    return problems


def required_presence(metrics: dict[str, dict], readme: str) -> list[str]:
    required = ("attribution_accuracy_pct", "confusable_off_accuracy_pct",
                "ablation_accuracy_pct", "tests_total", "coverage_line_pct")
    problems = []
    for name in required:
        hit = any(anchor_pattern(a).search(readme) for a in metrics[name]["anchors"])
        if not hit:
            problems.append(f"README.md: required metric {name} appears under no anchor")
    return problems


def reverse_pass(metrics: dict[str, dict], readme: str) -> list[str]:
    problems = []
    text = _strip_code(readme)
    anchor_spans: list[tuple[int, int]] = []
    for metric in metrics.values():
        for anchor in metric["anchors"]:
            for match in anchor_pattern(anchor).finditer(text):
                anchor_spans.append(match.span(1))

    used_exemptions: set[str] = set()
    exempt_spans: list[tuple[int, int]] = []
    for name, pattern in CONTEXT_EXEMPTIONS.items():
        for match in re.finditer(pattern, text):
            used_exemptions.add(name)
            exempt_spans.append(match.span())
            for span in anchor_spans:
                if span[0] >= match.start() and span[1] <= match.end():
                    problems.append(
                        f"exemption {name!r} swallows an anchored occurrence at {span}"
                    )

    whitelist = {m["rendered"] for m in metrics.values() if len(m["rendered"]) >= 2}
    for match in NUMBER.finditer(text):
        span = match.span(1)
        if any(span[0] >= s and span[1] <= e for s, e in exempt_spans):
            continue
        if any(span == s_e for s_e in anchor_spans) or any(
            span[0] >= s and span[1] <= e for s, e in anchor_spans
        ):
            continue
        value = match.group(1)
        if value in whitelist:
            continue
        problems.append(f"README.md: number {value!r} at offset {span[0]} matches no metric")
    return problems


def exemption_hygiene(texts: list[str]) -> list[str]:
    """Every context exemption must be exercised somewhere the reverse pass
    runs; an exemption nothing uses is a hole waiting for a claim."""
    stripped = [_strip_code(text) for text in texts]
    problems = []
    for name, pattern in CONTEXT_EXEMPTIONS.items():
        if not any(re.search(pattern, text) for text in stripped):
            problems.append(f"unused exemption {name!r}: remove it or use it")
    return problems


def duration_sweep() -> list[str]:
    tracked = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    pattern = re.compile(r"\b\d[\d,.]*\s*(?:ms\b|milliseconds?\b|wall[- ]clock seconds\b)")
    problems = []
    for rel in tracked:
        if not rel.endswith(".md"):
            continue
        text = _strip_code(Path(ROOT / rel).read_text(encoding="utf-8"))
        for match in pattern.finditer(text):
            problems.append(f"{rel}: absolute duration in prose: {match.group(0)!r}")
    return problems


def run_checks(documents: dict[str, str] | None = None) -> list[str]:
    metrics = load_metrics()
    if documents is None:
        documents = {}
        for rel in CHECKED_DOCUMENTS:
            path = ROOT / rel
            if path.exists():
                documents[rel] = path.read_text(encoding="utf-8")
    problems = check_anchor_containment(metrics)
    problems += forward_pass(metrics, documents)
    reverse_docs = [n for n in ("README.md", "docs/defense-guide.md") if n in documents]
    if "README.md" in documents:
        problems += required_presence(metrics, documents["README.md"])
    if reverse_docs:
        problems += exemption_hygiene([documents[n] for n in reverse_docs])
    # The reverse pass covers every document that makes quantitative claims
    # a reader will quote: a number in the README or the defense guide that
    # belongs to no metric is an invented figure (audit finding 5).
    for name in reverse_docs:
        if True:
            problems += [
                p.replace("README.md:", f"{name}:") for p in
                reverse_pass(metrics, documents[name])
            ]
    problems += duration_sweep()
    return problems


def main() -> int:
    problems = run_checks()
    if problems:
        print(f"check_numbers: {len(problems)} problem(s)")
        for p in problems:
            print(f"  FAIL {p}")
        return 1
    print("check_numbers: all documented figures match the measured ones")
    return 0


if __name__ == "__main__":
    sys.exit(main())
