"""Render the architecture diagram to a committed SVG with intrinsic size.

GitHub does not reliably render a Mermaid fence inside an img context, and an
SVG whose width is 100% collapses in some READMEs, so the committed artifact
is an SVG with explicit pixel dimensions, generated from the committed .mmd
source by the mermaid CLI with htmlLabels off and a white background. Rerun
with: python3 tools/render_diagram.py (requires mmdc on PATH).
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "evidence" / "architecture.mmd"
TARGET = ROOT / "docs" / "evidence" / "architecture.svg"
TARGET_WIDTH = 1100


def main() -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump({"theme": "neutral", "flowchart": {"htmlLabels": False}}, handle)
        config = handle.name
    subprocess.run(
        ["mmdc", "-i", str(SOURCE), "-o", str(TARGET), "-c", config, "-b", "white"],
        check=True,
        capture_output=True,
    )
    svg = TARGET.read_text(encoding="utf-8")
    match = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    if not match:
        raise SystemExit("rendered SVG has no viewBox; refusing to commit it")
    width, height = float(match.group(1)), float(match.group(2))
    scaled_height = round(TARGET_WIDTH * height / width)
    svg = svg.replace(
        'width="100%"', f'width="{TARGET_WIDTH}" height="{scaled_height}"', 1
    )
    TARGET.write_text(svg, encoding="utf-8")
    print(f"architecture.svg: {TARGET_WIDTH}x{scaled_height} from viewBox {width:.0f}x{height:.0f}")


if __name__ == "__main__":
    main()
