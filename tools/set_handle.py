"""Fill in the GitHub handle and insert the live badges into the README.

Badges are inserted at handle time rather than shipped with a placeholder,
because a workflow badge pointing at a repository that does not exist yet
renders as "unknown", and an "unknown" badge above a careful README undoes
it. The badge URLs are built from one REPO constant so the spelling in
every URL is the repository name exactly, which a test asserts.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "rag-stage-attribution"
PLACEHOLDER = "GITHUB_USERNAME"
MARKER = "<!-- badges -->"

BADGE_TEMPLATE = (
    "[![ci](https://github.com/{handle}/" + REPO + "/actions/workflows/ci.yml/badge.svg)]"
    "(https://github.com/{handle}/" + REPO + "/actions/workflows/ci.yml)\n"
    "[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)"
)


def main() -> int:
    if len(sys.argv) != 2 or not re.fullmatch(r"[A-Za-z0-9-]{1,39}", sys.argv[1]):
        print("usage: set_handle.py <github-handle>")
        return 1
    handle = sys.argv[1]
    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    if MARKER in readme:
        readme = readme.replace(MARKER, BADGE_TEMPLATE.format(handle=handle), 1)
    readme = readme.replace(PLACEHOLDER, handle)
    readme_path.write_text(readme, encoding="utf-8")

    touched = [readme_path.name]
    for rel in ("docs/defense-guide.md", ".github/CODEOWNERS"):
        path = ROOT / rel
        if path.exists() and PLACEHOLDER in path.read_text(encoding="utf-8"):
            path.write_text(
                path.read_text(encoding="utf-8").replace(PLACEHOLDER, handle), encoding="utf-8"
            )
            touched.append(rel)
    print(f"handle set to {handle}; badges inserted; files touched: {', '.join(touched)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
