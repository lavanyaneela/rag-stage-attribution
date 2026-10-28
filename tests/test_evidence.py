"""Evidence must be verifiable, so the evidence files are tested like code."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def test_the_architecture_svg_has_intrinsic_pixel_dimensions():
    svg = (ROOT / "docs" / "evidence" / "architecture.svg").read_text(encoding="utf-8")
    match = re.search(r'width="(\d+)" height="(\d+)"', svg)
    assert match, "SVG must carry explicit width and height, not 100%"
    assert 'width="100%"' not in svg


def test_the_architecture_svg_paints_a_light_background():
    svg = (ROOT / "docs" / "evidence" / "architecture.svg").read_text(encoding="utf-8")
    assert "background-color: white" in svg


def test_the_mermaid_source_is_committed_beside_the_render():
    assert (ROOT / "docs" / "evidence" / "architecture.mmd").exists()


def test_the_demo_manifest_lists_commands_with_their_exit_codes():
    manifest = json.loads(
        (ROOT / "docs" / "evidence" / "demo-manifest.json").read_text(encoding="utf-8")
    )
    segments = manifest["segments"]
    assert len(segments) == 6
    assert all("command" in s and "exit_code" in s for s in segments)
    # The refusal segment really exits 2 and the proof segment really exits 0.
    refusal = next(s for s in segments if "verdict --run /tmp/demo_tiny.json" in s["command"])
    assert refusal["exit_code"] == 2
    proof = next(s for s in segments if "prove_checker" in s["command"])
    assert proof["exit_code"] == 0


def test_every_manifest_command_appears_in_the_captured_transcript():
    manifest = json.loads(
        (ROOT / "docs" / "evidence" / "demo-manifest.json").read_text(encoding="utf-8")
    )
    capture = (ROOT / "docs" / "evidence" / "demo-capture.txt").read_text(encoding="utf-8")
    for segment in manifest["segments"]:
        assert f"$ {segment['command']}" in capture
        assert f"[exit {segment['exit_code']}]" in capture


def test_the_committed_gif_exists_and_is_not_bloated():
    gif = ROOT / "docs" / "evidence" / "demo.gif"
    assert gif.exists()
    assert gif.stat().st_size < 1_600_000


def test_badge_urls_spell_the_repository_name_exactly():
    import set_handle

    rendered = set_handle.BADGE_TEMPLATE.format(handle="someone")
    assert rendered.count("rag-stage-attribution") == 2
    assert "rag_stage_attribution" not in rendered  # underscore spelling would 404
    for url in re.findall(r"https://github\.com/[^)\s]+", rendered):
        assert "/rag-stage-attribution/" in url


def test_the_readme_carries_the_badge_marker_until_the_handle_is_set():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "<!-- badges -->" in readme
    assert "GITHUB_USERNAME" not in readme  # nothing renders a placeholder URL


def test_set_handle_inserts_badges_and_fills_the_handle(tmp_path, monkeypatch):
    import subprocess

    fake_root = tmp_path
    (fake_root / "tools").mkdir()
    (fake_root / "tools" / "set_handle.py").write_text(
        (ROOT / "tools" / "set_handle.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (fake_root / "README.md").write_text("# x\n\n<!-- badges -->\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(fake_root / "tools" / "set_handle.py"), "lavanyaneela"],
        capture_output=True,
        text=True,
    )
    readme = (fake_root / "README.md").read_text(encoding="utf-8")
    assert result.returncode == 0
    assert "actions/workflows/ci.yml/badge.svg" in readme
    assert "lavanyaneela/rag-stage-attribution" in readme
    assert "<!-- badges -->" not in readme


def test_set_handle_rejects_an_invalid_handle():
    import subprocess

    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "set_handle.py"), "not a handle"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1


def test_the_readme_example_fence_is_a_quotation_of_the_captured_transcript():
    """Pins audit finding 1: the README once showed a verdict share no run
    produces (a hand typed 73 where every real run prints 69), hidden in a
    code fence the reverse pass exempts. The fence is now held to the
    committed capture line by line, so an example output that drifts from
    the transcript fails the build instead of greeting a reviewer."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    capture = (ROOT / "docs" / "evidence" / "demo-capture.txt").read_text(encoding="utf-8")
    marker = "## What one run looks like"
    section = readme.split(marker, 1)[1]
    fence = section.split("```text", 1)[1].split("```", 1)[0]
    lines = [line for line in fence.splitlines() if line.strip()]
    assert len(lines) >= 4
    for line in lines:
        assert line in capture, f"README example line not in the captured transcript: {line!r}"
