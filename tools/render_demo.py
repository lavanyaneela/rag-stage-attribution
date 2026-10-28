"""Render the committed demo capture into a GIF, frame by replayed frame.

The GIF is a replay of captured stdout, not a recording of a staged
terminal: every character comes from docs/evidence/demo-capture.txt, which
tools/demo.sh wrote while running the real commands, and the manifest lists
each command with the exit code it really returned. Rendering uses PIL for
frames and ffmpeg's two pass palette for a small file. Rerun with:
python3 tools/render_demo.py (requires ffmpeg).
"""

from __future__ import annotations

import subprocess
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "docs" / "evidence" / "demo-capture.txt"
TARGET = ROOT / "docs" / "evidence" / "demo.gif"

COLS, ROWS = 100, 26
FONT_SIZE = 15
PAD = 14
FPS = 9

BG = (24, 26, 32)
FG = (214, 218, 224)
PROMPT = (126, 200, 141)
EXIT_OK = (126, 200, 141)
EXIT_BAD = (224, 140, 120)


def _font() -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", FONT_SIZE
    )


def wrap_lines(raw: str) -> list[str]:
    out: list[str] = []
    for line in raw.splitlines():
        if not line:
            out.append("")
            continue
        out.extend(textwrap.wrap(line, COLS, subsequent_indent="  ") or [""])
    return out


def render_frame(lines: list[str], font) -> Image.Image:
    char_w = font.getbbox("M")[2]
    line_h = FONT_SIZE + 6
    img = Image.new("RGB", (COLS * char_w + 2 * PAD, ROWS * line_h + 2 * PAD), BG)
    draw = ImageDraw.Draw(img)
    visible = lines[-ROWS:]
    for row, line in enumerate(visible):
        colour = FG
        if line.startswith("$ "):
            colour = PROMPT
        elif line.startswith("[exit 0]"):
            colour = EXIT_OK
        elif line.startswith("[exit"):
            colour = EXIT_BAD
        draw.text((PAD, PAD + row * line_h), line, font=font, fill=colour)
    return img


def main() -> None:
    font = _font()
    source_lines = wrap_lines(CAPTURE.read_text(encoding="utf-8"))
    frames: list[Image.Image] = []
    holds: list[int] = []
    shown: list[str] = []
    pending_plain = 0
    for line in source_lines:
        shown.append(line)
        is_marker = line.startswith("$ ") or line.startswith("[exit") or line == ""
        if not is_marker:
            pending_plain += 1
            if pending_plain < 2:
                continue  # batch plain output lines two per frame
        pending_plain = 0
        frames.append(render_frame(shown, font))
        if line.startswith("$ "):
            holds.append(8)  # linger on the command so a reader can read it
        elif line.startswith("[exit"):
            holds.append(10)
        elif line == "":
            holds.append(2)
        else:
            holds.append(3)
    holds[-1] = 30

    with tempfile.TemporaryDirectory() as tmp:
        index = 0
        for frame, hold in zip(frames, holds, strict=True):
            for _ in range(hold):
                frame.save(Path(tmp) / f"f{index:05d}.png")
                index += 1
        palette = Path(tmp) / "palette.png"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(FPS), "-i", f"{tmp}/f%05d.png",
                "-vf", "scale=800:-1:flags=lanczos,palettegen=max_colors=32",
                str(palette),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(FPS), "-i", f"{tmp}/f%05d.png",
                "-i", str(palette),
                "-lavfi", "scale=800:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=none",
                str(TARGET),
            ],
            check=True,
        )
    size_kb = TARGET.stat().st_size // 1024
    print(f"demo.gif: {len(frames)} distinct frames, {size_kb} KB")


if __name__ == "__main__":
    main()
