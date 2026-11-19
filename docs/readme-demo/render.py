"""Render the README demo GIFs (manual run, not part of CI).

Each scene is an HTML page that exposes `window.seek(t)` and draws one frame for time t
(seconds). This script seeks every frame, screenshots it with Playwright and builds a GIF
with Pillow. The same input always gives the same GIF.

    python docs/readme-demo/render.py            # all scenes
    python docs/readme-demo/render.py hero       # one scene
    python docs/readme-demo/render.py hero --at 2,5,8   # PNGs at these seconds, no GIF

One-time setup: `pip install playwright pillow && python -m playwright install chromium`.
"""

from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "assets" / "readme"
CHECK = HERE / "_check"  # scratch PNGs from --at (git-ignored)
MAX_MB = 5.0


@dataclass(frozen=True)
class Scene:
    html: str
    seconds: float
    still_at: float
    w: int = 1200
    h: int = 675
    fps: int = 12
    static: bool = False  # one PNG instead of a GIF


SCENES = {
    "hero": Scene("hero.html", seconds=17, still_at=15.8),
    "bands": Scene("bands.html", seconds=12, still_at=11),
    "evidence": Scene("evidence.html", seconds=12, still_at=11),
    "results": Scene("results.html", seconds=1, still_at=0, h=540, static=True),
    "pipeline": Scene("pipeline.html", seconds=15, still_at=14),
}


def shoot(page, t: float) -> Image.Image:
    page.evaluate("t => window.seek(t)", t)
    return Image.open(io.BytesIO(page.screenshot(type="png"))).convert("RGB")


def build_gif(frames: list[Image.Image], fps: int, path: Path) -> None:
    # One shared palette (sampled over the whole clip) so colours do not flicker between frames.
    step = max(1, len(frames) // 12)
    sample = frames[::step]
    sheet = Image.new("RGB", (frames[0].width, frames[0].height * len(sample)))
    for i, f in enumerate(sample):
        sheet.paste(f, (0, i * f.height))
    palette = sheet.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    quant = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    quant[0].save(
        path,
        save_all=True,
        append_images=quant[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=False,
        disposal=1,
    )


def render(name: str, scene: Scene, check: list[float] | None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CHECK.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": scene.w, "height": scene.h})
        page.goto((HERE / scene.html).as_uri())
        page.set_viewport_size({"width": scene.w, "height": scene.h})
        page.wait_for_function("window.ready === true")
        page.evaluate("document.fonts.ready")
        if check is not None:
            for t in check or (0.0, scene.still_at):
                shoot(page, t).save(CHECK / f"{name}-check-{t:g}.png")
            browser.close()
            return
        if scene.static:
            shoot(page, scene.still_at).save(OUT / f"{name}.png")
            browser.close()
            sys.stdout.write(f"{name}: {(OUT / f'{name}.png').stat().st_size / 1e3:.0f} kB\n")
            return
        n = round(scene.seconds * scene.fps)
        frames = [shoot(page, i / scene.fps) for i in range(n)]
        browser.close()
    build_gif(frames, scene.fps, OUT / f"{name}.gif")
    mb = (OUT / f"{name}.gif").stat().st_size / 1e6
    flag = "  OVER LIMIT" if mb > MAX_MB else ""
    sys.stdout.write(f"{name}: {n} frames, {mb:.2f} MB{flag}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", help=f"any of: {', '.join(SCENES)} (default: all)")
    ap.add_argument(
        "--at",
        type=lambda v: [float(x) for x in v.split(",")],
        help="write PNGs at these seconds, no GIF",
    )
    args = ap.parse_args()
    unknown = [s for s in args.scenes if s not in SCENES]
    if unknown:
        ap.error(f"unknown scene: {', '.join(unknown)}")
    for name in args.scenes or SCENES:
        render(name, SCENES[name], args.at)
    return 0


if __name__ == "__main__":
    sys.exit(main())
