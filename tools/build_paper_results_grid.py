#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "paper_results_grid.png"

FIGURES = [
    ("fig_analytical_1_relay_power.png", "Outage vs relay power"),
    ("fig_analytical_2_bs_power.png", "Outage vs BS power"),
    ("fig_soft_bcd.png", "Soft-BCD convergence"),
    ("fig_relay_power_strategy.png", "Strategies vs relay power"),
    ("fig_bs_power_strategy.png", "Strategies vs BS power"),
    ("fig_c1_cache_size.png", "Strategies vs relay cache size"),
    ("fig_c2_cache_size.png", "Strategies vs BS cache size"),
    ("fig_eta_skewness.png", "Strategies vs MZipf skewness"),
    ("fig_tau_plateau.png", "Strategies vs MZipf plateau"),
]


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    cols, rows = 3, 3
    cell_w, cell_h = 760, 560
    title_h = 64
    margin = 28
    page_w = cols * cell_w + (cols + 1) * margin
    page_h = rows * (cell_h + title_h) + (rows + 1) * margin
    canvas = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(28)

    for idx, (filename, title) in enumerate(FIGURES):
        r, c = divmod(idx, cols)
        x0 = margin + c * (cell_w + margin)
        y0 = margin + r * (cell_h + title_h + margin)
        image = Image.open(RESULTS / filename).convert("RGB")
        image.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        box = draw.textbbox((0, 0), title, font=title_font)
        draw.text(
            (x0 + (cell_w - (box[2] - box[0])) // 2, y0),
            title,
            fill="black",
            font=title_font,
        )
        image_x = x0 + (cell_w - image.width) // 2
        image_y = y0 + title_h + (cell_h - image.height) // 2
        canvas.paste(image, (image_x, image_y))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
