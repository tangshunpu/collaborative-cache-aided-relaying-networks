#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "results" / "paper_data"
OUT_FILE = ROOT / "results" / "experiment_results.pdf"

FIGURES = [
    ("fig_analytical_1_relay_power.png", "Fig. 1: Outage probability versus relay transmit power."),
    ("fig_analytical_2_bs_power.png", "Fig. 2: Outage probability versus BS transmit power."),
    ("fig_soft_bcd.png", "Fig. 3: Soft-BCD convergence behavior."),
    ("fig_relay_power_strategy.png", "Fig. 4: Strategy comparison versus relay transmit power."),
    ("fig_bs_power_strategy.png", "Fig. 5: Strategy comparison versus BS transmit power."),
    ("fig_c1_cache_size.png", "Fig. 6: Strategy comparison versus relay cache size C1."),
    ("fig_c2_cache_size.png", "Fig. 7: Strategy comparison versus BS cache size C2."),
    ("fig_eta_skewness.png", "Fig. 8: Strategy comparison versus MZipf skewness eta."),
    ("fig_tau_plateau.png", "Fig. 9: Strategy comparison versus MZipf plateau tau."),
]


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _page() -> Image.Image:
    return Image.new("RGB", (1654, 2339), "white")


def _center_text(draw: ImageDraw.ImageDraw, y: int, text: str, font: ImageFont.ImageFont) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((1654 - (box[2] - box[0])) // 2, y), text, fill="black", font=font)


def title_page() -> Image.Image:
    page = _page()
    draw = ImageDraw.Draw(page)
    title = _font(54)
    body = _font(30)
    small = _font(24)
    _center_text(draw, 300, "Collaborative Cache-Aided Relaying Networks", title)
    _center_text(draw, 380, "Reproducible Experiment Results", _font(42))
    lines = [
        "Paper: Collaborative cache-aided relaying networks:",
        "Performance evaluation and system optimization",
        "Authors: Shunpu Tang, Ke He, Lunyuan Chen, Lisheng Fan,",
        "Xianfu Lei, and Rose Qingyang Hu",
        "IEEE Journal on Selected Areas in Communications, 41(3):706-719, 2023",
    ]
    y = 620
    for line in lines:
        _center_text(draw, y, line, body)
        y += 50
    _center_text(draw, 1180, "Generated from the Python reproduction scripts in this repository.", small)
    return page


def figure_page(filename: str, caption: str) -> Image.Image:
    page = _page()
    draw = ImageDraw.Draw(page)
    caption_font = _font(30)
    image = Image.open(FIGURE_DIR / filename).convert("RGB")
    max_w, max_h = 1350, 1000
    image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    x = (1654 - image.width) // 2
    y = 360
    page.paste(image, (x, y))
    _center_text(draw, 220, caption, caption_font)
    return page


def main() -> None:
    pages = [title_page()]
    pages.extend(figure_page(filename, caption) for filename, caption in FIGURES)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(OUT_FILE, save_all=True, append_images=pages[1:])
    print(OUT_FILE)


if __name__ == "__main__":
    main()
