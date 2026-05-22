from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def apply_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.0,
            "lines.markersize": 8,
            "figure.figsize": (4.6, 3.45),
            "figure.dpi": 160,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "eps"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def save_csv(rows: list[dict[str, object]], out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def plot_strategy_curves(
    rows: list[dict[str, object]],
    out_dir: Path,
    stem: str,
    x_key: str,
    x_label: str,
    y_label: str = "Outage probability",
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
) -> None:
    apply_ieee_style()
    fig, ax = plt.subplots()
    styles = {
        "proposed": ("r-o", "Proposed"),
        "mpc": ("b-s", "MPC"),
        "epc": ("k-v", "EPC"),
    }
    for relays in sorted({int(row["relays"]) for row in rows}):
        subset = [row for row in rows if int(row["relays"]) == relays]
        x = [float(row[x_key]) for row in subset]
        for key, (style, label) in styles.items():
            ax.semilogy(x, [float(row[key]) for row in subset], style, label=label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if ylim:
        ax.set_ylim(*ylim)
    if xlim:
        ax.set_xlim(*xlim)
    ax.legend()
    ax.grid(False)
    save_figure(fig, out_dir, stem)


def plot_analysis_curves(
    rows: list[dict[str, object]],
    out_dir: Path,
    stem: str,
    x_label: str,
) -> None:
    apply_ieee_style()
    fig, ax = plt.subplots()
    color_by_n = {2: "b", 3: "r", 4: "k", 5: "m"}
    marker_by_n = {2: "*", 3: "^", 4: "o", 5: "d"}
    for relays in (2, 4, 3):
        subset = [row for row in rows if int(row["relays"]) == relays]
        if not subset:
            continue
        x = [float(row["power_dbm"]) for row in subset]
        c = color_by_n[relays]
        marker = marker_by_n[relays]
        ax.semilogy(x, [float(row["simulation"]) for row in subset], f"{c}-", label="Simulation")
        ax.semilogy(x, [float(row["analytical"]) for row in subset], f"{c}{marker}", label="Analytical")
        ax.semilogy(x, [float(row["asymptotic"]) for row in subset], f"{c}--", label="Asymptotic")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Outage probability")
    ax.set_ylim(1e-5, 1)
    ax.legend()
    ax.grid(False)
    save_figure(fig, out_dir, stem)
