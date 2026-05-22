#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "lines.markersize": 4.5,
            "figure.dpi": 180,
        }
    )


def _analysis_panel(ax: plt.Axes, csv_name: str, title: str) -> None:
    df = pd.read_csv(RESULTS / csv_name)
    df = df[df["relays"] == 3]
    x = df["power_dbm"]
    ax.semilogy(x, df["analytical"], "r-o", label="Analytical")
    ax.semilogy(x, df["simulation"], "b-s", label="Simulation")
    ax.semilogy(x, df["asymptotic"], "k--^", label="Asymptotic")
    ax.set_title(title)
    ax.set_xlabel("Transmit power (dBm)")
    ax.set_ylabel("Outage probability")
    ax.set_ylim(1e-4, 1)
    ax.grid(True, which="both", ls=":", lw=0.4)


def _strategy_panel(ax: plt.Axes, csv_name: str, title: str, x_col: str, x_label: str) -> None:
    df = pd.read_csv(RESULTS / csv_name)
    styles = {
        "proposed": ("r-o", "Proposed"),
        "mpc": ("b-s", "MPC"),
        "epc": ("k-v", "EPC"),
    }
    for relays in sorted(df["relays"].unique()):
        part = df[df["relays"] == relays]
        for col, (style, label) in styles.items():
            suffix = f", N={relays}" if relays == sorted(df["relays"].unique())[0] else ""
            ax.semilogy(part[x_col], part[col], style, label=label + suffix)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Outage probability")
    ax.grid(True, which="both", ls=":", lw=0.4)


def main() -> None:
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.6), constrained_layout=True)

    _analysis_panel(
        axes[0, 0],
        "fig_analytical_1_relay_power.csv",
        "(a) Analysis check: relay power, N=3",
    )
    _analysis_panel(
        axes[0, 1],
        "fig_analytical_2_bs_power.csv",
        "(b) Analysis check: BS power, N=3",
    )
    _strategy_panel(
        axes[0, 2],
        "fig_relay_power_strategy.csv",
        "(c) Caching strategy vs relay power",
        "power_dbm",
        "Relay power (dBm)",
    )
    _strategy_panel(
        axes[1, 0],
        "fig_bs_power_strategy.csv",
        "(d) Caching strategy vs BS power",
        "power_dbm",
        "BS power (dBm)",
    )
    _strategy_panel(
        axes[1, 1],
        "fig_c1_cache_size.csv",
        "(e) Caching strategy vs relay cache size",
        "c1",
        "Relay cache size C1",
    )
    _strategy_panel(
        axes[1, 2],
        "fig_c2_cache_size.csv",
        "(f) Caching strategy vs BS cache size",
        "c2",
        "BS cache size C2",
    )

    handles, labels = [], []
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l, strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
        ax.legend().remove()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False)

    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"key_results_summary.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(RESULTS / "key_results_summary.png")


if __name__ == "__main__":
    main()
