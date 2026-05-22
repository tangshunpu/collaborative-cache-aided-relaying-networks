from __future__ import annotations

from pathlib import Path

import numpy as np

from .model import (
    SystemConfig,
    analytical_mode_outages,
    asymptotic_mode_outages,
    dbm_to_snr,
    monte_carlo_mode_outages,
    mzip_popularity,
    system_outage,
)
from .plotting import (
    apply_ieee_style,
    plot_analysis_curves,
    plot_strategy_curves,
    save_csv,
    save_figure,
)
from .strategies import (
    bcd_objective_history,
    epc_strategy,
    mpc_strategy,
    proposed_soft_bcd,
)


POWER_GRID = [20, 25, 30, 35, 40, 45]


def _strategy_point(
    relays: int,
    relay_power_dbm: float,
    bs_power_dbm: float,
    relay_cache: int,
    bs_cache: int,
    tau: float,
    eta: float,
    cfg: SystemConfig,
) -> dict[str, float]:
    popularity = mzip_popularity(cfg.files, tau=tau, eta=eta)
    modes = analytical_mode_outages(
        relays,
        dbm_to_snr(relay_power_dbm, cfg.noise_power_w),
        dbm_to_snr(bs_power_dbm, cfg.noise_power_w),
        cfg,
    )
    q_mpc_1, q_mpc_2 = mpc_strategy(cfg.files, relay_cache, bs_cache)
    q_epc_1, q_epc_2 = epc_strategy(cfg.files, relays, relay_cache, bs_cache)
    q_prop_1, q_prop_2, history = proposed_soft_bcd(
        popularity, relay_cache, bs_cache, relays, modes
    )
    return {
        "proposed": system_outage(q_prop_1, q_prop_2, popularity, relays, modes),
        "mpc": system_outage(q_mpc_1, q_mpc_2, popularity, relays, modes),
        "epc": system_outage(q_epc_1, q_epc_2, popularity, relays, modes),
        "bcd_iterations": len(history),
    }


def analysis_power_sweep(
    out_dir: Path,
    sweep: str,
    samples: int = 200_000,
) -> list[dict[str, object]]:
    out_dir = Path(out_dir)
    cfg = SystemConfig()
    rows: list[dict[str, object]] = []

    popularity = mzip_popularity(cfg.files, cfg.mzip_tau, cfg.mzip_eta)
    for relays in (2, 3, 4):
        for power in POWER_GRID:
            relay_dbm = float(power if sweep == "relay" else 45.0)
            bs_dbm = float(power if sweep == "bs" else 45.0)
            snr_r = dbm_to_snr(relay_dbm, cfg.noise_power_w)
            snr_b = dbm_to_snr(bs_dbm, cfg.noise_power_w)
            modes = analytical_mode_outages(relays, snr_r, snr_b, cfg)
            q_relay, q_bs, _ = proposed_soft_bcd(popularity, 3, 10, relays, modes)
            sim_modes = monte_carlo_mode_outages(
                relays,
                snr_r,
                snr_b,
                cfg,
                samples=samples,
                seed=20260522 + relays * 1000 + int(power) + (0 if sweep == "relay" else 100),
            )
            asym_modes = asymptotic_mode_outages(relays, snr_r, snr_b, cfg)
            rows.append(
                {
                    "relays": relays,
                    "power_dbm": float(power),
                    "analytical": system_outage(q_relay, q_bs, popularity, relays, modes),
                    "simulation": system_outage(q_relay, q_bs, popularity, relays, sim_modes),
                    "asymptotic": system_outage(q_relay, q_bs, popularity, relays, asym_modes),
                }
            )

    stem = "fig_analytical_1_relay_power" if sweep == "relay" else "fig_analytical_2_bs_power"
    xlabel = (
        "Transmit SNR of relays (dBm)"
        if sweep == "relay"
        else "Transmit SNR of BS (dBm)"
    )
    save_csv(rows, out_dir, stem)
    plot_analysis_curves(rows, out_dir, stem, xlabel)
    return rows


def strategy_power_sweep(
    out_dir: Path,
    sweep: str,
) -> list[dict[str, object]]:
    out_dir = Path(out_dir)
    cfg = SystemConfig()
    rows: list[dict[str, object]] = []

    for relays in (3, 5):
        for power in POWER_GRID:
            vals = _strategy_point(
                relays,
                relay_power_dbm=float(power if sweep == "relay" else 45.0),
                bs_power_dbm=float(power if sweep == "bs" else 45.0),
                relay_cache=3,
                bs_cache=10,
                tau=0.0,
                eta=1.5,
                cfg=cfg,
            )
            rows.append({"relays": relays, "power_dbm": float(power), **vals})

    stem = "fig_relay_power_strategy" if sweep == "relay" else "fig_bs_power_strategy"
    xlabel = (
        "Transmit power at the relays $P_R$ (dBm)"
        if sweep == "relay"
        else "Transmit power at the BS $P_{BS}$ (dBm)"
    )
    save_csv(rows, out_dir, stem)
    plot_strategy_curves(rows, out_dir, stem, "power_dbm", xlabel, ylim=(4e-5, 1.0))
    return rows


def parameter_sweep(
    out_dir: Path,
    parameter: str,
) -> list[dict[str, object]]:
    out_dir = Path(out_dir)
    cfg = SystemConfig()
    rows: list[dict[str, object]] = []

    grids = {
        "c1": [1, 3, 5, 7, 9],
        "c2": [5, 10, 15, 20, 25],
        "eta": [1, 1.5, 2, 2.5, 3, 3.5],
        "tau": [0, 2.5, 5, 7.5, 10, 12.5],
    }
    for relays in (3, 5):
        for x in grids[parameter]:
            vals = _strategy_point(
                relays,
                relay_power_dbm=45.0,
                bs_power_dbm=45.0,
                relay_cache=int(x) if parameter == "c1" else 3,
                bs_cache=int(x) if parameter == "c2" else 10,
                tau=float(x) if parameter == "tau" else 0.0,
                eta=float(x) if parameter == "eta" else 1.5,
                cfg=cfg,
            )
            rows.append({"relays": relays, parameter: float(x), **vals})

    mapping = {
        "c1": ("fig_c1_cache_size", "Relay cache size $C_1$", (2e-6, 1e-3)),
        "c2": ("fig_c2_cache_size", "BS cache size $C_2$", (2e-6, 1e-3)),
        "eta": ("fig_eta_skewness", "Skewness parameter of MZipf distribution", (5e-8, 2e-3)),
        "tau": ("fig_tau_plateau", "Plateau parameter of MZipf distribution", (5e-6, 1.5e-3)),
    }
    stem, xlabel, ylim = mapping[parameter]
    save_csv(rows, out_dir, stem)
    plot_strategy_curves(rows, out_dir, stem, parameter, xlabel, ylim=ylim)
    return rows


def soft_bcd_figure(out_dir: Path) -> list[dict[str, object]]:
    out_dir = Path(out_dir)
    cfg = SystemConfig()
    popularity = mzip_popularity(cfg.files, cfg.mzip_tau, cfg.mzip_eta)
    modes = analytical_mode_outages(
        3,
        dbm_to_snr(45.0, cfg.noise_power_w),
        dbm_to_snr(45.0, cfg.noise_power_w),
        cfg,
    )
    soft_history = bcd_objective_history(
        popularity,
        3,
        10,
        3,
        modes,
        iterations=50,
        soften_amplitude=0.45,
        seed=20260522,
    )
    bcd_history = bcd_objective_history(
        popularity,
        3,
        10,
        3,
        modes,
        iterations=50,
        soften_amplitude=0.0,
    )
    soft = np.array(soft_history)
    bcd = np.array(bcd_history)

    count = min(len(soft), len(bcd))
    rows = [
        {"iteration": i + 1, "soft_bcd": float(soft[i]), "bcd": float(bcd[i])}
        for i in range(count)
    ]
    save_csv(rows, out_dir, "fig_soft_bcd")

    import matplotlib.pyplot as plt

    apply_ieee_style()
    fig, ax = plt.subplots()
    x = [row["iteration"] for row in rows]
    ax.semilogy(x, [row["soft_bcd"] for row in rows], "r-", label="Soft-BCD")
    ax.semilogy(x, [row["bcd"] for row in rows], "k-", label="BCD")
    ax.set_ylim(2.9e-4, 7e-4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Outage probability")
    ax.legend()
    ax.grid(False)
    save_figure(fig, out_dir, "fig_soft_bcd")
    return rows


def run_all(out_dir: Path, samples: int = 200_000) -> None:
    out_dir = Path(out_dir)
    analysis_power_sweep(out_dir, "relay", samples=samples)
    analysis_power_sweep(out_dir, "bs", samples=samples)
    strategy_power_sweep(out_dir, "relay")
    strategy_power_sweep(out_dir, "bs")
    for parameter in ("c1", "c2", "eta", "tau"):
        parameter_sweep(out_dir, parameter)
    soft_bcd_figure(out_dir)
