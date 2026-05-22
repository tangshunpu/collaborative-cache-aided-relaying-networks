from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import special


@dataclass(frozen=True)
class SystemConfig:
    files: int = 50
    file_size: float = 1.0
    latency_threshold: float = 1.0
    radius: float = 100.0
    path_loss: float = 3.0
    relay_bs_distance: float = 75.0
    t_intra_backhaul: float = 0.1
    t_core_backhaul: float = 0.2
    noise_power_w: float = 1e-6
    mzip_tau: float = 0.0
    mzip_eta: float = 1.5


@dataclass(frozen=True)
class ModeOutages:
    relay_hit: np.ndarray
    bs_hit: float
    source_fetch: float


def dbm_to_snr(power_dbm: float, noise_power_w: float = 1e-6) -> float:
    power_w = 10.0 ** ((power_dbm - 30.0) / 10.0)
    return power_w / noise_power_w


def mzip_popularity(files: int, tau: float = 0.0, eta: float = 1.5) -> np.ndarray:
    idx = np.arange(1, files + 1, dtype=float)
    pop = (idx + tau) ** (-eta)
    return pop / pop.sum()


def _lower_gamma(s: float, x: float) -> float:
    return float(special.gamma(s) * special.gammainc(s, x))


def gamma_single_hop(cfg: SystemConfig, backhaul_latency: float = 0.0) -> float:
    return 2.0 ** (cfg.file_size / (cfg.latency_threshold - backhaul_latency)) - 1.0


def gamma_two_hop(cfg: SystemConfig, backhaul_latency: float = 0.0) -> float:
    return 2.0 ** (
        2.0 * cfg.file_size / (cfg.latency_threshold - backhaul_latency)
    ) - 1.0


def pr_relay_hit_closed(
    cached_relays: int,
    relays: int,
    snr_relay: float,
    cfg: SystemConfig,
) -> float:
    gamma_cached = gamma_single_hop(cfg, 0.0)
    gamma_uncached = gamma_single_hop(cfg, cfg.t_intra_backhaul)
    sigma = cfg.path_loss
    v = 2.0 / sigma
    temp = 0.0

    for n1 in range(cached_relays + 1):
        for n2 in range(1, relays - cached_relays + 1):
            beta = (n1 * gamma_cached + n2 * gamma_uncached) / snr_relay
            coeff = (
                math.comb(cached_relays, n1)
                * math.comb(relays - cached_relays, n2)
                * (-1.0) ** (n1 + n2)
            )
            temp += coeff * _lower_gamma(v, beta * cfg.radius**sigma) / (
                sigma * beta**v
            )

    for n1 in range(1, cached_relays + 1):
        beta = n1 * gamma_cached / snr_relay
        coeff = math.comb(cached_relays, n1) * (-1.0) ** n1
        temp += coeff * _lower_gamma(v, beta * cfg.radius**sigma) / (
            sigma * beta**v
        )

    return float(1.0 + 2.0 * temp / cfg.radius**2)


def pr_two_hop_closed(
    relays: int,
    snr_relay: float,
    snr_bs: float,
    backhaul_latency: float,
    cfg: SystemConfig,
) -> float:
    gamma = gamma_two_hop(cfg, backhaul_latency)
    sigma = cfg.path_loss
    v = 2.0 / sigma
    temp = 0.0

    for n in range(1, relays + 1):
        beta = n * gamma / snr_relay
        coeff = math.comb(relays, n) * (-1.0) ** n
        bs_exp = math.exp(-n * gamma / (cfg.relay_bs_distance ** (-sigma) * snr_bs))
        temp += coeff * bs_exp * _lower_gamma(v, beta * cfg.radius**sigma) / (
            sigma * beta**v
        )

    return float(1.0 + 2.0 * temp / cfg.radius**2)


def analytical_mode_outages(
    relays: int,
    snr_relay: float,
    snr_bs: float,
    cfg: SystemConfig,
) -> ModeOutages:
    relay_hit = np.zeros(relays + 1, dtype=float)
    for g in range(1, relays + 1):
        relay_hit[g] = pr_relay_hit_closed(g, relays, snr_relay, cfg)
    return ModeOutages(
        relay_hit=relay_hit,
        bs_hit=pr_two_hop_closed(relays, snr_relay, snr_bs, 0.0, cfg),
        source_fetch=pr_two_hop_closed(
            relays, snr_relay, snr_bs, cfg.t_core_backhaul, cfg
        ),
    )


def asymptotic_pr_relay_hit(
    cached_relays: int,
    relays: int,
    snr_relay: float,
    cfg: SystemConfig,
) -> float:
    gamma_cached = gamma_single_hop(cfg, 0.0)
    gamma_uncached = gamma_single_hop(cfg, cfg.t_intra_backhaul)
    sigma = cfg.path_loss
    numerator = (
        2.0
        * gamma_cached**cached_relays
        * gamma_uncached ** (relays - cached_relays)
        * cfg.radius ** (relays * sigma)
    )
    return float(numerator / (snr_relay**relays * (sigma * relays + 2.0)))


def asymptotic_pr_two_hop(
    relays: int,
    snr_relay: float,
    snr_bs: float,
    backhaul_latency: float,
    cfg: SystemConfig,
) -> float:
    gamma = gamma_two_hop(cfg, backhaul_latency)
    sigma = cfg.path_loss
    total = 0.0
    for n in range(relays + 1):
        delta = sigma * (relays - n)
        total += (
            math.comb(relays, n)
            * snr_relay ** (n - relays)
            * snr_bs ** (-n)
            * cfg.relay_bs_distance ** (sigma * n)
            * cfg.radius**delta
            / (delta + 2.0)
        )
    return float(2.0 * gamma**relays * total)


def asymptotic_mode_outages(
    relays: int,
    snr_relay: float,
    snr_bs: float,
    cfg: SystemConfig,
) -> ModeOutages:
    relay_hit = np.zeros(relays + 1, dtype=float)
    for g in range(1, relays + 1):
        relay_hit[g] = asymptotic_pr_relay_hit(g, relays, snr_relay, cfg)
    return ModeOutages(
        relay_hit=relay_hit,
        bs_hit=asymptotic_pr_two_hop(relays, snr_relay, snr_bs, 0.0, cfg),
        source_fetch=asymptotic_pr_two_hop(
            relays, snr_relay, snr_bs, cfg.t_core_backhaul, cfg
        ),
    )


def monte_carlo_mode_outages(
    relays: int,
    snr_relay: float,
    snr_bs: float,
    cfg: SystemConfig,
    samples: int = 200_000,
    seed: int = 20260522,
) -> ModeOutages:
    rng = np.random.default_rng(seed)
    sigma = cfg.path_loss
    radius = cfg.radius
    relay_hit = np.zeros(relays + 1, dtype=float)

    for g in range(1, relays + 1):
        d = radius * np.sqrt(rng.random(samples))
        h_cached = rng.exponential(1.0, size=(samples, g)).max(axis=1)
        outage_cached = (
            snr_relay * h_cached * d ** (-sigma) < gamma_single_hop(cfg, 0.0)
        )
        if g == relays:
            relay_hit[g] = float(np.mean(outage_cached))
            continue
        h_uncached = rng.exponential(1.0, size=(samples, relays - g)).max(axis=1)
        outage_uncached = (
            snr_relay * h_uncached * d ** (-sigma)
            < gamma_single_hop(cfg, cfg.t_intra_backhaul)
        )
        relay_hit[g] = float(np.mean(outage_cached & outage_uncached))

    def two_hop(backhaul_latency: float) -> float:
        d = radius * np.sqrt(rng.random(samples))
        h_rd = rng.exponential(1.0, size=(samples, relays))
        h_br = rng.exponential(1.0, size=(samples, relays))
        rd = snr_relay * h_rd * d[:, None] ** (-sigma)
        br = snr_bs * h_br * cfg.relay_bs_distance ** (-sigma)
        selected = np.minimum(rd, br).max(axis=1)
        return float(np.mean(selected < gamma_two_hop(cfg, backhaul_latency)))

    return ModeOutages(
        relay_hit=relay_hit,
        bs_hit=two_hop(0.0),
        source_fetch=two_hop(cfg.t_core_backhaul),
    )


def system_outage(
    q_relay: np.ndarray,
    q_bs: np.ndarray,
    popularity: np.ndarray,
    relays: int,
    modes: ModeOutages,
) -> float:
    total = 0.0
    for q1, q2, ak in zip(q_relay, q_bs, popularity, strict=True):
        hit = 0.0
        for g in range(1, relays + 1):
            hit += (
                math.comb(relays, g)
                * modes.relay_hit[g]
                * q1**g
                * (1.0 - q1) ** (relays - g)
            )
        miss = (1.0 - q1) ** relays
        total += ak * (
            hit + miss * (q2 * modes.bs_hit + (1.0 - q2) * modes.source_fetch)
        )
    return float(total)
