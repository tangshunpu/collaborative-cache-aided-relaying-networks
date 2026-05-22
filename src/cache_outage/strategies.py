from __future__ import annotations

import math

import numpy as np

from .model import ModeOutages, system_outage


def mpc_strategy(files: int, relay_cache: int, bs_cache: int) -> tuple[np.ndarray, np.ndarray]:
    q_relay = np.zeros(files)
    q_bs = np.zeros(files)
    q_relay[: min(relay_cache, files)] = 1.0
    q_bs[: min(bs_cache, files)] = 1.0
    return q_relay, q_bs


def epc_strategy(
    files: int,
    relays: int,
    relay_cache: int,
    bs_cache: int,
) -> tuple[np.ndarray, np.ndarray]:
    q_relay = np.zeros(files)
    q_bs = np.zeros(files)
    relay_diversity_files = min(files, relays * relay_cache)
    if relay_diversity_files > 0:
        q_relay[:relay_diversity_files] = relay_cache / relay_diversity_files
    residual = files - relay_diversity_files
    if residual > 0:
        q_bs[relay_diversity_files:] = min(1.0, bs_cache / residual)
    return q_relay, q_bs


def _relay_derivative(q: float, q_bs: float, relays: int, modes: ModeOutages) -> float:
    total = 0.0
    for g in range(1, relays + 1):
        term = 0.0
        if g > 0:
            term += g * q ** (g - 1) * (1.0 - q) ** (relays - g)
        if relays - g > 0:
            term -= (relays - g) * q**g * (1.0 - q) ** (relays - g - 1)
        total += math.comb(relays, g) * modes.relay_hit[g] * term
    miss_outage = q_bs * modes.bs_hit + (1.0 - q_bs) * modes.source_fetch
    return total - relays * miss_outage * (1.0 - q) ** (relays - 1)


def optimize_q1_given_q2(
    q_bs: np.ndarray,
    popularity: np.ndarray,
    relay_cache: int,
    relays: int,
    modes: ModeOutages,
    iterations: int = 45,
) -> np.ndarray:
    files = len(popularity)

    def response(lam: float) -> np.ndarray:
        q = np.zeros(files)
        for k in range(files):
            def h(x: float) -> float:
                return popularity[k] * _relay_derivative(x, q_bs[k], relays, modes) + lam

            if h(0.0) >= 0.0:
                q[k] = 0.0
            elif h(1.0) <= 0.0:
                q[k] = 1.0
            else:
                lo, hi = 0.0, 1.0
                for _ in range(45):
                    mid = 0.5 * (lo + hi)
                    if h(mid) > 0.0:
                        hi = mid
                    else:
                        lo = mid
                q[k] = 0.5 * (lo + hi)
        return q

    low, high = -1.0, 1.0
    while response(low).sum() < relay_cache:
        low *= 2.0
    while response(high).sum() > relay_cache:
        high *= 2.0

    q = np.zeros(files)
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        q = response(mid)
        if q.sum() > relay_cache:
            low = mid
        else:
            high = mid
    return q


def optimize_q2_given_q1(
    q_relay: np.ndarray,
    popularity: np.ndarray,
    bs_cache: int,
    relays: int,
    modes: ModeOutages,
) -> np.ndarray:
    score = popularity * (modes.bs_hit - modes.source_fetch) * (1.0 - q_relay) ** relays
    order = np.argsort(score)
    q_bs = np.zeros_like(q_relay)
    q_bs[order[:bs_cache]] = 1.0
    return q_bs


def proposed_soft_bcd(
    popularity: np.ndarray,
    relay_cache: int,
    bs_cache: int,
    relays: int,
    modes: ModeOutages,
    iterations: int = 12,
    soften: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    q_relay = np.zeros_like(popularity)
    history: list[float] = []
    for _ in range(iterations):
        q_bs_binary = optimize_q2_given_q1(q_relay, popularity, bs_cache, relays, modes)
        q_bs_for_q1 = np.where(q_bs_binary > 0.5, 1.0 - soften, soften)
        q_relay = optimize_q1_given_q2(
            q_bs_for_q1, popularity, relay_cache, relays, modes
        )
        q_bs_binary = optimize_q2_given_q1(q_relay, popularity, bs_cache, relays, modes)
        history.append(system_outage(q_relay, q_bs_binary, popularity, relays, modes))
    return q_relay, q_bs_binary, history
