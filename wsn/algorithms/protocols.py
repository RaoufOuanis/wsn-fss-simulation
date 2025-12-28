from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from ..models import Network
from ..fitness import FitnessParams
from .base import OptimizationResult


def _default_k(n_alive: int) -> int:
    return int(np.clip(round(0.05 * n_alive), 1, max(1, n_alive // 2)))


def _make_rng(base_seed: int, round_idx: int, salt: int) -> np.random.Generator:
    """
    Robust per-round RNG.
    SeedSequence avoids collisions and makes randomness stable across platforms.
    'salt' separates streams between different protocols.
    """
    ss = np.random.SeedSequence([int(base_seed), int(round_idx), int(salt)])
    return np.random.default_rng(ss)


# -----------------------------------------------------------------------------
# LEACH
# -----------------------------------------------------------------------------

@dataclass
class LEACHParams:
    p_opt: float = 0.05
    seed: int = 0
    min_ch: int = 1
    max_ch: Optional[int] = None


@dataclass
class LEACHState:
    last_ch_round: np.ndarray  # (N,)

    @staticmethod
    def initialize(n_nodes: int) -> "LEACHState":
        return LEACHState(last_ch_round=np.full(int(n_nodes), -10**9, dtype=int))


def run_leach_wsn(
    net: Network,
    fit_params: FitnessParams,
    params: LEACHParams,
    round_idx: int,
    state: LEACHState,
) -> OptimizationResult:
    """Centralized LEACH baseline with the classic epoch-based eligibility set G."""

    rng = _make_rng(params.seed, round_idx, salt=11)
    alive = net.alive_mask
    alive_idx = np.where(alive)[0]

    if alive_idx.size == 0:
        return OptimizationResult(best_ch_indices=np.array([], dtype=int), best_fitness=1.0, history={})

    p = float(params.p_opt)
    p = float(np.clip(p, 1e-6, 1.0))
    epoch = int(max(1, round(1.0 / p)))

    eligible = alive & ((round_idx - state.last_ch_round) >= epoch)
    eligible_idx = np.where(eligible)[0]

    # LEACH threshold
    r_mod = int(round_idx % epoch)
    denom = max(1e-12, 1.0 - p * r_mod)
    T = p / denom

    ch_mask = np.zeros(net.n_nodes, dtype=bool)
    if eligible_idx.size > 0:
        ch_mask[eligible_idx] = rng.random(size=eligible_idx.size) < T

    ch_idx = np.where(ch_mask)[0]

    # Enforce CH count constraints
    if params.max_ch is not None:
        max_ch = int(params.max_ch)
        if ch_idx.size > max_ch:
            order = np.argsort(-net.residual_energy[ch_idx])
            ch_idx = ch_idx[order[:max_ch]]

    if ch_idx.size < int(params.min_ch):
        need = int(params.min_ch) - int(ch_idx.size)
        candidates = np.setdiff1d(alive_idx, ch_idx, assume_unique=False)
        if candidates.size > 0:
            order = np.argsort(-net.residual_energy[candidates])
            add = candidates[order[: min(need, candidates.size)]]
            ch_idx = np.unique(np.concatenate([ch_idx, add]))

    # Update rotation state
    state.last_ch_round[ch_idx] = int(round_idx)

    return OptimizationResult(best_ch_indices=np.sort(ch_idx), best_fitness=0.0, history={})


# -----------------------------------------------------------------------------
# SEP (two-level heterogeneity)
# -----------------------------------------------------------------------------

@dataclass
class SEPParams:
    p_opt: float = 0.05
    seed: int = 0
    min_ch: int = 1
    max_ch: Optional[int] = None
    e0: float = 0.01
    e_adv: float = 0.05
    adv_fraction: float = 0.2


@dataclass
class SEPState:
    last_ch_round: np.ndarray

    @staticmethod
    def initialize(n_nodes: int) -> "SEPState":
        return SEPState(last_ch_round=np.full(int(n_nodes), -10**9, dtype=int))


def run_sep_wsn(
    net: Network,
    fit_params: FitnessParams,
    params: SEPParams,
    round_idx: int,
    state: SEPState,
) -> OptimizationResult:
    """SEP baseline (two-level heterogeneity) with epoch eligibility."""

    rng = _make_rng(params.seed, round_idx, salt=22)
    alive = net.alive_mask
    alive_idx = np.where(alive)[0]

    if alive_idx.size == 0:
        return OptimizationResult(best_ch_indices=np.array([], dtype=int), best_fitness=1.0, history={})

    p_opt = float(params.p_opt)
    p_opt = float(np.clip(p_opt, 1e-6, 1.0))

    m = float(params.adv_fraction)
    e0 = float(params.e0)
    e_adv = float(params.e_adv)
    a = (e_adv - e0) / max(1e-12, e0)

    p_norm = p_opt / (1.0 + a * m)
    p_adv = p_opt * (1.0 + a) / (1.0 + a * m)

    # Identify advanced nodes
    is_adv = net.initial_energy > (e0 + 1e-12)

    p_i = np.where(is_adv, p_adv, p_norm)
    p_i = np.clip(p_i, 1e-6, 1.0)

    epoch_i = np.maximum(1, np.round(1.0 / p_i).astype(int))
    eligible = alive & ((round_idx - state.last_ch_round) >= epoch_i)

    ch_mask = np.zeros(net.n_nodes, dtype=bool)
    eligible_idx = np.where(eligible)[0]
    if eligible_idx.size > 0:
        r_mod = (round_idx % epoch_i[eligible_idx]).astype(float)
        denom = np.maximum(1e-12, 1.0 - p_i[eligible_idx] * r_mod)
        T = p_i[eligible_idx] / denom
        ch_mask[eligible_idx] = rng.random(size=eligible_idx.size) < T

    ch_idx = np.where(ch_mask)[0]

    if params.max_ch is not None:
        max_ch = int(params.max_ch)
        if ch_idx.size > max_ch:
            order = np.argsort(-net.residual_energy[ch_idx])
            ch_idx = ch_idx[order[:max_ch]]

    if ch_idx.size < int(params.min_ch):
        need = int(params.min_ch) - int(ch_idx.size)
        candidates = np.setdiff1d(alive_idx, ch_idx, assume_unique=False)
        if candidates.size > 0:
            order = np.argsort(-net.residual_energy[candidates])
            add = candidates[order[: min(need, candidates.size)]]
            ch_idx = np.unique(np.concatenate([ch_idx, add]))

    state.last_ch_round[ch_idx] = int(round_idx)

    return OptimizationResult(best_ch_indices=np.sort(ch_idx), best_fitness=0.0, history={})


# -----------------------------------------------------------------------------
# HEED (centralized, HEED-inspired)
# -----------------------------------------------------------------------------

@dataclass
class HEEDParams:
    p_init: float = 0.05
    c_min: float = 0.02
    n_iter: int = 3
    seed: int = 0
    min_ch: int = 1
    max_ch: Optional[int] = None


def run_heed_wsn(
    net: Network,
    fit_params: FitnessParams,
    params: HEEDParams,
    round_idx: int,
) -> OptimizationResult:
    """HEED-inspired baseline (centralized approximation).

    - Initial CH probability proportional to residual energy ratio.
    - Probability doubles each iteration until capped at 1.
    - Uncovered alive nodes are promoted (highest energy) to guarantee coverage.
    """

    rng = _make_rng(params.seed, round_idx, salt=33)

    alive = net.alive_mask
    alive_idx = np.where(alive)[0]
    if alive_idx.size == 0:
        return OptimizationResult(best_ch_indices=np.array([], dtype=int), best_fitness=1.0, history={})

    e_ratio = net.residual_energy / (net.initial_energy + 1e-12)
    p = float(params.p_init) * e_ratio
    p = np.clip(p, float(params.c_min), 1.0)

    ch_set: set[int] = set()

    for _ in range(int(params.n_iter)):
        tentative = alive & (rng.random(size=net.n_nodes) < p)
        # add tentative CHs (no Python loop over indices)
        idx = np.where(tentative)[0]
        if idx.size:
            ch_set.update(map(int, idx))

        if not ch_set:
            best = int(alive_idx[int(np.argmax(net.residual_energy[alive_idx]))])
            ch_set.add(best)

        ch = np.fromiter(ch_set, dtype=int)
        ch.sort()

        # coverage check (alive nodes only)
        _, dist_to_ch, _ = net.assign_clusters(ch, rc=fit_params.rc)
        uncovered = alive & (dist_to_ch > float(fit_params.rc))
        if not np.any(uncovered):
            break

        # promote uncovered nodes (highest energy)
        unc_idx = np.where(uncovered)[0]
        order = np.argsort(-net.residual_energy[unc_idx])
        promote_n = int(min(max(1, 0.05 * alive_idx.size), unc_idx.size))
        promote = unc_idx[order[:promote_n]]
        if promote.size:
            ch_set.update(map(int, promote))

        p = np.minimum(1.0, 2.0 * p)

    ch_idx = np.fromiter(ch_set, dtype=int)
    ch_idx.sort()

    # Enforce count constraints
    if params.max_ch is not None and ch_idx.size > int(params.max_ch):
        order = np.argsort(-net.residual_energy[ch_idx])
        ch_idx = ch_idx[order[: int(params.max_ch)]]

    if ch_idx.size < int(params.min_ch):
        need = int(params.min_ch) - int(ch_idx.size)
        candidates = np.setdiff1d(alive_idx, ch_idx, assume_unique=False)
        if candidates.size > 0:
            order = np.argsort(-net.residual_energy[candidates])
            add = candidates[order[: min(need, candidates.size)]]
            ch_idx = np.unique(np.concatenate([ch_idx, add]))

    return OptimizationResult(best_ch_indices=np.sort(ch_idx), best_fitness=0.0, history={})


# -----------------------------------------------------------------------------
# Greedy heuristic baseline
# -----------------------------------------------------------------------------

@dataclass
class GreedyParams:
    seed: int = 0
    n_ch: int = 0  # if <=0, use default k rule
    w_energy: float = 0.7
    w_sink: float = 0.3


def run_greedy_wsn(
    net: Network,
    fit_params: FitnessParams,
    params: GreedyParams,
    round_idx: int,
) -> OptimizationResult:
    """Simple energy-aware greedy baseline (top-k by score + minimal repair)."""

    rng = _make_rng(params.seed, round_idx, salt=44)
    alive = net.alive_mask
    alive_idx = np.where(alive)[0]

    if alive_idx.size == 0:
        return OptimizationResult(best_ch_indices=np.array([], dtype=int), best_fitness=1.0, history={})

    k = int(params.n_ch)
    if k <= 0:
        k = _default_k(int(alive_idx.size))
    k = int(min(k, alive_idx.size))

    e_ratio = net.residual_energy / (net.initial_energy + 1e-12)
    d_norm = net.dists_to_sink / (max(1e-12, net.max_dists_to_sink))

    score = float(params.w_energy) * e_ratio - float(params.w_sink) * d_norm

    # break ties with a small random jitter (deterministic via seed/round/salt)
    score = score + 1e-6 * rng.normal(size=net.n_nodes)

    idx_sorted = alive_idx[np.argsort(-score[alive_idx])]
    ch_set = set(map(int, idx_sorted[:k]))

    # minimal repair for coverage under Rc
    ch = np.array(sorted(ch_set), dtype=int)
    _, dist_to_ch, _ = net.assign_clusters(ch, rc=fit_params.rc)
    uncovered = alive & (dist_to_ch > float(fit_params.rc))

    if np.any(uncovered):
        unc_idx = np.where(uncovered)[0]
        order = np.argsort(-score[unc_idx])
        promote_n = int(min(max(1, 0.05 * alive_idx.size), unc_idx.size))
        promote = unc_idx[order[:promote_n]]
        if promote.size:
            ch_set.update(map(int, promote))

    ch_idx = np.array(sorted(ch_set), dtype=int)
    return OptimizationResult(best_ch_indices=ch_idx, best_fitness=0.0, history={})
