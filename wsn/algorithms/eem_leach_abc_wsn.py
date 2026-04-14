"""
EEM-LEACH-ABC — Energy-Efficient Multi-hop LEACH with ABC optimisation.

Reference
---------
Zhang, Liu & Trik, "Energy efficient multi hop clustering using Artificial
Bee Colony metaheuristic in WSN", Scientific Reports 15:26803, 2025.
DOI: 10.1038/s41598-025-12321-y

The paper describes TWO levels of optimisation:

1.  **Per-round CH election** (Algorithm 1 / Eq. 6):
    Each alive node *n* gets a score  Fitness(n) = E_res(n) / E_avg(n).
    The top  CR × N_alive  nodes (by a weighted score combining fitness and
    sink proximity) are elected CHs.

2.  **Offline parameter tuning via ABC** (Eq. 9-12):
    The ABC meta-heuristic searches for the best (CR, µ) pair that maximises
    β₁·FND + β₂·HND + β₃·LND.  This is done **before** the main simulation.
    We implement this as a single tuning run at the start.

For fair comparison inside our simulator (where all algorithms share the
same fitness/repair/energy pipeline), we implement the per-round CH
election faithfully and pre-tune (CR, µ) with a short ABC search.

Multi-hop CH→BS routing and energy deduction are handled by the simulator's
existing pipeline (`apply_round_energy`), ensuring perfect fairness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..models import Network
from ..fitness import FitnessParams
from .base import OptimizationResult


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_rng(base_seed: int, round_idx: int) -> np.random.Generator:
    ss = np.random.SeedSequence([int(base_seed), int(round_idx), 99])
    return np.random.default_rng(ss)


# ── parameters ──────────────────────────────────────────────────────────────

@dataclass
class EEMParams:
    """Per-round protocol parameters (after ABC tuning)."""

    cr: float = 0.10          # cluster-head ratio  (paper range [0.01, 0.25])
    mu: float = 0.70          # weight factor: balance energy vs sink proximity
    seed: int = 0
    min_ch: int = 1
    max_ch: Optional[int] = None
    epoch: int = 5            # re-election cooldown (rounds)


@dataclass
class EEMState:
    """Persistent inter-round state (CH rotation history)."""

    last_ch_round: np.ndarray   # (N,)

    @staticmethod
    def initialize(n_nodes: int) -> "EEMState":
        return EEMState(
            last_ch_round=np.full(int(n_nodes), -10**9, dtype=int),
        )


# ── per-round CH election (Algorithm 1 / Eq. 1, 6) ─────────────────────────

def run_eem_leach_abc_wsn(
    net: Network,
    fit_params: FitnessParams,
    params: EEMParams,
    round_idx: int,
    state: EEMState,
) -> OptimizationResult:
    """Select CHs for one round using the EEM-LEACH-ABC protocol.

    Implements the energy-aware CH election described in the paper:
    1. Compute per-node fitness  f(n) = E_res(n) / E_avg  (Eq. 6)
    2. Compute hierarchical weight  W(n) = E_res/E_max + d_min/d(n,sink)  (Eq. 1)
    3. Combine:  score(n) = µ · f(n)_norm + (1-µ) · W(n)_norm
    4. Elect top  CR × N_alive  nodes (subject to epoch eligibility)
    """

    rng = _make_rng(params.seed, round_idx)
    alive = net.alive_mask
    alive_idx = np.where(alive)[0]

    if alive_idx.size == 0:
        return OptimizationResult(
            best_ch_indices=np.array([], dtype=int),
            best_fitness=1.0,
            history={},
        )

    n_alive = int(alive_idx.size)

    # ── 1. Per-node fitness  Fitness(n) = E_res(n) / E_avg  (Eq. 6) ──

    e_res = net.residual_energy[alive_idx]              # (n_alive,)
    e_avg = float(np.mean(e_res)) + 1e-30
    fitness_node = e_res / e_avg                        # > 1 ⇒ above average

    # ── 2. Hierarchical weight  W_i = E_ri/E_max + d_min/d(i,s)  (Eq. 1) ──

    e_max = float(np.max(net.initial_energy)) + 1e-30
    d_sink = net.dists_to_sink[alive_idx]               # (n_alive,)
    d_min = float(np.min(d_sink)) + 1e-30
    w_hier = (e_res / e_max) + (d_min / (d_sink + 1e-30))

    # ── 3. Combined score  (normalise both to [0,1] then blend) ──

    def _norm01(x: np.ndarray) -> np.ndarray:
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi - lo < 1e-30:
            return np.ones_like(x) * 0.5
        return (x - lo) / (hi - lo)

    mu = float(params.mu)
    score = mu * _norm01(fitness_node) + (1.0 - mu) * _norm01(w_hier)

    # ── 4. Epoch eligibility (rotation, prevent re-election too soon) ──

    eligible = (round_idx - state.last_ch_round[alive_idx]) >= int(params.epoch)
    # If nobody is eligible, reset eligibility for all alive nodes
    if not np.any(eligible):
        eligible = np.ones(n_alive, dtype=bool)

    score[~eligible] = -np.inf  # disqualify ineligible nodes

    # ── 5. Elect top  CR × N_alive  nodes ──

    n_ch = max(int(params.min_ch), int(round(float(params.cr) * n_alive)))
    if params.max_ch is not None:
        n_ch = min(n_ch, int(params.max_ch))
    n_ch = min(n_ch, n_alive)

    # Indices within alive_idx array, sorted by score desc
    top_k_local = np.argsort(-score)[:n_ch]
    ch_idx = alive_idx[top_k_local]

    # Enforce min_ch (pick highest residual energy if score ties / all -inf)
    if ch_idx.size < int(params.min_ch):
        need = int(params.min_ch) - int(ch_idx.size)
        candidates = np.setdiff1d(alive_idx, ch_idx, assume_unique=False)
        if candidates.size > 0:
            order = np.argsort(-net.residual_energy[candidates])
            add = candidates[order[: min(need, candidates.size)]]
            ch_idx = np.unique(np.concatenate([ch_idx, add]))

    # ── Update rotation state ──
    state.last_ch_round[ch_idx] = int(round_idx)

    return OptimizationResult(
        best_ch_indices=np.sort(ch_idx),
        best_fitness=0.0,
        history={},
    )
