# wsn/fitness.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from .models import Network


@dataclass
class FitnessParams:
    """Objective function parameters.

    Fitness is computed as a weighted sum of bounded terms in [0, 1]:

        F(H) = w1 * CE + w2 * (CeD + CeS)/2 + w3 * CeL + lam * P

    - CE  : CH energy consumption fraction (favor high residual energy).
    - CeD : normalized intra-cluster distances.
    - CeS : normalized CH-to-sink distances.
    - CeL : normalized load imbalance (variance of cluster sizes).
    - P   : radius constraint penalty based on Rc.

    Lower is better.
    """

    w1: float = 0.4
    w2: float = 0.4
    w3: float = 0.2
    lam: float = 1.0
    rc: float = 25.0


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def fitness(
    net: Network,
    ch_indices: np.ndarray,
    params: FitnessParams,
) -> Tuple[float, Dict[str, float]]:
    """Compute fitness F(H) in [0,1] and return component details.

    Dead nodes are ignored in distance/load/penalty terms. CH candidates are
    filtered to alive nodes; if none remain, the fitness is set to 1.0.
    """

    ch = np.asarray(ch_indices, dtype=int).reshape(-1)
    alive = net.alive_mask
    n_alive = int(np.sum(alive))

    if n_alive <= 0:
        return 1.0, {"CE": 1.0, "CeD": 1.0, "CeS": 1.0, "CeL": 1.0, "P": 1.0}

    # Filter CHs to alive nodes and remove duplicates
    if ch.size == 0:
        return 1.0, {"CE": 1.0, "CeD": 1.0, "CeS": 1.0, "CeL": 1.0, "P": 1.0}

    ch = np.unique(ch[alive[ch]])
    if ch.size == 0:
        return 1.0, {"CE": 1.0, "CeD": 1.0, "CeS": 1.0, "CeL": 1.0, "P": 1.0}

    # Cluster assignment (vectorized, full length N)
    assignments, dist_to_ch, in_radius = net.assign_clusters(ch, rc=params.rc)

    # Restrict terms to alive nodes
    a = assignments[alive]
    d = dist_to_ch[alive]
    inr = in_radius[alive]

    diag = net.diag if net.diag > 0 else 1.0

    # --------------------
    # CE: CH energy fraction consumed
    # --------------------
    frac_consumed = 1.0 - (net.residual_energy[ch] / (net.initial_energy[ch] + 1e-12))
    CE = float(np.mean(np.clip(frac_consumed, 0.0, 1.0))) if ch.size else 1.0

    # --------------------
    # CeD: normalized intra-cluster distances
    # --------------------
    CeD = float(np.sum(d) / (n_alive * diag + 1e-12))
    CeD = _clip01(CeD)

    # --------------------
    # CeS: normalized CH->sink distances
    # --------------------
    d_sink = net.dists_to_sink[ch]
    d_max = net.max_dists_to_sink if net.max_dists_to_sink > 0 else float(np.max(net.dists_to_sink))
    d_max = d_max if d_max > 0 else 1.0
    CeS = float(np.mean(d_sink) / (d_max + 1e-12))
    CeS = _clip01(CeS)

    # --------------------
    # CeL: normalized load imbalance via variance of cluster sizes
    # --------------------
    counts = np.bincount(a, minlength=net.n_nodes).astype(float)
    cluster_sizes = counts[ch]

    if cluster_sizes.size > 1:
        var = float(np.var(cluster_sizes))
        # Popoviciu: Var(X) <= (b-a)^2 / 4, here a=0, b=n_alive
        max_var = (float(n_alive) ** 2) / 4.0
        CeL = var / max_var if max_var > 0 else 1.0
        CeL = _clip01(CeL)
    else:
        # One CH: variance is 0 but load balancing is poor
        CeL = 1.0

    # --------------------
    # P(H): radius penalty
    # --------------------
    beyond = ~inr
    p1 = float(np.mean(beyond))

    if np.any(beyond):
        exceed = np.maximum(0.0, d - float(params.rc))
        p2 = float(np.mean(exceed / (diag + 1e-12)))
    else:
        p2 = 0.0

    P = _clip01(p1 + p2)

    # --------------------
    # Aggregate fitness
    # --------------------
    F = (
        float(params.w1) * CE
        + float(params.w2) * (CeD + CeS) / 2.0
        + float(params.w3) * CeL
        + float(params.lam) * P
    )
    F = _clip01(F)

    return F, {"CE": _clip01(CE), "CeD": CeD, "CeS": CeS, "CeL": _clip01(CeL), "P": P}


def fitness_ch_selection(
    net: Network,
    ch_indices: np.ndarray,
    residual_energy: np.ndarray,
    initial_energy: np.ndarray,
    params: FitnessParams,
) -> float:
    """Compatibility helper used by the current FSS implementation.

    The optimization is evaluated under the network's current state. The
    residual_energy/initial_energy arguments are ignored (kept only to avoid
    refactoring the FSS file again).  All algorithms must use the same fitness.
    """
    f, _ = fitness(net, ch_indices, params)
    return float(f)
