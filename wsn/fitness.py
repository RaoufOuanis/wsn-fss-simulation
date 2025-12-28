# wsn/fitness.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np

from .models import Network
from .energy import RadioParams
from .multihop import dijkstra_costs_and_next_hops, relay_packet_counts
from .repair import (
    RepairParams,
    sanitize_ch_set,
    repair_ch_set_to_cover_all_alive,
    repair_ch_set_to_cover_and_connect_to_sink,
)


@dataclass
class FitnessParams:
    """Objective function parameters.

    Paper-aligned core (single-hop) defines:

        F(S(H+)) = w1 * C_E + w2 * (C̃_D + C̃_S)/2 + w3 * C̃_L

    with deterministic repair H+ = Repair_t(H0) enforcing strict Rc coverage,
    and a repair-dependence regularizer:

        P(H0) = |H+ \ H0| / N_t
        Fitness(H0) = F(S(H+)) + lam * P(H0)

    Multi-hop extension implemented here (CH->sink only):
      - replace C̃_S by C̃_S^{mh} computed using Dijkstra on per-hop energy cost
      - add a small relay-load term C̃_R to reduce hotspotting
      - extend repair to enforce sink connectivity under r_tx by promoting relays

    Notes
    -----
    - Only the base objective (including relay term) is clipped to [0, 1].
    - The regularized fitness is NOT clipped.
    - Repair is deterministic; ties are broken by smallest index.
    """

    # Paper weights
    w1: float = 0.4
    w2: float = 0.4
    w3: float = 0.2

    # Repair-dependence regularizer weight (paper's lambda)
    lam: float = 1.0

    # Geometry
    rc: float = 25.0

    # Multi-hop settings
    multihop: bool = True
    r_tx: float = 50.0
    w_relay: float = 0.05  # small anti-hotspot term weight

    # Models
    radio: RadioParams = field(default_factory=RadioParams)
    repair: RepairParams = field(default_factory=RepairParams)


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def fitness(
    net: Network,
    ch_indices: np.ndarray,
    params: FitnessParams,
) -> Tuple[float, Dict[str, float]]:
    """Compute Fitness(H0) and return component details."""

    alive = net.alive_mask
    n_alive = int(np.sum(alive))
    if n_alive <= 0:
        return 1.0, {"CE": 1.0, "CeD": 1.0, "CeS": 1.0, "CeL": 1.0, "CeR": 1.0, "P": 1.0, "F_base": 1.0}

    # Candidate set H0 (sanitized to alive unique indices)
    H0 = sanitize_ch_set(net, np.asarray(ch_indices, dtype=int).reshape(-1))

    # Deterministic repair
    if bool(params.multihop):
        H_plus = repair_ch_set_to_cover_and_connect_to_sink(
            net,
            ch0=H0,
            rc=float(params.rc),
            r_tx=float(params.r_tx),
            params=params.repair,
        )
    else:
        H_plus = repair_ch_set_to_cover_all_alive(net, H0, rc=float(params.rc), params=params.repair)

    if H_plus.size == 0:
        return 1.0, {"CE": 1.0, "CeD": 1.0, "CeS": 1.0, "CeL": 1.0, "CeR": 1.0, "P": 1.0, "F_base": 1.0}

    # Assignment: nearest CH in H_plus (tie-breaking by smallest CH index via sorted H_plus)
    assignments, dist_to_ch, _in_radius = net.assign_clusters(H_plus, rc=float(params.rc))

    # Restrict terms to alive nodes
    a = assignments[alive]
    d = dist_to_ch[alive]

    diag = net.diag if net.diag > 0 else 1.0

    # --------------------
    # C_E: CH energy fraction consumed (Eq. 7)
    # --------------------
    frac_consumed = 1.0 - (net.residual_energy[H_plus] / (net.initial_energy[H_plus] + 1e-12))
    CE = float(np.mean(np.clip(frac_consumed, 0.0, 1.0)))

    # --------------------
    # C̃_D: normalized member->CH distances (Eq. 9)
    # --------------------
    CeD = float(np.sum(d) / (n_alive * diag + 1e-12))
    CeD = _clip01(CeD)

    # --------------------
    # C̃_S or C̃_S^{mh}
    # --------------------
    CeR = 0.0
    if bool(params.multihop):
        # Dijkstra costs (per-packet energy) over CH-only graph + sink
        kappa, next_hop = dijkstra_costs_and_next_hops(
            net=net,
            ch_indices=H_plus,
            radio=params.radio,
            r_tx=float(params.r_tx),
        )

        # Safe normalization upper bound
        d_max = net.max_dists_to_sink if net.max_dists_to_sink > 0 else float(np.max(net.dists_to_sink))
        d_max = d_max if d_max > 0 else 1.0
        r_tx_f = float(max(1e-9, float(params.r_tx)))
        hops_max = int(np.ceil(d_max / r_tx_f)) + 1
        K_max = float(hops_max) * float(params.radio.tx_energy(params.radio.l_data, float(params.r_tx)) + params.radio.rx_energy(params.radio.l_data))
        K_max = K_max if K_max > 0 else 1.0

        kappa_safe = np.asarray(kappa, dtype=float)
        # If something is unreachable (should be prevented by repair), saturate at K_max
        kappa_safe = np.where(np.isfinite(kappa_safe), kappa_safe, K_max)

        CeS = float(np.sum(kappa_safe) / (n_alive * K_max + 1e-12))
        CeS = _clip01(CeS)

        # Relay hotspot term: variance of relay packet counts in shortest-path tree
        q = relay_packet_counts(H_plus, kappa_safe, next_hop)
        m = int(q.size)
        if m <= 1:
            CeR = 0.0
        else:
            var = float(np.var(q))
            CeR = _clip01(float(4.0 * var / float((m - 1) ** 2)))

    else:
        # Paper's single-hop surrogate
        d_sink = net.dists_to_sink[H_plus]
        d_max = net.max_dists_to_sink if net.max_dists_to_sink > 0 else float(np.max(net.dists_to_sink))
        d_max = d_max if d_max > 0 else 1.0
        CeS = float(np.sum(d_sink) / (n_alive * d_max + 1e-12))
        CeS = _clip01(CeS)

    # --------------------
    # C̃_L: normalized load imbalance (Eq. 10)
    # --------------------
    counts = np.bincount(a, minlength=net.n_nodes).astype(float)
    cluster_sizes = counts[H_plus]

    if n_alive <= 1 or cluster_sizes.size == 0:
        CeL = 0.0
    else:
        mu = float(np.mean(cluster_sizes))
        CL = float(np.mean((cluster_sizes - mu) ** 2))
        denom = float((n_alive - 1) ** 2)
        CeL = float(4.0 * CL / denom) if denom > 0 else 0.0
        CeL = _clip01(CeL)

    # --------------------
    # Base objective in [0,1]
    # --------------------
    F_article = (
        float(params.w1) * _clip01(CE)
        + float(params.w2) * (CeD + CeS) / 2.0
        + float(params.w3) * CeL
    )

    if bool(params.multihop) and float(params.w_relay) > 0.0:
        wR = float(params.w_relay)
        wR = min(1.0, max(0.0, wR))
        F_base = (1.0 - wR) * float(F_article) + wR * float(CeR)
    else:
        F_base = float(F_article)

    F_base = _clip01(float(F_base))

    # --------------------
    # Repair-dependence regularizer P(H0) in [0,1] (Eq. 12)
    # --------------------
    if H0.size == 0:
        added = int(H_plus.size)
    else:
        added = int(np.setdiff1d(H_plus, H0, assume_unique=True).size)
    P = _clip01(float(added) / float(n_alive))

    # --------------------
    # Regularized surrogate fitness (Eq. 13) - do NOT clip
    # --------------------
    F = float(F_base + float(params.lam) * P)

    return F, {
        "CE": _clip01(CE),
        "CeD": CeD,
        "CeS": CeS,
        "CeL": CeL,
        "CeR": _clip01(CeR),
        "P": P,
        "F_base": F_base,
    }


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
    refactoring the FSS file again). All algorithms must use the same fitness.
    """
    f, _ = fitness(net, ch_indices, params)
    return float(f)
