from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import math
import numpy as np

from ..models import Network
from ..fitness import FitnessParams, fitness
from .base import OptimizationResult
from .decode import decode_topk_bounded


@dataclass
class EMOGJOParams:
    """EMO-GJO baseline (Soft Computing 2024) as a strict-fairness optimizer.

    Strict fairness (default):
    - Same bounded top-k decoding as PSO/GWO/ABC/SO/GJO.
    - Same Fitness_t(H0) from the framework (which applies Repair_t internally).
    - Same iteration budget n_iter and population size.

    Optional: a paper-style CH-selection fitness is available for ablations via
    use_paper_ch_fitness=True (NOT used in strict-fairness experiments).
    """

    pop_size: int = 30
    n_iter: int = 60
    seed: int = 0

    min_ch: int = 1
    max_ch: int = 20

    # Paper parameters referenced in the user's guide
    c1: float = 1.5
    b_levy: float = 1.5

    # CH selection fitness weights (Eq.19 in the user's guide)
    u1_mne: float = 0.35
    u2_innc: float = 0.25
    u3_dsch: float = 0.20
    u4_dchbs: float = 0.10
    u5_nd: float = 0.10

    # Energy term ambiguity handling
    # - "monotone" : energy becomes a cost (penalize low-energy CHs) (recommended)
    # - "literal"  : uses energy directly as a cost (not recommended but kept for ablation)
    energy_mode: str = "monotone"

    # Small constant for divisions
    eps: float = 1e-12

    # Strict-fairness evaluation (default): optimize framework fitness()
    use_paper_ch_fitness: bool = False

    # Numerical guard for continuous state (keeps sigmoid stable)
    x_clip: float = 20.0


def _levy_flight(rng: np.random.Generator, size: int, beta: float) -> np.ndarray:
    """Generate Lévy-flight steps using Mantegna's algorithm."""
    beta = float(beta)
    if not (0.0 < beta <= 2.0):
        beta = 1.5

    # sigma_u per Mantegna
    # https://en.wikipedia.org/wiki/L%C3%A9vy_flight#Mantegna%27s_algorithm
    num = math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
    den = math.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
    sigma_u = (num / max(den, 1e-30)) ** (1.0 / beta)

    u = rng.normal(0.0, sigma_u, size=int(size))
    v = rng.normal(0.0, 1.0, size=int(size))
    step = u / (np.abs(v) ** (1.0 / beta) + 1e-12)
    return step.astype(float)


def _assign_by_potential(
    net: Network,
    ch: np.ndarray,
    *,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each alive non-CH node to a CH using the paper's potential score.

    score(node->CH_j) = E_CH[j] / (dist(node, CH_j) + eps)

    Returns:
    - assigned_ch: array of length N with CH index for each node (or -1 if dead)
    - dist_to_ch: array of length N with dist(node, assigned CH) (NaN if dead)
    """
    n = int(net.n_nodes)
    assigned = np.full(n, -1, dtype=int)
    dist_to = np.full(n, np.nan, dtype=float)

    alive = net.alive_mask
    if ch.size == 0 or not np.any(alive):
        return assigned, dist_to

    ch = np.asarray(ch, dtype=int).reshape(-1)
    ch = ch[(ch >= 0) & (ch < n)]
    ch = np.unique(ch[alive[ch]])
    if ch.size == 0:
        return assigned, dist_to

    # Precompute CH energies
    e_ch = net.residual_energy[ch].astype(float)

    alive_idx = np.where(alive)[0]
    for i in alive_idx:
        ii = int(i)
        if ii in set(map(int, ch.tolist())):
            assigned[ii] = ii
            dist_to[ii] = 0.0
            continue
        d = net.dists[ii, ch].astype(float)
        score = e_ch / (d + float(eps))
        j = int(np.argmax(score))
        assigned[ii] = int(ch[j])
        dist_to[ii] = float(d[j])

    return assigned, dist_to


def emogjo_ch_fitness(net: Network, ch: np.ndarray, p: EMOGJOParams) -> Tuple[float, Dict[str, float]]:
    """Compute the EMOGJO CH-selection fitness (weighted sum).

    This is implemented as a *minimization* objective for compatibility with the framework.

    Components (paper names): MNE, INNC, DSCH, DCHBS, ND.

    Because some paper definitions can be ambiguous in sign/normalization, we provide
    a clear, monotone, bounded version by default (energy_mode="monotone").
    """
    alive = net.alive_mask
    n_alive = int(np.sum(alive))
    if n_alive <= 0:
        return 1.0, {"MNE": 1.0, "INNC": 1.0, "DSCH": 1.0, "DCHBS": 1.0, "ND": 1.0}

    ch = np.asarray(ch, dtype=int).reshape(-1)
    ch = ch[(ch >= 0) & (ch < net.n_nodes)]
    ch = np.unique(ch[alive[ch]])
    if ch.size == 0:
        # Worst
        return 1.0, {"MNE": 1.0, "INNC": 1.0, "DSCH": 1.0, "DCHBS": 1.0, "ND": 1.0}

    eps = float(p.eps)
    diag = float(net.diag) if float(getattr(net, "diag", 0.0)) > 0 else 1.0

    # Assignment for DSCH and ND
    assigned, dist_to = _assign_by_potential(net, ch, eps=eps)

    alive_idx = np.where(alive)[0]
    # DSCH: average node->CH distance (alive nodes only), normalized
    d_alive = dist_to[alive_idx]
    d_alive = d_alive[np.isfinite(d_alive)]
    DSCH = float(np.mean(d_alive) / (diag + eps)) if d_alive.size else 1.0
    DSCH = float(np.clip(DSCH, 0.0, 1.0))

    # DCHBS: average CH->BS distance normalized
    dchbs = float(np.mean(net.dists_to_sink[ch]) / (diag + eps))
    DCHBS = float(np.clip(dchbs, 0.0, 1.0))

    # ND: proxy for "number of CH" (lower is better => fewer CHs)
    ND = float(ch.size / max(1, n_alive))
    ND = float(np.clip(ND, 0.0, 1.0))

    # MNE: energy cost
    e_ratio = net.residual_energy[ch] / (net.initial_energy[ch] + eps)
    mean_e = float(np.mean(np.clip(e_ratio, 0.0, 1.0)))
    if str(p.energy_mode).lower() == "literal":
        # Literal-as-cost (discouraged): makes low energy look good.
        MNE = float(np.clip(mean_e, 0.0, 1.0))
    else:
        # Monotone cost: penalize low-energy CHs.
        MNE = float(np.clip(1.0 - mean_e, 0.0, 1.0))

    # INNC: CH separation cost via nearest-neighbor distance among CHs
    if ch.size <= 1:
        INNC = 1.0
    else:
        dmat = net.dists[np.ix_(ch, ch)].astype(float)
        np.fill_diagonal(dmat, np.inf)
        nn = np.min(dmat, axis=1)
        mean_nn = float(np.mean(nn[np.isfinite(nn)]))
        nn_norm = float(np.clip(mean_nn / (diag + eps), 0.0, 1.0))
        INNC = float(np.clip(1.0 - nn_norm, 0.0, 1.0))

    fit = (
        float(p.u1_mne) * MNE
        + float(p.u2_innc) * INNC
        + float(p.u3_dsch) * DSCH
        + float(p.u4_dchbs) * DCHBS
        + float(p.u5_nd) * ND
    )

    if not np.isfinite(fit):
        fit = 1.0

    return float(fit), {"MNE": MNE, "INNC": INNC, "DSCH": DSCH, "DCHBS": DCHBS, "ND": ND}


def run_emogjo_wsn(net: Network, fit_params: FitnessParams, params: EMOGJOParams) -> OptimizationResult:
    rng = np.random.default_rng(int(params.seed))

    pop = int(max(2, params.pop_size))
    dim = int(net.n_nodes)

    X = rng.normal(size=(pop, dim)).astype(float)
    fit_vals = np.full(pop, np.inf, dtype=float)

    nfe = 0

    alive_mask = getattr(net, "alive_mask", np.ones(dim, dtype=bool))

    def decode(i: int) -> np.ndarray:
        return decode_topk_bounded(
            X[i],
            alive_mask=alive_mask,
            rng=rng,
            k_min=int(params.min_ch),
            k_max=int(params.max_ch),
            use_sigmoid=True,
        )

    def evaluate(i: int) -> float:
        nonlocal nfe
        ch = decode(i)
        if bool(getattr(params, "use_paper_ch_fitness", False)):
            f, _ = emogjo_ch_fitness(net, ch, params)
        else:
            f, _ = fitness(net, ch, fit_params)
        nfe += 1
        return float(f)

    history: Dict[str, list] = {"best_f": [], "iter": []}

    # Fairness: target NFE ~= pop_size * n_iter (same scale as PSO/GWO/SO/GJO).
    # We evaluate the initial population once, then do (n_iter-1) update steps.
    n_iter = int(max(0, params.n_iter))
    if n_iter <= 0:
        best_ch = decode(0)
        return OptimizationResult(best_ch_indices=best_ch, best_fitness=float("inf"), history=history, nfe=0.0)

    for i in range(pop):
        fit_vals[i] = evaluate(i)

    best0 = int(np.argmin(fit_vals))
    history["best_f"].append(float(fit_vals[best0]))
    history["iter"].append(0)

    for it in range(1, n_iter):
        # Leaders (male/female) and prey
        order = np.argsort(fit_vals)
        male = int(order[0])
        female = int(order[1]) if pop > 1 else int(order[0])
        prey = X[male].copy()

        t = float(it) / max(1.0, float(n_iter))
        E0 = 2.0 * (1.0 - t)
        EE = float(params.c1) * E0 * (2.0 * float(rng.random()) - 1.0)

        X_new = X.copy()

        for i in range(pop):
            Xi = X[i]

            r1 = rng.random(size=dim)
            r2 = rng.random(size=dim)
            A = 2.0 * abs(EE) * r1 - abs(EE)
            C = 2.0 * r2
            rl = _levy_flight(rng, size=dim, beta=float(params.b_levy))

            if abs(EE) > 1.0:
                # Exploration: random peer + prey guidance
                j = int(rng.integers(0, pop))
                Xr = X[j]
                Y1 = Xr - A * rl * np.abs(C * Xr - Xi)
                Y2 = prey - A * rl * np.abs(C * prey - Xi)
                Xcand = 0.5 * (Y1 + Y2)
            else:
                # Exploitation: male/female guidance
                Xm = X[male]
                Xf = X[female]
                Y1 = Xm - A * rl * np.abs(C * Xm - Xi)
                Y2 = Xf - A * rl * np.abs(C * Xf - Xi)
                Xcand = 0.5 * (Y1 + Y2)

            x_clip = float(getattr(params, "x_clip", 20.0))
            if x_clip > 0:
                Xcand = np.clip(Xcand, -x_clip, x_clip)
            X_new[i] = Xcand

        # Greedy accept
        for i in range(pop):
            old_fit = float(fit_vals[i])
            Xi_old = X[i]
            X[i] = X_new[i]
            f_new = evaluate(i)
            if f_new < old_fit:
                fit_vals[i] = f_new
            else:
                X[i] = Xi_old

        best = int(np.argmin(fit_vals))
        history["best_f"].append(float(fit_vals[best]))
        history["iter"].append(int(it))

    best = int(np.argmin(fit_vals))
    best_ch = decode(best)
    return OptimizationResult(best_ch_indices=best_ch, best_fitness=float(fit_vals[best]), history=history, nfe=float(nfe))
