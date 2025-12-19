from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np

from ..models import Network
from ..fitness import fitness, FitnessParams
from .base import OptimizationResult


@dataclass
class PSOParams:
    n_particles: int = 30
    n_iter: int = 100
    w_inertia: float = 0.7
    c1: float = 1.5
    c2: float = 1.5
    vmax: float = 4.0
    seed: int = 0
    min_ch: int = 1
    max_ch: int = 20


def _decode_particle(
    x: np.ndarray,
    params: PSOParams,
    rng: np.random.Generator,
    alive_mask: np.ndarray,
) -> np.ndarray:
    """Convert continuous position to a CH index set.

    Notes:
    - Dead nodes are excluded.
    - k is sampled in [min_ch, max_ch] but cannot exceed the number of alive nodes.
    """
    n = int(x.shape[0])
    probs = 1.0 / (1.0 + np.exp(-x))  # sigmoid

    alive_idx = np.where(alive_mask)[0]
    if alive_idx.size == 0:
        return np.array([], dtype=int)

    k_min = int(max(1, params.min_ch))
    k_max = int(min(params.max_ch, alive_idx.size))
    k = int(rng.integers(k_min, k_max + 1)) if k_max >= k_min else int(k_max)

    # rank only alive nodes
    alive_scores = probs[alive_idx]
    idx_sorted = alive_idx[np.argsort(-alive_scores)]
    ch = idx_sorted[:k]
    return np.sort(np.unique(ch))


def run_pso_wsn(
    net: Network,
    fit_params: FitnessParams,
    pso_params: PSOParams,
) -> OptimizationResult:
    rng = np.random.default_rng(int(pso_params.seed))
    n = int(net.n_nodes)

    x = rng.normal(size=(int(pso_params.n_particles), n))
    v = rng.normal(scale=0.1, size=(int(pso_params.n_particles), n))
    
    pbest = x.copy()
    pbest_fit = np.full(int(pso_params.n_particles), np.inf)
    gbest = x[0].copy()
    gbest_fit = np.inf

    history: Dict[str, list] = {"best_f": [], "iter": []}

    alive_mask = getattr(net, "alive_mask", np.ones(n, dtype=bool))

    for it in range(int(pso_params.n_iter)):
        for i in range(int(pso_params.n_particles)):
            ch = _decode_particle(x[i], pso_params, rng, alive_mask)
            f_val, _ = fitness(net, ch, fit_params)
            if f_val < pbest_fit[i]:
                pbest_fit[i] = f_val
                pbest[i] = x[i].copy()
            if f_val < gbest_fit:
                gbest_fit = f_val
                gbest = x[i].copy()

        # update velocities and positions
        for i in range(int(pso_params.n_particles)):
            r1 = rng.random(size=n)
            r2 = rng.random(size=n)
            v[i] = (
                float(pso_params.w_inertia) * v[i]
                + float(pso_params.c1) * r1 * (pbest[i] - x[i])
                + float(pso_params.c2) * r2 * (gbest - x[i])
            )
            v[i] = np.clip(v[i], -float(pso_params.vmax), float(pso_params.vmax))
            x[i] += v[i]

        history["best_f"].append(float(gbest_fit))
        history["iter"].append(int(it))

    best_ch = _decode_particle(gbest, pso_params, rng, alive_mask)
    return OptimizationResult(best_ch_indices=best_ch, best_fitness=float(gbest_fit), history=history)
