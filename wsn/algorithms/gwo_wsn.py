from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np

from ..models import Network
from ..fitness import fitness, FitnessParams
from .base import OptimizationResult


@dataclass
class GWOParams:
    n_wolves: int = 30
    n_iter: int = 100
    seed: int = 0
    min_ch: int = 1
    max_ch: int = 20


def _decode_wolf(
    x: np.ndarray,
    params: GWOParams,
    rng: np.random.Generator,
    alive_mask: np.ndarray,
) -> np.ndarray:
    n = int(x.shape[0])
    probs = 1.0 / (1.0 + np.exp(-x))

    alive_idx = np.where(alive_mask)[0]
    if alive_idx.size == 0:
        return np.array([], dtype=int)

    k_min = int(max(1, params.min_ch))
    k_max = int(min(params.max_ch, alive_idx.size))
    k = int(rng.integers(k_min, k_max + 1)) if k_max >= k_min else int(k_max)

    alive_scores = probs[alive_idx]
    idx_sorted = alive_idx[np.argsort(-alive_scores)]
    ch = idx_sorted[:k]
    return np.sort(np.unique(ch))


def run_gwo_wsn(
    net: Network,
    fit_params: FitnessParams,
    gwo_params: GWOParams,
) -> OptimizationResult:
    rng = np.random.default_rng(int(gwo_params.seed))
    n = int(net.n_nodes)

    X = rng.normal(size=(int(gwo_params.n_wolves), n))

    alpha = X[0].copy()
    beta = X[1].copy() if int(gwo_params.n_wolves) > 1 else X[0].copy()
    delta = X[2].copy() if int(gwo_params.n_wolves) > 2 else X[0].copy()

    f_alpha = f_beta = f_delta = np.inf

    history: Dict[str, list] = {"best_f": [], "iter": []}

    alive_mask = getattr(net, "alive_mask", np.ones(n, dtype=bool))

    for t in range(int(gwo_params.n_iter)):
        a = 2.0 - 2.0 * (float(t) / max(1, int(gwo_params.n_iter)))

        # Evaluate and update alpha/beta/delta
        for i in range(int(gwo_params.n_wolves)):
            ch = _decode_wolf(X[i], gwo_params, rng, alive_mask)
            f, _ = fitness(net, ch, fit_params)

            if f < f_alpha:
                delta, f_delta = beta, f_beta
                beta, f_beta = alpha, f_alpha
                alpha, f_alpha = X[i].copy(), f
            elif f < f_beta:
                delta, f_delta = beta, f_beta
                beta, f_beta = X[i].copy(), f
            elif f < f_delta:
                delta, f_delta = X[i].copy(), f

        # Position updates
        for i in range(int(gwo_params.n_wolves)):
            r1 = rng.random(size=n)
            r2 = rng.random(size=n)
            A1 = 2.0 * a * r1 - a
            C1 = 2.0 * r2

            r1 = rng.random(size=n)
            r2 = rng.random(size=n)
            A2 = 2.0 * a * r1 - a
            C2 = 2.0 * r2

            r1 = rng.random(size=n)
            r2 = rng.random(size=n)
            A3 = 2.0 * a * r1 - a
            C3 = 2.0 * r2

            D_alpha = np.abs(C1 * alpha - X[i])
            D_beta = np.abs(C2 * beta - X[i])
            D_delta = np.abs(C3 * delta - X[i])

            X1 = alpha - A1 * D_alpha
            X2 = beta - A2 * D_beta
            X3 = delta - A3 * D_delta

            X[i] = (X1 + X2 + X3) / 3.0

        history["best_f"].append(float(f_alpha))
        history["iter"].append(int(t))

    best_ch = _decode_wolf(alpha, gwo_params, rng, alive_mask)
    return OptimizationResult(best_ch_indices=best_ch, best_fitness=float(f_alpha), history=history)
