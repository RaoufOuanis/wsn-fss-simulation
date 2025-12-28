from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..models import Network
from ..fitness import FitnessParams, fitness
from .base import OptimizationResult
from .decode import decode_topk_bounded


@dataclass
class GJOParams:
    """Golden Jackal Optimization (GJO) parameters for per-round CH selection.

    Notes for fairness:
    - Uses the same bounded top-k decoding as PSO/GWO/ABC.
    - Evaluates candidates via the same Fitness_t(H0) (which performs Repair_t).
    - Iteration budget n_iter is expected to be budget-matched externally.
    """

    pop_size: int = 30
    n_iter: int = 60
    seed: int = 0
    min_ch: int = 1
    max_ch: int = 20

    # Exploration/exploitation schedule
    # E0(t) = 2*(1 - t/T) gives a decreasing envelope in [0,2]
    # Actual E sampled in [-E0, +E0]. Exploration if |E| >= 1.

    noise_scale: float = 0.02


def run_gjo_wsn(
    net: Network,
    fit_params: FitnessParams,
    gjo_params: GJOParams,
) -> OptimizationResult:
    rng = np.random.default_rng(int(gjo_params.seed))
    n = int(net.n_nodes)
    alive_mask = getattr(net, "alive_mask", np.ones(n, dtype=bool))

    pop = int(max(2, gjo_params.pop_size))

    # Representation requested by the user guide: X in [0,1]
    X = rng.random(size=(pop, n), dtype=float)
    fit_vals = np.full(pop, np.inf, dtype=float)

    nfe = 0

    def decode_scores(x: np.ndarray) -> np.ndarray:
        return decode_topk_bounded(
            x,
            alive_mask=alive_mask,
            rng=rng,
            k_min=int(gjo_params.min_ch),
            k_max=int(gjo_params.max_ch),
            use_sigmoid=True,
        )

    def evaluate(i: int) -> float:
        nonlocal nfe
        ch = decode_scores(X[i])
        f, _ = fitness(net, ch, fit_params)
        nfe += 1
        return float(f)

    for i in range(pop):
        fit_vals[i] = evaluate(i)

    def update_leaders() -> tuple[int, int]:
        order = np.argsort(fit_vals)
        male = int(order[0])
        female = int(order[1]) if pop > 1 else int(order[0])
        return male, female

    male, female = update_leaders()

    history: Dict[str, list] = {"best_f": [], "iter": []}

    for it in range(int(gjo_params.n_iter)):
        t = float(it) / max(1.0, float(gjo_params.n_iter))
        E0 = 2.0 * (1.0 - t)  # decreases from 2 -> 0
        E = E0 * (2.0 * float(rng.random()) - 1.0)

        X_new = X.copy()

        # Pre-sample noise
        noise = float(gjo_params.noise_scale) * (1.0 - t)

        for i in range(pop):
            Xi = X[i]

            r1 = rng.random(size=n)
            r2 = rng.random(size=n)
            A = 2.0 * abs(E) * r1 - abs(E)
            C = 2.0 * r2

            if abs(E) >= 1.0:
                # Exploration: guided by a random peer
                j = int(rng.integers(0, pop))
                Xr = X[j]
                D = np.abs(C * Xr - Xi)
                X_candidate = Xr - A * D
            else:
                # Exploitation: guided by the two leaders
                Xm = X[male]
                Xf = X[female]
                Dm = np.abs(C * Xm - Xi)
                Df = np.abs(C * Xf - Xi)
                X1 = Xm - A * Dm
                X2 = Xf - A * Df
                X_candidate = 0.5 * (X1 + X2)

            if noise > 0.0:
                X_candidate = X_candidate + noise * rng.normal(size=n)

            X_new[i] = np.clip(X_candidate, 0.0, 1.0)

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

        male, female = update_leaders()
        history["best_f"].append(float(fit_vals[male]))
        history["iter"].append(int(it))

    best_ch = decode_scores(X[male])
    return OptimizationResult(best_ch_indices=best_ch, best_fitness=float(fit_vals[male]), history=history, nfe=float(nfe))
