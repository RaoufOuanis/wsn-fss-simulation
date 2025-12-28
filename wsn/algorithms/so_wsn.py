from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..models import Network
from ..fitness import FitnessParams, fitness
from .base import OptimizationResult
from .decode import decode_topk_bounded


@dataclass
class SOParams:
    """Snake Optimizer (SO) parameters for per-round CH selection.

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

    # SO control (kept simple + configurable; avoids extra heavy logging)
    step_explore: float = 0.6
    step_exploit: float = 0.3
    noise_scale: float = 0.05

    # Optional anti-stagnation (disabled by default for strict fairness)
    stagnation_iters: int = 0
    reinit_prob: float = 0.0
    reinit_frac: float = 0.2


def run_so_wsn(
    net: Network,
    fit_params: FitnessParams,
    so_params: SOParams,
) -> OptimizationResult:
    rng = np.random.default_rng(int(so_params.seed))
    n = int(net.n_nodes)
    alive_mask = getattr(net, "alive_mask", np.ones(n, dtype=bool))

    pop = int(max(2, so_params.pop_size))
    if pop % 2 != 0:
        pop += 1  # enforce even split male/female

    # Representation requested by the user guide: X in [0,1]
    X = rng.random(size=(pop, n), dtype=float)

    fit_vals = np.full(pop, np.inf, dtype=float)
    best_idx = 0
    best_fit = float("inf")

    half = pop // 2
    male_idx = np.arange(0, half, dtype=int)
    female_idx = np.arange(half, pop, dtype=int)

    nfe = 0

    def decode(i: int) -> np.ndarray:
        return decode_topk_bounded(
            X[i],
            alive_mask=alive_mask,
            rng=rng,
            k_min=int(so_params.min_ch),
            k_max=int(so_params.max_ch),
            use_sigmoid=True,
        )

    def evaluate(i: int) -> float:
        nonlocal nfe
        ch = decode(i)
        f, _ = fitness(net, ch, fit_params)
        nfe += 1
        return float(f)

    # Initial evaluation
    for i in range(pop):
        fit_vals[i] = evaluate(i)

    best_idx = int(np.argmin(fit_vals))
    best_fit = float(fit_vals[best_idx])

    def best_in(group: np.ndarray) -> int:
        j = int(group[int(np.argmin(fit_vals[group]))])
        return j

    best_male = best_in(male_idx)
    best_female = best_in(female_idx)

    # Per-agent stagnation counters (optional)
    no_improve = np.zeros(pop, dtype=int)

    history: Dict[str, list] = {"best_f": [], "iter": []}

    for it in range(int(so_params.n_iter)):
        # Simple phase schedule (temperature decreasing): exploration early, exploitation late
        t = float(it) / max(1.0, float(so_params.n_iter))
        explore = t < 0.5

        step = float(so_params.step_explore if explore else so_params.step_exploit)
        noise = float(so_params.noise_scale) * (1.0 - t)

        X_new = X.copy()

        # Vectorized-ish updates by gender; evaluation remains per-agent.
        # Males
        r = rng.random(size=(half, n))
        if explore:
            # drift towards random females + noise
            mates = rng.integers(low=0, high=female_idx.size, size=half)
            Xm = X[male_idx]
            Xf = X[female_idx[mates]]
            X_new[male_idx] = Xm + step * r * (Xf - Xm) + noise * rng.normal(size=(half, n))
        else:
            # exploit: move towards global and best male
            Xm = X[male_idx]
            Xg = X[best_idx]
            Xbm = X[best_male]
            X_new[male_idx] = Xm + step * r * (0.5 * (Xg - Xm) + 0.5 * (Xbm - Xm)) + noise * rng.normal(size=(half, n))

        # Females
        r = rng.random(size=(pop - half, n))
        if explore:
            # drift towards random males + noise
            mates = rng.integers(low=0, high=male_idx.size, size=pop - half)
            Xf = X[female_idx]
            Xm = X[male_idx[mates]]
            X_new[female_idx] = Xf + step * r * (Xm - Xf) + noise * rng.normal(size=(pop - half, n))
        else:
            Xf = X[female_idx]
            Xg = X[best_idx]
            Xbf = X[best_female]
            X_new[female_idx] = Xf + step * r * (0.5 * (Xg - Xf) + 0.5 * (Xbf - Xf)) + noise * rng.normal(size=(pop - half, n))

        # Clip to [0,1] as required
        X_new = np.clip(X_new, 0.0, 1.0)

        # Greedy selection
        for i in range(pop):
            old_fit = float(fit_vals[i])
            Xi_old = X[i]
            X[i] = X_new[i]
            f_new = evaluate(i)

            if f_new < old_fit:
                fit_vals[i] = f_new
                no_improve[i] = 0
            else:
                X[i] = Xi_old
                no_improve[i] += 1

                # Optional anti-stagnation (disabled by default)
                if int(so_params.stagnation_iters) > 0 and float(so_params.reinit_prob) > 0.0:
                    if no_improve[i] >= int(so_params.stagnation_iters) and rng.random() < float(so_params.reinit_prob):
                        frac = float(np.clip(float(so_params.reinit_frac), 0.0, 1.0))
                        m = int(max(1, round(frac * n)))
                        j = rng.choice(np.arange(n), size=m, replace=False)
                        X[i, j] = rng.random(size=m)
                        X[i] = np.clip(X[i], 0.0, 1.0)
                        fit_vals[i] = evaluate(i)
                        no_improve[i] = 0

        # Update leaders
        best_idx = int(np.argmin(fit_vals))
        best_fit = float(fit_vals[best_idx])
        best_male = best_in(male_idx)
        best_female = best_in(female_idx)

        history["best_f"].append(best_fit)
        history["iter"].append(int(it))

    best_ch = decode(best_idx)
    return OptimizationResult(best_ch_indices=best_ch, best_fitness=float(best_fit), history=history, nfe=float(nfe))
