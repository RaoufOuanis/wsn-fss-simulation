from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..models import Network
from ..fitness import FitnessParams, fitness
from .base import OptimizationResult
from .decode import decode_topk_bounded


@dataclass
class ESOGJOParams:
    """ESO-GJO baseline (strict-fair variant for this repo).

    Paper view (high-level):
      - ESO (enhanced Snake Optimizer) selects CHs (with Brownian exploitation)
      - GJO is used for routing.

    Repo adaptation for fairness:
      - We keep the repo's routing / repair / multi-hop inside `fitness()`.
      - ESO-GJO is implemented as an optimizer over CH selection only.
      - Uses the same bounded top-k decode and the same `fitness()` as PSO/GWO/ABC/SO/GJO/EMOGJO.
      - NFE is counted as actual `fitness()` calls, targeting ~ pop_size * n_iter.
    """

    pop_size: int = 30
    n_iter: int = 60
    seed: int = 0

    min_ch: int = 1
    max_ch: int = 20

    # ESO schedule (paper defaults where given)
    c1: float = 0.5  # Q coefficient
    c2: float = 0.05  # exploration step coefficient

    # Brownian exploitation parameters
    brown_mu: float = 0.0
    brown_sigma: float = 1.0

    # Optional small Gaussian jitter (kept tiny; 0 disables)
    noise_scale: float = 0.0

    # Backward-learning init (GJO-inspired) -- lightweight, keeps NFE budget unchanged
    use_backward_learning_init: bool = True
    lcg_a: int = 1664525
    lcg_c: int = 1013904223
    lcg_m: int = 2**32

    # Worst replacement ("eggs") applied *within* the per-iteration candidate generation
    # so we do not increase per-iteration evaluation count.
    egg_frac: float = 0.05
    egg_prob: float = 0.10


def _lcg_uniform01(seed: int, a: int, c: int, m: int, n: int) -> np.ndarray:
    """Generate n floats in [0,1) using a simple LCG (deterministic)."""
    x = int(seed) & 0xFFFFFFFF
    out = np.empty(int(n), dtype=float)
    for i in range(int(n)):
        x = (a * x + c) % m
        out[i] = float(x) / float(m)
    return out


def run_esogjo_wsn(net: Network, fit_params: FitnessParams, params: ESOGJOParams) -> OptimizationResult:
    rng = np.random.default_rng(int(params.seed))

    dim = int(net.n_nodes)
    alive_mask = getattr(net, "alive_mask", np.ones(dim, dtype=bool))

    pop = int(max(2, params.pop_size))
    if pop % 2 != 0:
        pop += 1  # enforce even male/female split

    n_iter = int(max(0, params.n_iter))
    if n_iter <= 0:
        # Still return a valid decoded solution.
        x0 = rng.random(size=dim, dtype=float)
        ch0 = decode_topk_bounded(
            x0,
            alive_mask=alive_mask,
            rng=rng,
            k_min=int(params.min_ch),
            k_max=int(params.max_ch),
            use_sigmoid=True,
        )
        return OptimizationResult(best_ch_indices=ch0, best_fitness=float("inf"), history={"best_f": [], "iter": []}, nfe=0.0)

    # Representation in [0,1] (same convention as SO/GJO)
    X = rng.random(size=(pop, dim), dtype=float)

    # Backward learning init without increasing population size:
    # replace the second half with "opposite" solutions derived from the first half.
    if bool(params.use_backward_learning_init):
        half = pop // 2
        A = np.min(X[:half], axis=0)
        B = np.max(X[:half], axis=0)
        z = _lcg_uniform01(int(params.seed) + 17, int(params.lcg_a), int(params.lcg_c), int(params.lcg_m), half)
        for i in range(half):
            opp = float(z[i]) * (A + B) - X[i]
            X[half + i] = np.clip(opp, 0.0, 1.0)

    fit_vals = np.full(pop, np.inf, dtype=float)
    nfe = 0

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
        f, _ = fitness(net, ch, fit_params)
        nfe += 1
        return float(f)

    # Initial evaluation (counts toward the fairness budget)
    for i in range(pop):
        fit_vals[i] = evaluate(i)

    history: Dict[str, list] = {"best_f": [float(np.min(fit_vals))], "iter": [0]}

    half = pop // 2
    male_idx = np.arange(0, half, dtype=int)
    female_idx = np.arange(half, pop, dtype=int)

    def best_in(group: np.ndarray) -> int:
        return int(group[int(np.argmin(fit_vals[group]))])

    best_idx = int(np.argmin(fit_vals))
    best_fit = float(fit_vals[best_idx])

    # Main loop: do (n_iter-1) update steps to keep NFE scale aligned with other optimizers.
    for it in range(1, n_iter):
        # Paper-like schedules
        # Temp = exp(-t/T) in (0,1]
        Temp = float(np.exp(-float(it) / max(1.0, float(n_iter))))
        # Q = c1 * exp((t-T)/T)
        Q = float(params.c1) * float(np.exp((float(it) - float(n_iter)) / max(1.0, float(n_iter))))

        food = X[best_idx].copy()

        X_new = X.copy()

        # Identify a few worst individuals (for "eggs" reset)
        egg_n = int(max(0, round(float(params.egg_frac) * float(pop))))
        if egg_n > 0:
            worst = np.argsort(fit_vals)[-egg_n:]
        else:
            worst = np.array([], dtype=int)

        # Exploration when Q < 0.25
        explore = Q < 0.25

        # Pre-sample Brownian and jitter
        brown = np.abs(rng.normal(loc=float(params.brown_mu), scale=max(1e-12, float(params.brown_sigma)), size=(pop, dim)))
        noise = float(params.noise_scale) * (1.0 - float(it) / max(1.0, float(n_iter)))

        # Update males
        for idx in male_idx:
            Xi = X[idx]

            if worst.size and idx in worst and rng.random() < float(params.egg_prob):
                cand = rng.random(size=dim)
            elif explore:
                j = int(rng.choice(male_idx))
                Xr = X[j]
                Ar = float(np.exp(-float(fit_vals[j]) / (float(fit_vals[idx]) + 1e-12)))
                step = float(params.c2) * Ar
                sign = 1.0 if rng.random() < 0.5 else -1.0
                cand = Xr + sign * step * rng.random(size=dim)
            else:
                # Exploitation with Brownian
                if Temp > 0.6:
                    sign = 1.0 if rng.random() < 0.5 else -1.0
                    cand = food + sign * brown[idx] * Temp * rng.random(size=dim) * (food - Xi)
                else:
                    # Fight or mating with a random female
                    j = int(rng.choice(female_idx))
                    Xp = X[j]
                    r = rng.random(size=dim)
                    if rng.random() < 0.5:
                        cand = Xi + Temp * r * (Xi - Xp)
                    else:
                        cand = Xi + Temp * r * (Xp - Xi)

            if noise > 0.0:
                cand = cand + noise * rng.normal(size=dim)
            X_new[idx] = np.clip(cand, 0.0, 1.0)

        # Update females
        for idx in female_idx:
            Xi = X[idx]

            if worst.size and idx in worst and rng.random() < float(params.egg_prob):
                cand = rng.random(size=dim)
            elif explore:
                j = int(rng.choice(female_idx))
                Xr = X[j]
                Ar = float(np.exp(-float(fit_vals[j]) / (float(fit_vals[idx]) + 1e-12)))
                step = float(params.c2) * Ar
                sign = 1.0 if rng.random() < 0.5 else -1.0
                cand = Xr + sign * step * rng.random(size=dim)
            else:
                if Temp > 0.6:
                    sign = 1.0 if rng.random() < 0.5 else -1.0
                    cand = food + sign * brown[idx] * Temp * rng.random(size=dim) * (food - Xi)
                else:
                    j = int(rng.choice(male_idx))
                    Xp = X[j]
                    r = rng.random(size=dim)
                    if rng.random() < 0.5:
                        cand = Xi + Temp * r * (Xi - Xp)
                    else:
                        cand = Xi + Temp * r * (Xp - Xi)

            if noise > 0.0:
                cand = cand + noise * rng.normal(size=dim)
            X_new[idx] = np.clip(cand, 0.0, 1.0)

        # Greedy accept (one evaluation per individual)
        for i in range(pop):
            old_fit = float(fit_vals[i])
            Xi_old = X[i]
            X[i] = X_new[i]
            f_new = evaluate(i)
            if f_new < old_fit:
                fit_vals[i] = f_new
            else:
                X[i] = Xi_old

        best_idx = int(np.argmin(fit_vals))
        best_fit = float(fit_vals[best_idx])
        history["best_f"].append(best_fit)
        history["iter"].append(int(it))

    best_ch = decode(best_idx)
    return OptimizationResult(best_ch_indices=best_ch, best_fitness=float(best_fit), history=history, nfe=float(nfe))
