from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np

from ..models import Network
from ..fitness import fitness, FitnessParams
from .decode import decode_topk_bounded
from .base import OptimizationResult


@dataclass
class ABCParams:
    n_food_sources: int = 20
    n_iter: int = 100
    limit: int = 10
    seed: int = 0
    min_ch: int = 1
    max_ch: int = 20


def _decode_food(
    x: np.ndarray,
    params: ABCParams,
    rng: np.random.Generator,
    alive_mask: np.ndarray,
) -> np.ndarray:
    return decode_topk_bounded(
        x,
        alive_mask=alive_mask,
        rng=rng,
        k_min=int(params.min_ch),
        k_max=int(params.max_ch),
        use_sigmoid=True,
    )


def run_abc_wsn(
    net: Network,
    fit_params: FitnessParams,
    abc_params: ABCParams,
) -> OptimizationResult:
    rng = np.random.default_rng(int(abc_params.seed))
    n = int(net.n_nodes)

    X = rng.normal(size=(int(abc_params.n_food_sources), n))
    fit_vals = np.full(int(abc_params.n_food_sources), np.inf)
    trial = np.zeros(int(abc_params.n_food_sources), dtype=int)

    alive_mask = getattr(net, "alive_mask", np.ones(n, dtype=bool))

    nfe = 0

    def evaluate(i: int) -> None:
        ch = _decode_food(X[i], abc_params, rng, alive_mask)
        f, _ = fitness(net, ch, fit_params)
        fit_vals[i] = f
        nonlocal nfe
        nfe += 1

    for i in range(int(abc_params.n_food_sources)):
        evaluate(i)

    history: Dict[str, list] = {"best_f": [], "iter": []}

    for it in range(int(abc_params.n_iter)):
        # Employed bees
        for i in range(int(abc_params.n_food_sources)):
            k = int(rng.integers(0, int(abc_params.n_food_sources)))
            while k == i:
                k = int(rng.integers(0, int(abc_params.n_food_sources)))

            phi = rng.uniform(-1.0, 1.0, size=n)
            v = X[i] + phi * (X[i] - X[k])

            old = X[i].copy()
            old_fit = float(fit_vals[i])

            X[i] = v
            evaluate(i)

            if float(fit_vals[i]) >= old_fit:
                X[i] = old
                fit_vals[i] = old_fit
                trial[i] += 1
            else:
                trial[i] = 0

        # Onlooker bees
        inv_fit = 1.0 / (1.0 + fit_vals)
        prob = inv_fit / (np.sum(inv_fit) + 1e-12)

        for _ in range(int(abc_params.n_food_sources)):
            i = int(rng.choice(np.arange(int(abc_params.n_food_sources)), p=prob))
            k = int(rng.integers(0, int(abc_params.n_food_sources)))
            while k == i:
                k = int(rng.integers(0, int(abc_params.n_food_sources)))

            phi = rng.uniform(-1.0, 1.0, size=n)
            v = X[i] + phi * (X[i] - X[k])

            old = X[i].copy()
            old_fit = float(fit_vals[i])

            X[i] = v
            evaluate(i)

            if float(fit_vals[i]) >= old_fit:
                X[i] = old
                fit_vals[i] = old_fit
                trial[i] += 1
            else:
                trial[i] = 0

        # Scout bees
        for i in range(int(abc_params.n_food_sources)):
            if int(trial[i]) > int(abc_params.limit):
                X[i] = rng.normal(size=n)
                evaluate(i)
                trial[i] = 0

        best_idx = int(np.argmin(fit_vals))
        history["best_f"].append(float(fit_vals[best_idx]))
        history["iter"].append(int(it))

    best_idx = int(np.argmin(fit_vals))
    best_ch = _decode_food(X[best_idx], abc_params, rng, alive_mask)
    best_fit = float(fit_vals[best_idx])
    return OptimizationResult(best_ch_indices=best_ch, best_fitness=best_fit, history=history, nfe=float(nfe))
