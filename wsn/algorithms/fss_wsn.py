# wsn/algorithms/fss_wsn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..fitness import FitnessParams, fitness
from ..models import Network
from .base import OptimizationResult


@dataclass
class FSSParams:
    """FSS-WSN parameters (executed per round).

    This implementation is aligned with the paper description:

    - Phase I (learning): GRASP-style greedy randomized construction that
      iteratively adds cluster-heads (CHs) until coverage (Rc) is achieved
      or a safety limit Kmax is reached; followed by a bounded local search.
    - Phase II (intensification): compute a fixed set from an elite pool and
      restart GRASP biased by that fixed set; followed by the same local search.

    Practical performance notes
    ---------------------------
    - Distances are never recomputed. Network caches a full distance matrix.
    - Coverage is managed via a cached within-Rc adjacency and incremental
      updates (no repeated assign_clusters calls during construction).
    - Fitness evaluations are memoized intra-round (safe because energy is fixed
      during CH selection).
    """

    # Reproducibility
    seed: int = 0

    # Iteration budget: if max_iter1/max_iter2 are 0, they are derived from n_iter
    n_iter: int = 60
    max_iter1: int = 0
    max_iter2: int = 0

    # Elite pool (size B) and fixed set thresholds
    elite_size: int = 10
    tau: float = 0.6
    theta: float = 0.3

    # GRASP parameters
    gamma: float = 0.2  # RCL aggressiveness in [0,1]
    alpha1: float = 0.5  # energy weight
    alpha2: float = 0.3  # sink-centrality weight
    alpha3: float = 0.2  # density weight

    # Local search parameters
    Lmax: int = 10
    k_nn: int = 10

    # Safety cap on the CH set size during construction
    Kmax: int = 0

    # Optional elite diversity control (Jaccard distance threshold); 0 disables
    delta: float = 0.0

    # Phase control
    use_phase2: bool = True


def _sorted_key(ch: np.ndarray) -> Tuple[int, ...]:
    ch = np.asarray(ch, dtype=int).reshape(-1)
    if ch.size == 0:
        return tuple()
    return tuple(sorted(set(map(int, ch.tolist()))))


def _jaccard_distance(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    if union <= 0:
        return 0.0
    return 1.0 - (inter / union)


def _default_kmax(n_alive: int) -> int:
    # Safety cap: large enough to reach coverage in typical settings without
    # allowing degenerate huge CH sets.
    # For N=100 -> 30; N=200 -> 60; N=500 -> 150
    return int(np.clip(int(round(0.30 * n_alive)), 5, max(5, n_alive)))


class FSSWSN:
    """Fixed Set Search for WSN (FSS-WSN), executed per round."""

    def __init__(self, net: Network, fit_params: FitnessParams, params: FSSParams, rng: np.random.Generator):
        self.net = net
        self.fit_params = fit_params
        self.p = params
        self.rng = rng

        # Intra-round memoization: key(tuple(sorted(CH))) -> fitness float
        self._fit_cache: Dict[Tuple[int, ...], float] = {}

        # NFE counters:
        # - nfe: number of *actual* fitness evaluations (cache misses)
        # - nfe_requests: number of eval_fitness calls (hits + misses)
        self.nfe: int = 0
        self.nfe_requests: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0

        # Cache arrays used by GRASP/local search (geometry-only)
        cache = net.get_grasp_cache(rc=float(fit_params.rc), k_nn=int(max(1, params.k_nn)))
        self.within_rc = cache["within_rc"]  # (N,N) bool
        self.rho = cache["rho"]              # (N,) float
        self.sink_term = cache["sink_term"]  # (N,) float
        self.knn = cache["knn"]              # (N,k_nn) int

    # ----------------------------
    # Fitness evaluation with cache + NFE counting
    # ----------------------------
    def eval_fitness(self, ch: np.ndarray) -> float:
        """Evaluate fitness with memoization.

        NFE definition used here:
          - nfe increments only on cache miss (i.e., actual call to fitness()).
          - nfe_requests counts total eval_fitness calls (hits + misses).
        """
        self.nfe_requests += 1

        key = _sorted_key(ch)
        cached = self._fit_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return float(cached)

        f, _ = fitness(self.net, np.fromiter(key, dtype=int), self.fit_params)
        f = float(f)
        self._fit_cache[key] = f

        self.nfe += 1
        self.cache_misses += 1
        return f

    # ----------------------------
    # GRASP construction (coverage-driven)
    # ----------------------------
    def construct(self, start_set: Optional[Sequence[int]] = None, kmax: int = 0) -> np.ndarray:
        """Build a CH set using greedy randomized construction with RCL.

        Construction adds nodes until all alive nodes are covered within Rc
        or a safety size cap Kmax is reached.
        """
        alive = self.net.alive_mask
        n_alive = int(np.sum(alive))
        if n_alive <= 0:
            return np.array([], dtype=int)

        Kmax = int(kmax) if int(kmax) > 0 else _default_kmax(n_alive)

        # Normalize greedy weights (robust to misconfigured inputs)
        a1, a2, a3 = float(self.p.alpha1), float(self.p.alpha2), float(self.p.alpha3)
        s = a1 + a2 + a3
        if s <= 0:
            a1, a2, a3 = 1.0, 0.0, 0.0
        else:
            a1, a2, a3 = a1 / s, a2 / s, a3 / s

        # Base greedy score h_t(i): energy + sink centrality + static density
        energy_ratio = self.net.residual_energy / (self.net.initial_energy + 1e-12)
        energy_ratio = np.clip(energy_ratio, 0.0, 1.0)
        h = a1 * energy_ratio + a2 * self.sink_term + a3 * self.rho

        # Start from optional fixed set
        is_ch = np.zeros(self.net.n_nodes, dtype=bool)
        H: List[int] = []
        if start_set is not None:
            start = np.unique(np.asarray(start_set, dtype=int).reshape(-1))
            start = start[alive[start]]
            for i in start.tolist():
                ii = int(i)
                if not is_ch[ii]:
                    is_ch[ii] = True
                    H.append(ii)

        # Compute uncovered (alive nodes not covered by current H)
        uncovered = alive.copy()
        if H:
            covered = np.any(self.within_rc[np.asarray(H, dtype=int)], axis=0)
            uncovered &= ~covered

        # Incremental coverage counts: number of currently uncovered nodes each node can cover
        cover_counts = self.within_rc[:, uncovered].sum(axis=1).astype(np.int32)
        cover_counts[~alive] = 0
        cover_counts[is_ch] = 0

        gamma = float(np.clip(self.p.gamma, 0.0, 1.0))

        # Main loop: add CHs until coverage is achieved or Kmax reached
        while bool(np.any(uncovered)) and len(H) < Kmax:
            cand_mask = alive & (~is_ch) & (cover_counts > 0)
            cand_idx = np.where(cand_mask)[0]
            if cand_idx.size == 0:
                break  # should be rare because each uncovered node can cover itself

            vals = h[cand_idx]
            hmax = float(np.max(vals))
            hmin = float(np.min(vals))
            thr = hmax - gamma * (hmax - hmin)

            rcl_idx = cand_idx[vals >= thr]
            u = int(self.rng.choice(rcl_idx))
            H.append(u)
            is_ch[u] = True

            # Update uncovered and cover_counts incrementally
            newly_covered = self.within_rc[u] & uncovered
            if np.any(newly_covered):
                uncovered[newly_covered] = False
                cover_counts -= self.within_rc[:, newly_covered].sum(axis=1).astype(np.int32)

            cover_counts[u] = 0

        # Ensure non-empty solution
        if not H:
            alive_idx = np.where(alive)[0]
            best = int(alive_idx[int(np.argmax(h[alive_idx]))])
            H = [best]

        return np.asarray(H, dtype=int)

    # ----------------------------
    # Local search (swap, kNN, first-improvement)
    # ----------------------------
    def local_search(self, ch: np.ndarray) -> np.ndarray:
        """Apply bounded local search with kNN-restricted swap neighborhood.

        Runtime-safe choice:
        - Each improving move samples a single CH u and tests swaps with v drawn
          from kNN(u), accepting first improvement.
        - Fitness evals bounded by ~O(Lmax*k_nn) (plus cache hits).
        """
        Lmax = int(self.p.Lmax)
        if Lmax <= 0:
            return np.asarray(ch, dtype=int)

        alive = self.net.alive_mask
        cur = np.unique(np.asarray(ch, dtype=int).reshape(-1))
        cur = cur[alive[cur]]
        if cur.size == 0:
            return np.array([], dtype=int)

        cur_set: Set[int] = set(map(int, cur.tolist()))
        cur_f = self.eval_fitness(cur)

        is_ch = np.zeros(self.net.n_nodes, dtype=bool)
        is_ch[list(cur_set)] = True

        for _ in range(Lmax):
            u = int(self.rng.choice(np.fromiter(cur_set, dtype=int)))
            neigh = self.knn[u]
            cand_v = neigh[alive[neigh] & (~is_ch[neigh])]
            if cand_v.size == 0:
                break

            cand_v = self.rng.permutation(cand_v)
            improved = False
            for v in cand_v:
                v = int(v)
                new_set = cur_set.copy()
                new_set.remove(u)
                new_set.add(v)
                new_arr = np.fromiter(new_set, dtype=int)

                f_new = self.eval_fitness(new_arr)
                if f_new < cur_f:
                    cur_set = new_set
                    cur_f = f_new
                    is_ch[u] = False
                    is_ch[v] = True
                    improved = True
                    break

            if not improved:
                break

        return np.fromiter(cur_set, dtype=int)

    # ----------------------------
    # Elite pool + fixed set
    # ----------------------------
    def update_elite(
        self,
        elite: List[Tuple[int, ...]],
        elite_sets: List[Set[int]],
        elite_f: List[float],
        ch: np.ndarray,
        f: float,
    ) -> None:
        """Update elite pool with optional diversity control."""
        key = _sorted_key(ch)
        if not key:
            return

        if key in elite:
            return

        s = set(key)

        delta = float(self.p.delta)
        if delta > 0.0 and elite_sets:
            best_f = float(np.min(elite_f)) if elite_f else float("inf")
            improve_margin = 1e-4
            if not (f < best_f - improve_margin):
                for es in elite_sets:
                    if _jaccard_distance(s, es) < delta:
                        return

        if len(elite) < int(self.p.elite_size):
            elite.append(key)
            elite_sets.append(s)
            elite_f.append(float(f))
            return

        worst_idx = int(np.argmax(elite_f))
        if float(f) < float(elite_f[worst_idx]):
            elite[worst_idx] = key
            elite_sets[worst_idx] = s
            elite_f[worst_idx] = float(f)

    def build_fixed_set(self, elite: List[Tuple[int, ...]]) -> np.ndarray:
        """Compute fixed set F_t from elite pool using frequency + energy safety."""
        alive = self.net.alive_mask
        if not elite:
            return np.array([], dtype=int)

        counts = np.zeros(self.net.n_nodes, dtype=float)
        for key in elite:
            if key:
                counts[np.fromiter(key, dtype=int)] += 1.0
        freq = counts / float(len(elite))

        energy_ratio = self.net.residual_energy / (self.net.initial_energy + 1e-12)
        fixed_mask = (freq >= float(self.p.tau)) & (energy_ratio >= float(self.p.theta)) & alive
        fixed = np.where(fixed_mask)[0]
        return fixed.astype(int)

    # ----------------------------
    # Main procedure
    # ----------------------------
    def run(self) -> Tuple[np.ndarray, float]:
        """Run FSS-WSN once for the current round and return best CH set + fitness."""
        alive = self.net.alive_mask
        n_alive = int(np.sum(alive))
        if n_alive <= 0:
            return np.array([], dtype=int), 1.0

        # derive budgets
        if int(self.p.max_iter1) > 0:
            max1 = int(self.p.max_iter1)
        else:
            max1 = int(max(1, round(0.60 * int(self.p.n_iter))))

        if int(self.p.max_iter2) > 0:
            max2 = int(self.p.max_iter2)
        else:
            max2 = int(max(0, int(self.p.n_iter) - max1))

        if not bool(self.p.use_phase2):
            max2 = 0

        kmax = int(self.p.Kmax) if int(self.p.Kmax) > 0 else _default_kmax(n_alive)

        elite: List[Tuple[int, ...]] = []
        elite_sets: List[Set[int]] = []
        elite_f: List[float] = []

        # Phase I
        for _ in range(max1):
            ch = self.construct(start_set=None, kmax=kmax)
            ch = self.local_search(ch)
            f = self.eval_fitness(ch)
            self.update_elite(elite, elite_sets, elite_f, ch, f)

        # Phase II
        if max2 > 0 and elite:
            fixed = self.build_fixed_set(elite)
            for _ in range(max2):
                start = fixed if fixed.size > 0 else None
                ch = self.construct(start_set=start, kmax=kmax)
                ch = self.local_search(ch)
                f = self.eval_fitness(ch)
                self.update_elite(elite, elite_sets, elite_f, ch, f)

        if not elite:
            alive_idx = np.where(alive)[0]
            best = int(alive_idx[int(np.argmax(self.net.residual_energy[alive_idx]))])
            best_arr = np.array([best], dtype=int)
            return best_arr, self.eval_fitness(best_arr)

        best_idx = int(np.argmin(elite_f))
        best_key = elite[best_idx]
        best_ch = np.fromiter(best_key, dtype=int)
        return best_ch, float(elite_f[best_idx])


def run_fss_wsn(net: Network, fit_params: FitnessParams, fss_params: FSSParams) -> OptimizationResult:
    """Entry point used by runner/app (one round). Returns OptimizationResult with NFE."""
    rng = np.random.default_rng(int(fss_params.seed))
    algo = FSSWSN(net=net, fit_params=fit_params, params=fss_params, rng=rng)

    best_ch, best_f = algo.run()

    # Put extra info in history as well (useful even if caller ignores 'nfe')
    history = {
        "nfe_requests": int(algo.nfe_requests),
        "cache_hits": int(algo.cache_hits),
        "cache_misses": int(algo.cache_misses),
        "cache_size": int(len(algo._fit_cache)),
    }

    return OptimizationResult(
        best_ch_indices=np.asarray(best_ch, dtype=int),
        best_fitness=float(best_f),
        history=history,
        nfe=float(algo.nfe),
    )
