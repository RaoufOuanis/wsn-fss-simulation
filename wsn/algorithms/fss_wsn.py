# wsn/algorithms/fss_wsn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..fitness import FitnessParams, fitness
from ..models import Network
from ..multihop import dijkstra_costs_and_next_hops
from ..repair import repair_ch_set_to_cover_all_alive, repair_ch_set_to_cover_and_connect_to_sink, sanitize_ch_set
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

    # Coverage-gain weighting during GRASP construction
    #
    # Background
    # - A pure h(i) greedy score can over-fragment in corner-sink settings (too many CHs),
    #   hurting aggregation and multi-hop throughput.
    # - Multiplying h(i) by a (coverage_gain) factor fixes corner but may be too aggressive
    #   when the sink is near the center.
    #
    # Hybrid policy (recommended)
    # - If coverage_gain_auto=True, we compute a sink "eccentricity" in [0,1] based on how
    #   far the sink is from the area center, and interpolate between:
    #     w=0  -> legacy behavior (no coverage-gain weighting)
    #     w=1  -> corner-robust behavior (full coverage-gain weighting)
    coverage_gain_auto: bool = True
    coverage_gain_weight: float = 1.0

    # Local search parameters
    Lmax: int = 0  # Changed from 10→0: empirical validation shows ~10× CPU reduction with no lifetime loss (≤2000 rounds)
    k_nn: int = 10

    # Safety cap on the CH set size during construction
    Kmax: int = 0

    # Optional elite diversity control (Jaccard distance threshold); 0 disables
    delta: float = 0.0

    # Phase control
    use_phase2: bool = True

    # Phase II safety: avoid selecting a phase-II best (by fitness) that is likely
    # to reduce lifetime due to higher per-round energy consumption.
    phase2_energy_guard: bool = True
    phase2_energy_guard_eps: float = 0.0  # default: do not allow higher estimated energy cost


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

        # Phase-II guard diagnostics (filled in run())
        self.phase2_guard_triggered: bool = False
        self.phase2_best_energy: float = float("nan")
        self.phase1_best_energy: float = float("nan")
        self.best_source: str = "unknown"  # "phase1" or "phase2"

        # Best-of-phase diagnostics (filled in run())
        self.phase1_best_key: Tuple[int, ...] | None = None
        self.phase2_best_key: Tuple[int, ...] | None = None
        self.phase1_best_f: float = float("nan")
        self.phase2_best_f: float = float("nan")
        self.phase1_components: Dict[str, float] | None = None
        self.phase2_components: Dict[str, float] | None = None

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

    def estimate_round_energy_cost(self, ch: np.ndarray) -> float:
        """Estimate per-round energy cost for a CH set (proxy for lifetime).

        This is a deterministic, geometry-based estimate aligned with the radio model.
        It does NOT mutate the network energies.

        Rationale
        ---------
        Phase II can discover solutions with slightly better surrogate fitness but
        worse real energy dynamics (thus lower R_last/LND). This proxy is used as
        a guardrail to prevent such regressions.
        """
        alive = self.net.alive_mask
        n_alive = int(np.sum(alive))
        if n_alive <= 0:
            return 0.0

        # Candidate set H0 (sanitized to alive unique indices)
        H0 = sanitize_ch_set(self.net, np.asarray(ch, dtype=int).reshape(-1))

        # Deterministic repair (must match simulator/fitness)
        if bool(self.fit_params.multihop):
            H_plus = repair_ch_set_to_cover_and_connect_to_sink(
                self.net,
                ch0=H0,
                rc=float(self.fit_params.rc),
                r_tx=float(self.fit_params.r_tx),
                params=self.fit_params.repair,
            )
        else:
            H_plus = repair_ch_set_to_cover_all_alive(
                self.net,
                ch0=H0,
                rc=float(self.fit_params.rc),
                params=self.fit_params.repair,
            )

        if H_plus.size == 0:
            return float("inf")

        # Assignment to nearest CH in H_plus
        assignments, _dist_to_ch, _in_radius = self.net.assign_clusters(H_plus, rc=float(self.fit_params.rc))

        # Sanitize CH set
        ch_idx = np.unique(np.asarray(H_plus, dtype=int).reshape(-1))
        ch_idx = ch_idx[alive[ch_idx]]
        if ch_idx.size == 0:
            return float("inf")

        is_ch = np.zeros(self.net.n_nodes, dtype=bool)
        is_ch[ch_idx] = True

        # Members: alive non-CH (only those assigned to alive CH)
        members_all = np.where(alive & (~is_ch))[0]
        if members_all.size > 0:
            m_ch = np.asarray(assignments[members_all], dtype=int)
            in_range = (m_ch >= 0) & (m_ch < self.net.n_nodes)
            m_ch_safe = np.where(in_range, m_ch, 0)
            valid = in_range & is_ch[m_ch_safe] & alive[m_ch_safe]

            members = members_all[valid]
            ch_of_member = m_ch_safe[valid]
        else:
            members = np.array([], dtype=int)
            ch_of_member = np.array([], dtype=int)

        radio = self.fit_params.radio

        # Member -> CH distances
        e_member_rx = 0.0
        e_member_tx = 0.0
        e_ch_tx_ctrl = 0.0
        e_ch_rx = 0.0
        e_ch_da = 0.0

        member_counts = np.zeros(self.net.n_nodes, dtype=float)
        if members.size > 0:
            member_counts = np.bincount(ch_of_member, minlength=self.net.n_nodes).astype(float)
            d_m2ch = self.net.dists[members, ch_of_member]

            # Control (CH->members) + member RX
            e_tx_ctrl = radio.tx_energy_vec(radio.l_ctrl, d_m2ch)
            e_ch_tx_ctrl = float(np.sum(e_tx_ctrl))
            e_member_rx = float(members.size) * float(radio.rx_energy(radio.l_ctrl))

            # Join
            e_member_tx += float(np.sum(radio.tx_energy_vec(radio.l_ctrl, d_m2ch)))
            e_ch_rx += float(np.sum(member_counts) * float(radio.rx_energy(radio.l_ctrl)))

            # Data
            e_member_tx += float(np.sum(radio.tx_energy_vec(radio.l_data, d_m2ch)))
            e_ch_rx += float(np.sum(member_counts) * float(radio.rx_energy(radio.l_data)))
            e_ch_da += float(np.sum(member_counts) * float(radio.E_da * float(radio.l_data)))

        # CH -> sink energy (single-hop approx or Dijkstra mh cost)
        if bool(self.fit_params.multihop):
            kappa, _next_hop = dijkstra_costs_and_next_hops(
                net=self.net,
                ch_indices=ch_idx,
                radio=radio,
                r_tx=float(self.fit_params.r_tx),
            )
            kappa = np.asarray(kappa, dtype=float)
            # Defensive: saturate unreachable at large cost
            kappa = np.where(np.isfinite(kappa), kappa, 0.0)
            e_ch_to_sink = float(np.sum(kappa))
        else:
            d_sink = self.net.dists_to_sink[ch_idx]
            e_ch_to_sink = float(np.sum(radio.tx_energy_vec(radio.l_data, d_sink)))

        return float(e_member_rx + e_member_tx + e_ch_tx_ctrl + e_ch_rx + e_ch_da + e_ch_to_sink)

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

        # Hybrid switch weight in [0,1]
        w_cov = float(np.clip(getattr(self.p, "coverage_gain_weight", 1.0), 0.0, 1.0))
        if bool(getattr(self.p, "coverage_gain_auto", True)):
            # Eccentricity: 0 at center, 1 at farthest corner.
            cx = 0.5 * float(self.net.area_size)
            cy = 0.5 * float(self.net.area_size)
            sx = float(self.net.sink[0])
            sy = float(self.net.sink[1])
            d = float(np.hypot(sx - cx, sy - cy))
            dmax = float(np.hypot(cx, cy))  # distance from center to corner
            ecc = d / max(1e-12, dmax)
            # Multi-hop stress proxy: when d_max_to_sink <= r_tx, CH->sink is mostly 1-hop,
            # so over-fragmentation is less harmful. As the required hop count grows,
            # we progressively enable the anti-fragmentation weighting.
            r_tx_f = float(max(1e-12, float(self.fit_params.r_tx)))
            d_max_sink = float(np.max(self.net.dists_to_sink[alive]))
            hop_ratio = d_max_sink / r_tx_f  # ~1 => mostly 1-hop, larger => multi-hop regime
            tx_stress = float(np.clip((hop_ratio - 1.0) / 2.0, 0.0, 1.0))

            w_cov = float(np.clip(ecc * tx_stress, 0.0, 1.0))

        # Main loop: add CHs until coverage is achieved or Kmax reached
        while bool(np.any(uncovered)) and len(H) < Kmax:
            cand_mask = alive & (~is_ch) & (cover_counts > 0)
            cand_idx = np.where(cand_mask)[0]
            if cand_idx.size == 0:
                break  # should be rare because each uncovered node can cover itself

            # Coverage-gain weighting: prefer candidates that cover more currently-uncovered nodes.
            # Hybrid: interpolate between legacy (w_cov=0) and full weighting (w_cov=1).
            cov = cover_counts[cand_idx].astype(float)
            cov_max = float(np.max(cov)) if cov.size > 0 else 1.0
            cov_norm = cov / max(1.0, cov_max)  # in [0,1]
            cov_scale = (0.5 + 0.5 * cov_norm)  # scale in [0.5, 1.0]
            vals = h[cand_idx] * ((1.0 - w_cov) + w_cov * cov_scale)
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

        best1_key: Tuple[int, ...] | None = None
        best1_f: float = float("inf")

        best2_key: Tuple[int, ...] | None = None
        best2_f: float = float("inf")

        # Phase I
        for _ in range(max1):
            ch = self.construct(start_set=None, kmax=kmax)
            ch = self.local_search(ch)
            f = self.eval_fitness(ch)
            self.update_elite(elite, elite_sets, elite_f, ch, f)

            key = _sorted_key(ch)
            if key and float(f) < float(best1_f):
                best1_key = key
                best1_f = float(f)

        # Phase II
        if max2 > 0 and elite:
            fixed = self.build_fixed_set(elite)
            for _ in range(max2):
                start = fixed if fixed.size > 0 else None
                ch = self.construct(start_set=start, kmax=kmax)
                ch = self.local_search(ch)
                f = self.eval_fitness(ch)
                self.update_elite(elite, elite_sets, elite_f, ch, f)

                key = _sorted_key(ch)
                if key and float(f) < float(best2_f):
                    best2_key = key
                    best2_f = float(f)

        if not elite:
            alive_idx = np.where(alive)[0]
            best = int(alive_idx[int(np.argmax(self.net.residual_energy[alive_idx]))])
            best_arr = np.array([best], dtype=int)
            return best_arr, self.eval_fitness(best_arr)

        best_idx = int(np.argmin(elite_f))
        best_key = elite[best_idx]

        # Phase-II guardrail: if phase-II improved fitness but worsened estimated energy,
        # keep the best Phase-I solution.
        best_f = float(elite_f[best_idx])
        chosen_key = best_key
        # Default: if Phase II isn't used (or produced no candidate), treat as Phase I.
        self.best_source = "phase1"
        self.phase2_guard_triggered = False
        self.phase2_best_energy = float("nan")
        self.phase1_best_energy = float("nan")

        # If Phase II ran and produced at least one candidate, we can attribute the best.
        if bool(self.p.use_phase2) and best2_key is not None:
            # If chosen matches Phase I best, source is Phase I; otherwise Phase II.
            self.best_source = "phase1" if (best1_key is not None and chosen_key == best1_key) else "phase2"

        if bool(self.p.use_phase2) and bool(self.p.phase2_energy_guard) and best1_key is not None and best2_key is not None:
            # Only meaningful if elite best is strictly better than the best Phase-I.
            if float(best_f) < float(best1_f) - 1e-12:
                e_best = self.estimate_round_energy_cost(np.fromiter(best_key, dtype=int))
                e_best1 = self.estimate_round_energy_cost(np.fromiter(best1_key, dtype=int))
                self.phase2_best_energy = float(e_best)
                self.phase1_best_energy = float(e_best1)
                eps = float(max(0.0, float(self.p.phase2_energy_guard_eps)))
                if np.isfinite(e_best) and np.isfinite(e_best1) and float(e_best) > float(e_best1) * (1.0 + eps):
                    chosen_key = best1_key
                    best_f = float(best1_f)
                    self.phase2_guard_triggered = True
                    self.best_source = "phase1"

        # Store best-of-phase diagnostics (components are computed on demand at end)
        self.phase1_best_key = best1_key
        self.phase2_best_key = best2_key
        self.phase1_best_f = float(best1_f) if np.isfinite(best1_f) else float("nan")
        self.phase2_best_f = float(best2_f) if np.isfinite(best2_f) else float("nan")

        if best1_key is not None:
            _f1, c1 = fitness(self.net, np.fromiter(best1_key, dtype=int), self.fit_params)
            self.phase1_components = {k: float(v) for k, v in (c1 or {}).items()}
        else:
            self.phase1_components = None

        if best2_key is not None:
            _f2, c2 = fitness(self.net, np.fromiter(best2_key, dtype=int), self.fit_params)
            self.phase2_components = {k: float(v) for k, v in (c2 or {}).items()}
        else:
            self.phase2_components = None

        best_ch = np.fromiter(chosen_key, dtype=int)
        return best_ch, float(best_f)


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

        # Phase-II guardrail diagnostics
        "phase2_energy_guard": bool(getattr(fss_params, "phase2_energy_guard", False)),
        "phase2_energy_guard_eps": float(getattr(fss_params, "phase2_energy_guard_eps", float("nan"))),
        "phase2_guard_triggered": bool(algo.phase2_guard_triggered),
        "phase2_best_energy_est": float(algo.phase2_best_energy),
        "phase1_best_energy_est": float(algo.phase1_best_energy),
        "best_source": str(algo.best_source),

        # Best-of-phase (Phase I vs Phase II) diagnostics
        "phase1_best_f": float(algo.phase1_best_f),
        "phase2_best_f": float(algo.phase2_best_f),
    }

    # Flatten component breakdowns (if available)
    if algo.phase1_components is not None:
        for k, v in algo.phase1_components.items():
            history[f"phase1_{k}"] = float(v)
    if algo.phase2_components is not None:
        for k, v in algo.phase2_components.items():
            history[f"phase2_{k}"] = float(v)

    return OptimizationResult(
        best_ch_indices=np.asarray(best_ch, dtype=int),
        best_fitness=float(best_f),
        history=history,
        nfe=float(algo.nfe),
    )
