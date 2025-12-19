# wsn/models.py
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


class Network:
    """Static wireless sensor network with vectorized per-round state.

    This project evaluates centralized, single-hop clustering (CH -> BS).

    Geometry is static and cached once:
      - inter-node distance matrix D (NxN)
      - node-to-sink distances d_sink (N,)

    Energy/liveness are dynamic and represented as arrays:
      - initial_energy (N,)
      - residual_energy (N,)
      - alive_mask (N,)

    Notes
    -----
    - The network does not enforce protocol-specific CH rotation rules.
      Protocol baselines (LEACH/SEP/HEED) keep their own per-run state.
    - Once a node is dead, it remains dead for the rest of the simulation.
    """

    def __init__(
        self,
        positions: np.ndarray,
        sink_position: Tuple[float, float],
        area_size: float,
        initial_energy: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.positions = np.asarray(positions, dtype=float)
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError("positions must have shape (N, 2)")

        self.n_nodes = int(self.positions.shape[0])
        self.area_size = float(area_size)
        self.sink = np.asarray(sink_position, dtype=float)
        if self.sink.shape != (2,):
            raise ValueError("sink_position must be a tuple (x, y)")

        self.initial_energy = np.asarray(initial_energy, dtype=float).reshape(-1)
        if self.initial_energy.shape[0] != self.n_nodes:
            raise ValueError("initial_energy must have length N")

        self.residual_energy = self.initial_energy.copy()
        self.alive_mask = np.ones(self.n_nodes, dtype=bool)

        self.rng = rng if rng is not None else np.random.default_rng(0)

        # --- Precompute geometry (STATIC) ---
        self._dist_matrix = self._compute_distance_matrix(self.positions)
        self._dist_to_sink = np.linalg.norm(self.positions - self.sink, axis=1)
        self._max_dist_to_sink = float(np.max(self._dist_to_sink)) if self.n_nodes else 0.0
        self._diag = float(np.sqrt(2.0) * self.area_size)

        # Cached structures used by GRASP/FSS (keyed by (rc, k_nn)).
        # The network geometry is static, so these can be reused across rounds.
        self._grasp_cache: dict[tuple[float, int], dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def random_network(
        n_nodes: int,
        area_size: float = 100.0,
        sink_pos: Optional[Tuple[float, float]] = None,
        e0: float = 0.1,
        heterogenous: bool = False,
        adv_fraction: float = 0.2,
        e_adv: float = 0.2,
        seed: int = 0,
    ) -> "Network":
        """Generate a random deployment with optional 2-level heterogeneity."""
        rng = np.random.default_rng(seed)
        xs = rng.uniform(0.0, float(area_size), size=int(n_nodes))
        ys = rng.uniform(0.0, float(area_size), size=int(n_nodes))
        positions = np.stack([xs, ys], axis=1)

        if sink_pos is None:
            sink_pos = (float(area_size) / 2.0, float(area_size) / 2.0)

        if heterogenous:
            is_adv = rng.random(size=int(n_nodes)) < float(adv_fraction)
            e_init = np.where(is_adv, float(e_adv), float(e0))
        else:
            e_init = np.full(int(n_nodes), float(e0), dtype=float)

        return Network(
            positions=positions,
            sink_position=sink_pos,
            area_size=float(area_size),
            initial_energy=e_init,
            rng=rng,
        )

    @staticmethod
    def _compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
        """Full NxN Euclidean distance matrix."""
        diff = coords[:, None, :] - coords[None, :, :]
        return np.linalg.norm(diff, axis=2)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def dists(self) -> np.ndarray:
        return self._dist_matrix

    @property
    def dists_to_sink(self) -> np.ndarray:
        return self._dist_to_sink

    @property
    def max_dists_to_sink(self) -> float:
        return self._max_dist_to_sink

    @property
    def diag(self) -> float:
        return self._diag

    def get_positions_array(self) -> np.ndarray:
        return self.positions

    def sink_position(self) -> Tuple[float, float]:
        return float(self.sink[0]), float(self.sink[1])

    def get_alive_mask(self) -> np.ndarray:
        return self.alive_mask.copy()

    def get_alive_indices(self) -> np.ndarray:
        return np.where(self.alive_mask)[0]

    def reset_energies(self) -> None:
        self.residual_energy = self.initial_energy.copy()
        self.alive_mask[:] = True

    # ------------------------------------------------------------------
    # Vectorized clustering
    # ------------------------------------------------------------------
    def assign_clusters(self, ch_indices, rc: float):
        """Assign each node to its nearest CH (vectorized).

        Parameters
        ----------
        ch_indices : array-like
            Indices of CH nodes.
        rc : float
            Cluster radius.

        Returns
        -------
        assignments : ndarray (N,)
            assignments[i] = index of CH serving node i
        dist_to_ch : ndarray (N,)
            distance from node i to its assigned CH
        in_radius : ndarray (N,) of bool
            True iff dist_to_ch[i] <= rc
        """
        ch_indices = np.asarray(ch_indices, dtype=int).reshape(-1)
        if ch_indices.size == 0:
            raise ValueError("Empty CH set is not allowed")

        # Distances from all nodes to all CHs: shape (N, |H|)
        D = self._dist_matrix[:, ch_indices]
        nearest = np.argmin(D, axis=1)
        assignments = ch_indices[nearest]
        dist_to_ch = D[np.arange(self.n_nodes), nearest]
        in_radius = dist_to_ch <= float(rc)
        return assignments, dist_to_ch, in_radius

    # ------------------------------------------------------------------
    # GRASP/FSS helpers (cached)
    # ------------------------------------------------------------------
    def get_grasp_cache(self, rc: float, k_nn: int = 10) -> dict[str, np.ndarray]:
        """Return cached arrays used by GRASP/FSS for a given Rc and kNN.

        The cache includes:
        - within_rc: (N,N) bool adjacency where within_rc[i,j] = (d(i,j) <= Rc)
        - rho: (N,) local coverage density rho(i) = |{j : d(i,j)<=Rc}| / N
        - sink_term: (N,) centrality term 1 - d(i,BS)/D_S in [0,1]
        - knn: (N,k_nn) int indices of the k_nn nearest neighbors of each node

        Notes
        -----
        - All arrays are geometry-only and can be reused across rounds.
        - The returned arrays are safe to treat as read-only.
        """
        rc_f = float(rc)
        k = int(max(1, k_nn))
        key = (rc_f, k)
        cached = self._grasp_cache.get(key)
        if cached is not None:
            return cached

        within_rc = self._dist_matrix <= rc_f
        rho = np.mean(within_rc, axis=1).astype(float)

        d_max = self._max_dist_to_sink if self._max_dist_to_sink > 0 else float(np.max(self._dist_to_sink))
        d_max = d_max if d_max > 0 else 1.0
        sink_term = 1.0 - (self._dist_to_sink / (d_max + 1e-12))
        sink_term = np.clip(sink_term, 0.0, 1.0).astype(float)

        # kNN indices (exclude self at column 0)
        # argpartition is O(N) per row and faster than full argsort.
        kk = min(self.n_nodes, k + 1)
        knn = np.argpartition(self._dist_matrix, kth=kk - 1, axis=1)[:, :kk]
        # remove self if present and keep exactly k neighbors
        out = np.empty((self.n_nodes, k), dtype=int)
        for i in range(self.n_nodes):
            row = knn[i]
            row = row[row != i]
            if row.size < k:
                # pad (rare for very small N)
                pad = np.full(k - row.size, i, dtype=int)
                row = np.concatenate([row, pad])
            out[i] = row[:k]

        cached = {"within_rc": within_rc, "rho": rho, "sink_term": sink_term, "knn": out}
        self._grasp_cache[key] = cached
        return cached

