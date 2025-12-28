# wsn/repair.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .models import Network


@dataclass(frozen=True)
class RepairParams:
    """Parameters for deterministic repair mapping Repair_t(·).

    The paper specifies that when the candidate CH set is empty, the initial CH
    is chosen using the same greedy score used in the constructive phase
    (energy + sink-centrality + density), with ties broken by smallest index.
    """
    alpha1: float = 0.5  # energy weight
    alpha2: float = 0.3  # sink-centrality weight
    alpha3: float = 0.2  # density weight


def sanitize_ch_set(net: Network, ch: np.ndarray) -> np.ndarray:
    """Sanitize a CH set to unique alive indices within [0, N)."""
    n = int(net.n_nodes)
    if n <= 0:
        return np.array([], dtype=int)

    ch = np.asarray(ch, dtype=int).reshape(-1)
    if ch.size == 0:
        return np.array([], dtype=int)

    ch = ch[(ch >= 0) & (ch < n)]
    if ch.size == 0:
        return np.array([], dtype=int)

    alive = net.alive_mask
    ch = np.unique(ch[alive[ch]])
    return ch.astype(int)


def _normalize_alphas(p: RepairParams) -> Tuple[float, float, float]:
    a1, a2, a3 = float(p.alpha1), float(p.alpha2), float(p.alpha3)
    s = a1 + a2 + a3
    if s <= 0:
        return 1.0, 0.0, 0.0
    return a1 / s, a2 / s, a3 / s


def _greedy_score(net: Network, rc: float, p: RepairParams) -> np.ndarray:
    """Compute the constructive greedy score h_t(i) used by Repair when H0 is empty."""
    a1, a2, a3 = _normalize_alphas(p)

    cache = net.get_grasp_cache(rc=float(rc), k_nn=1)
    rho = cache["rho"].astype(float)
    sink_term = cache["sink_term"].astype(float)

    energy_ratio = net.residual_energy / (net.initial_energy + 1e-12)
    energy_ratio = np.clip(energy_ratio, 0.0, 1.0).astype(float)

    return a1 * energy_ratio + a2 * sink_term + a3 * rho


def repair_ch_set_to_cover_all_alive(
    net: Network,
    ch0: np.ndarray,
    rc: float,
    params: RepairParams = RepairParams(),
) -> np.ndarray:
    """Deterministic Repair_t(H0) enforcing strict Rc-coverage (hard constraint).

    Steps (paper Section 3.5):
      1) If H0 is empty: pick one alive node with highest greedy score (ties -> smallest index).
      2) While some alive nodes are uncovered: add node covering maximum uncovered nodes (ties -> smallest index).
      3) Return H+ (sorted unique indices).

    Note: assignment to nearest CH is performed elsewhere (Network.assign_clusters).
    """
    n = int(net.n_nodes)
    alive = net.alive_mask
    if n <= 0 or not bool(np.any(alive)):
        return np.array([], dtype=int)

    ch = sanitize_ch_set(net, ch0)

    # 1) Non-empty sanitization using greedy score (ties -> smallest index)
    if ch.size == 0:
        h = _greedy_score(net, rc=float(rc), p=params)
        alive_idx = np.where(alive)[0]
        best_val = float(np.max(h[alive_idx]))
        top = alive_idx[h[alive_idx] == best_val]
        u0 = int(np.min(top))  # deterministic tie-break
        ch = np.array([u0], dtype=int)

    cache = net.get_grasp_cache(rc=float(rc), k_nn=1)
    within_rc = cache["within_rc"]  # (N,N) bool, includes self-coverage

    is_ch = np.zeros(n, dtype=bool)
    is_ch[ch] = True

    # uncovered alive nodes
    covered = np.any(within_rc[ch], axis=0)
    uncovered = alive & (~covered)

    # incremental coverage counts against current uncovered set
    cover_counts = within_rc[:, uncovered].sum(axis=1).astype(np.int32)
    cover_counts[~alive] = 0
    cover_counts[is_ch] = 0

    # 2) Greedy additions until coverage
    while bool(np.any(uncovered)):
        cand = np.where((alive & ~is_ch) & (cover_counts > 0))[0]
        if cand.size == 0:
            # Safety: since within_rc[i,i] is True, this should not happen.
            u = int(np.min(np.where(uncovered)[0]))
        else:
            best_cov = int(np.max(cover_counts[cand]))
            top = cand[cover_counts[cand] == best_cov]
            u = int(np.min(top))  # deterministic tie-break

        is_ch[u] = True
        ch = np.append(ch, u)

        newly_covered = within_rc[u] & uncovered
        if bool(np.any(newly_covered)):
            uncovered[newly_covered] = False
            cover_counts -= within_rc[:, newly_covered].sum(axis=1).astype(np.int32)

        cover_counts[u] = 0

    return np.unique(ch).astype(int)


def _bfs_path_to_targets(
    start: int,
    targets: np.ndarray,
    neighbors: list[np.ndarray],
    alive: np.ndarray,
) -> np.ndarray:
    """Return shortest hop-count path from start to any target (inclusive).

    Deterministic tie-break: neighbors are assumed sorted ascending.
    """
    n = int(alive.size)
    tgt = np.zeros(n, dtype=bool)
    tgt[targets] = True

    if tgt[start]:
        return np.array([int(start)], dtype=int)

    prev = np.full(n, -1, dtype=int)
    vis = np.zeros(n, dtype=bool)

    q = [int(start)]
    vis[start] = True

    found = -1
    while q:
        u = q.pop(0)
        for v in neighbors[u]:
            if not alive[v] or vis[v]:
                continue
            vis[v] = True
            prev[v] = u
            if tgt[v]:
                found = v
                q.clear()
                break
            q.append(int(v))

    if found < 0:
        return np.array([], dtype=int)

    # Reconstruct path: found -> start
    path = [int(found)]
    cur = int(found)
    while cur != int(start):
        cur = int(prev[cur])
        if cur < 0:
            return np.array([], dtype=int)
        path.append(cur)
    path.reverse()
    return np.asarray(path, dtype=int)


def repair_ch_set_to_cover_and_connect_to_sink(
    net: Network,
    ch0: np.ndarray,
    rc: float,
    r_tx: float,
    params: RepairParams = RepairParams(),
) -> np.ndarray:
    """Repair enforcing (1) Rc coverage and (2) CH->sink multi-hop connectivity.

    Stage 1 (paper-aligned):
      - Apply :func:`repair_ch_set_to_cover_all_alive` to obtain a strictly Rc-covering CH set.

    Stage 2 (multi-hop extension):
      - Ensure every CH in H+ has a path to the sink using hops of length <= r_tx.
      - Relays are chosen by promoting intermediate alive nodes on shortest hop-count paths.

    Notes
    -----
    - This is deterministic given fixed geometry/energies.
    - If no alive node can connect to the sink directly within r_tx, stage 2 is impossible
      and the function returns the stage-1 CH set.
    """
    ch = repair_ch_set_to_cover_all_alive(net, ch0=ch0, rc=float(rc), params=params)

    n = int(net.n_nodes)
    alive = net.alive_mask
    if n <= 0 or ch.size == 0 or not bool(np.any(alive)):
        return ch

    r_tx_f = float(r_tx)
    if r_tx_f <= 0:
        return ch

    # Any node that can reach the sink in one hop under r_tx
    sink_direct = alive & (net.dists_to_sink <= r_tx_f)
    if not bool(np.any(sink_direct)):
        # Cannot ever connect to sink under this r_tx
        return ch

    # Build adjacency under r_tx over ALL nodes (we may promote relays)
    cache_tx = net.get_grasp_cache(rc=r_tx_f, k_nn=1)
    within_tx = cache_tx["within_rc"]

    # Precompute neighbors (sorted) for deterministic BFS
    neighbors: list[np.ndarray] = []
    for i in range(n):
        nbr = np.where(within_tx[i] & alive)[0]
        nbr = nbr[nbr != i]
        neighbors.append(np.sort(nbr).astype(int))

    is_ch = np.zeros(n, dtype=bool)
    is_ch[ch] = True

    # Ensure at least one CH has direct sink connectivity (otherwise promote a root)
    if not bool(np.any(is_ch & sink_direct)):
        cand = np.where(sink_direct)[0]
        # choose highest residual energy, tie -> smallest index
        e = net.residual_energy[cand]
        best_e = float(np.max(e))
        top = cand[e == best_e]
        root = int(np.min(top))
        is_ch[root] = True
        ch = np.append(ch, root)

    # Helper: compute which CHs are connected to sink via CH-only hops
    def _connected_ch_mask() -> np.ndarray:
        connected = np.zeros(n, dtype=bool)
        q = list(np.where(is_ch & sink_direct)[0])
        for u in q:
            connected[u] = True
        # BFS on CH-only graph
        while q:
            u = q.pop(0)
            # CH neighbors within r_tx
            for v in neighbors[u]:
                if not is_ch[v] or connected[v]:
                    continue
                connected[v] = True
                q.append(int(v))
        return connected

    # Iteratively connect unreachable CHs by promoting relay paths
    max_iter = int(n)  # safety
    for _ in range(max_iter):
        connected = _connected_ch_mask()
        unreachable_ch = np.where(is_ch & alive & (~connected))[0]
        if unreachable_ch.size == 0:
            break

        start = int(np.min(unreachable_ch))
        targets = np.where((is_ch & connected) | sink_direct)[0]
        path = _bfs_path_to_targets(start=start, targets=targets, neighbors=neighbors, alive=alive)
        if path.size == 0:
            # No path exists under r_tx to any sink-connected region
            break

        # Promote every intermediate node on the path (excluding start) to CH
        for v in path[1:]:
            if not is_ch[int(v)]:
                is_ch[int(v)] = True
                ch = np.append(ch, int(v))

    return np.unique(ch).astype(int)
