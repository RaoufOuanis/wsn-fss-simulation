from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Tuple
import heapq
import numpy as np

from .models import Network


class RadioModel(Protocol):
    """Protocol for the radio model used by multi-hop routines."""

    l_data: int

    def tx_energy(self, l_bits: int, d: float) -> float:  # pragma: no cover
        ...

    def rx_energy(self, l_bits: int) -> float:  # pragma: no cover
        ...


@dataclass
class MultiHopParams:
    """Parameters for multi-hop CH->sink routing."""

    r_tx: float = 50.0


def dijkstra_costs_and_next_hops(
    net: Network,
    ch_indices: np.ndarray,
    radio: RadioModel,
    r_tx: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-CH shortest path energy costs to sink and next hops.

    The graph contains only the CH nodes in ``ch_indices`` plus a sink super-node.

    Edges:
      - CH <-> CH if d(u,v) <= r_tx with per-packet energy weight:
            w(u,v) = E_TX(L, d(u,v)) + E_RX(L)
      - CH <-> sink if d(u,sink) <= r_tx with per-packet weight:
            w(u,sink) = E_TX(L, d(u,sink))

    Running Dijkstra from the sink yields:
      - kappa[i]  : minimum per-packet energy cost from CH i to sink
      - next_hop[i]: the next node on the shortest path (CH index in the global
                     numbering) or -1 for direct-to-sink.

    Ties are broken deterministically by choosing the smallest predecessor index.

    Returns
    -------
    kappa : ndarray (K,)
        Per-CH minimum per-packet energy cost to sink.
    next_hop : ndarray (K,)
        Next hop for each CH; value is another CH node index, or -1 for sink.
    """

    ch = np.asarray(ch_indices, dtype=int).reshape(-1)
    if ch.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=int)

    # Ensure deterministic ordering
    ch = np.unique(ch)
    k = int(ch.size)
    sink_local = k

    # Build adjacency lists (local indices 0..k-1 plus sink_local)
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(k + 1)]

    # CH-CH edges
    D = net.dists[np.ix_(ch, ch)]
    L = int(getattr(radio, "l_data"))
    rx = float(radio.rx_energy(L))

    r_tx_f = float(r_tx)
    if r_tx_f <= 0:
        # No connectivity possible
        kappa = np.full(k, np.inf, dtype=float)
        next_hop = np.full(k, -1, dtype=int)
        return kappa, next_hop

    for i in range(k):
        for j in range(i + 1, k):
            dij = float(D[i, j])
            if dij <= r_tx_f:
                w = float(radio.tx_energy(L, dij) + rx)
                adj[i].append((j, w))
                adj[j].append((i, w))

    # CH-sink edges
    d_sink = net.dists_to_sink[ch]
    for i in range(k):
        ds = float(d_sink[i])
        if ds <= r_tx_f:
            w = float(radio.tx_energy(L, ds))
            adj[i].append((sink_local, w))
            adj[sink_local].append((i, w))

    # Dijkstra from sink
    INF = float("inf")
    dist = np.full(k + 1, INF, dtype=float)
    prev = np.full(k + 1, -1, dtype=int)

    dist[sink_local] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, sink_local)]

    while pq:
        du, u = heapq.heappop(pq)
        if du != dist[u]:
            continue

        for v, w in adj[u]:
            nd = du + w
            # Deterministic tie-break: prefer smaller predecessor index
            if (nd < dist[v] - 1e-12) or (abs(nd - dist[v]) <= 1e-12 and (prev[v] < 0 or u < prev[v])):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    kappa = dist[:k].copy()

    # Next hop from CH i to sink is prev[i] (node closer to sink)
    next_hop = np.full(k, -1, dtype=int)
    for i in range(k):
        p = int(prev[i])
        if p < 0:
            next_hop[i] = -1
        elif p == sink_local:
            next_hop[i] = -1
        else:
            next_hop[i] = int(ch[p])

    return kappa, next_hop


def relay_packet_counts(
    ch_indices: np.ndarray,
    kappa: np.ndarray,
    next_hop: np.ndarray,
) -> np.ndarray:
    """Compute relay packet counts q_h for each CH in a shortest-path routing tree.

    Each CH generates one packet. Packets are forwarded hop-by-hop following ``next_hop``.

    q_h counts how many packets CH h transmits (its own + relayed). If routing is a
    directed forest to the sink, then:

        q_h = 1 + sum_{child routes to h} q_child.

    Parameters
    ----------
    ch_indices : ndarray (K,)
        Unique sorted CH indices.
    kappa : ndarray (K,)
        Shortest path costs (used only to define a far-to-near order).
    next_hop : ndarray (K,)
        Next hop CH index or -1 for sink.

    Returns
    -------
    q : ndarray (K,)
        Packets forwarded by each CH (includes own)."""

    ch = np.asarray(ch_indices, dtype=int).reshape(-1)
    if ch.size == 0:
        return np.zeros(0, dtype=float)

    ch = np.unique(ch)
    k = int(ch.size)
    # Map global CH index -> local position
    loc: Dict[int, int] = {int(ch[i]): i for i in range(k)}

    q = np.ones(k, dtype=float)

    # Process farthest to nearest (descending kappa); unreachable treated as farthest
    kappa_safe = np.asarray(kappa, dtype=float)
    kappa_safe = np.where(np.isfinite(kappa_safe), kappa_safe, np.max(np.where(np.isfinite(kappa_safe), kappa_safe, 0.0)) + 1.0)
    order = np.argsort(kappa_safe)[::-1]

    for i in order:
        nh = int(next_hop[i])
        if nh < 0:
            continue
        p = loc.get(nh)
        if p is None:
            continue
        q[p] += q[i]

    return q
