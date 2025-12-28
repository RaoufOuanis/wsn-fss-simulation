from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np

from .models import Network
from .multihop import MultiHopParams, dijkstra_costs_and_next_hops


@dataclass
class RadioParams:
    """LEACH-style first order radio model parameters."""

    E_elec: float = 50e-9       # J/bit
    eps_fs: float = 10e-12      # J/bit/m^2
    eps_mp: float = 0.0013e-12  # J/bit/m^4
    E_da: float = 5e-9          # J/bit (data aggregation)
    l_data: int = 4000          # bits
    l_ctrl: int = 200           # bits
    d0: float = 87.0            # threshold distance (m)

    def tx_energy(self, l_bits: int, d: float) -> float:
        d = float(d)
        l_bits = int(l_bits)
        if d < self.d0:
            return l_bits * self.E_elec + l_bits * self.eps_fs * (d ** 2)
        return l_bits * self.E_elec + l_bits * self.eps_mp * (d ** 4)

    def tx_energy_vec(self, l_bits: int, d: np.ndarray) -> np.ndarray:
        """Vectorized transmit energy."""
        d = np.asarray(d, dtype=float)
        l_bits_f = float(int(l_bits))
        base = l_bits_f * self.E_elec
        e = np.empty_like(d, dtype=float)
        mask = d < float(self.d0)
        e[mask] = base + l_bits_f * self.eps_fs * (d[mask] ** 2)
        e[~mask] = base + l_bits_f * self.eps_mp * (d[~mask] ** 4)
        return e

    def rx_energy(self, l_bits: int) -> float:
        return float(int(l_bits)) * self.E_elec


def apply_round_energy(
    net: Network,
    ch_indices: np.ndarray,
    assignments: np.ndarray,
    radio: RadioParams,
    multihop: MultiHopParams | None = None,
) -> Dict[str, float]:
    """
    Option A (strict clustering, single-hop):
      - member -> CH (ctrl + join + data) ONLY if assigned to an alive CH
      - NO direct-to-sink fallback for members (repair in runner enforces feasibility)
      - CH receives join/data and pays aggregation cost per received data packet
      - CH -> sink attempted by all alive CHs; energy is debited for attempts
      - Throughput counts ONLY successful CH->sink TX (pre_energy >= required_energy)
      - delivered_reports counts reports delivered via successful CH TX:
            sum_{successful CH}(members_of_CH + 1 self-report)
    """

    n = int(net.n_nodes)
    if n <= 0:
        return {"alive": 0, "total_energy": 0.0, "pkts_to_sink": 0, "n_ch": 0, "delivered_reports": 0}

    alive0 = net.alive_mask.copy()
    if not np.any(alive0):
        return {"alive": 0, "total_energy": 0.0, "pkts_to_sink": 0, "n_ch": 0, "delivered_reports": 0}

    # --- sanitize CH set ---
    ch = np.asarray(ch_indices, dtype=int).reshape(-1)
    if ch.size == 0:
        alive_idx = np.where(alive0)[0]
        ch = np.array([int(alive_idx[int(np.argmax(net.residual_energy[alive_idx]))])], dtype=int)

    ch = ch[(ch >= 0) & (ch < n)]
    ch = np.unique(ch[alive0[ch]])
    if ch.size == 0:
        alive_idx = np.where(alive0)[0]
        best = int(alive_idx[int(np.argmax(net.residual_energy[alive_idx]))])
        ch = np.array([best], dtype=int)

    is_ch = np.zeros(n, dtype=bool)
    is_ch[ch] = True

    assigned = np.asarray(assignments, dtype=int).reshape(-1)
    if assigned.shape[0] != n:
        raise ValueError("assignments must have shape (N,)")

    # Members: alive non-CH
    members_all = np.where(alive0 & ~is_ch)[0]

    # Valid members: assigned to an alive CH (with repair, this should be all members_all)
    if members_all.size > 0:
        m_ch = assigned[members_all]
        in_range = (m_ch >= 0) & (m_ch < n)
        m_ch_safe = np.where(in_range, m_ch, 0)
        valid = in_range & is_ch[m_ch_safe] & alive0[m_ch_safe]

        members = members_all[valid]
        ch_of_member = m_ch_safe[valid]
    else:
        members = np.array([], dtype=int)
        ch_of_member = np.array([], dtype=int)

    member_counts = np.zeros(n, dtype=float)

    # -------------------------
    # 1) Member <-> CH phases
    # -------------------------
    if members.size > 0:
        member_counts = np.bincount(ch_of_member, minlength=n).astype(float)
        d_m2ch = net.dists[members, ch_of_member]

        # 1.a) CH control to members (approx per-member unicast)
        e_tx_ctrl_per_member = radio.tx_energy_vec(radio.l_ctrl, d_m2ch)
        e_rx_ctrl = radio.rx_energy(radio.l_ctrl)

        net.residual_energy[members] -= e_rx_ctrl

        ch_tx_ctrl = np.bincount(ch_of_member, weights=e_tx_ctrl_per_member, minlength=n)
        net.residual_energy -= ch_tx_ctrl

        # 1.b) Join
        e_tx_join = radio.tx_energy_vec(radio.l_ctrl, d_m2ch)
        net.residual_energy[members] -= e_tx_join
        net.residual_energy -= member_counts * radio.rx_energy(radio.l_ctrl)

        # 2) Data
        e_tx_data = radio.tx_energy_vec(radio.l_data, d_m2ch)
        net.residual_energy[members] -= e_tx_data
        net.residual_energy -= member_counts * radio.rx_energy(radio.l_data)
        net.residual_energy -= member_counts * (radio.E_da * float(radio.l_data))

    # -------------------------
    # 3) CH -> sink (single-hop or multi-hop)
    # -------------------------
    pkts_to_sink = 0
    delivered_reports = 0
    # Multi-hop diagnostics (kept NaN when not applicable)
    pkt_hops = 0
    mh_avg_path_hops = float("nan")
    mh_q_max = float("nan")
    mh_jain_q = float("nan")

    if multihop is None:
        # Paper-aligned single-hop CH->sink
        ch_alive0 = alive0[ch]
        d_sink = net.dists_to_sink[ch]
        e_tx_sink = radio.tx_energy_vec(radio.l_data, d_sink)

        e_pre_tx = net.residual_energy[ch]
        can_tx = ch_alive0 & (e_pre_tx >= e_tx_sink)
        pkts_to_sink = int(np.sum(can_tx))

        # Debit attempt for all alive CHs (prevents "immortal CH" freezing LND)
        net.residual_energy[ch[ch_alive0]] -= e_tx_sink[ch_alive0]

        delivered_reports = int(np.sum(member_counts[ch[can_tx]]) + pkts_to_sink)
        # Single-hop: each delivered CH packet corresponds to exactly one packet-hop.
        pkt_hops = int(pkts_to_sink)

    else:
        # Multi-hop extension: members->CH unchanged, CH->sink routed with Dijkstra.
        r_tx = float(multihop.r_tx)
        if r_tx > 0:
            # Active CHs after member phases
            ch_active_mask = alive0[ch] & (net.residual_energy[ch] > 0.0)
            ch_active = ch[ch_active_mask]

            if ch_active.size > 0:
                kappa, next_hop = dijkstra_costs_and_next_hops(
                    net=net,
                    ch_indices=ch_active,
                    radio=radio,
                    r_tx=r_tx,
                )

                # Build local mapping
                ch_active = np.unique(ch_active)
                k = int(ch_active.size)
                loc = {int(ch_active[i]): i for i in range(k)}

                # parent local index: -1 for sink, -2 for unreachable
                parent = np.full(k, -2, dtype=int)
                for i in range(k):
                    nh = int(next_hop[i])
                    if nh < 0:
                        parent[i] = -1
                    else:
                        parent[i] = int(loc.get(nh, -2))

                # Path hop counts (CH->...->sink) from the parent pointers.
                hop_len = np.full(k, np.inf, dtype=float)
                for i in range(k):
                    p = int(parent[i])
                    if p == -1:
                        hop_len[i] = 1.0
                        continue
                    if p < 0:
                        continue
                    seen = 0
                    cur = i
                    hops = 0
                    while True:
                        seen += 1
                        if seen > k + 1:
                            # Defensive: break potential cycles (shouldn't happen with Dijkstra tree)
                            hops = 0
                            break
                        pp = int(parent[cur])
                        if pp == -1:
                            hops += 1
                            break
                        if pp < 0:
                            hops = 0
                            break
                        hops += 1
                        cur = pp
                    hop_len[i] = float(hops) if hops > 0 else np.inf

                # Order far-to-near by kappa (unreachable last)
                kappa_safe = np.asarray(kappa, dtype=float)
                kappa_safe = np.where(np.isfinite(kappa_safe), kappa_safe, np.inf)
                order = np.argsort(kappa_safe)[::-1]

                # Each CH originates one aggregated packet; its "report weight" is (members + 1)
                pkt_cnt = np.ones(k, dtype=int)
                rep_w = (member_counts[ch_active] + 1.0).astype(float)

                # Successful per-CH transmitted packet counts (hotspot / Jain)
                q_tx = np.zeros(k, dtype=float)

                L = int(radio.l_data)
                rx_cost = float(radio.rx_energy(L))

                for i in order:
                    if pkt_cnt[i] <= 0:
                        continue

                    sender = int(ch_active[i])
                    p = int(parent[i])

                    # If sender already dead, drop
                    if net.residual_energy[sender] <= 0.0:
                        pkt_cnt[i] = 0
                        rep_w[i] = 0.0
                        continue

                    if p == -1:
                        # Direct hop to sink
                        d = float(net.dists_to_sink[sender])
                        tx_cost = float(pkt_cnt[i]) * float(radio.tx_energy(L, d))
                        pre = float(net.residual_energy[sender])
                        ok = pre >= tx_cost
                        # Debit attempt
                        net.residual_energy[sender] -= tx_cost
                        if ok:
                            pkts_to_sink += int(pkt_cnt[i])
                            pkt_hops += int(pkt_cnt[i])
                            q_tx[i] += float(pkt_cnt[i])
                            delivered_reports += int(round(float(rep_w[i])))
                        pkt_cnt[i] = 0
                        rep_w[i] = 0.0
                    elif p >= 0:
                        receiver = int(ch_active[p])
                        d = float(net.dists[sender, receiver])
                        tx_cost = float(pkt_cnt[i]) * float(radio.tx_energy(L, d))
                        rx_tot = float(pkt_cnt[i]) * rx_cost

                        pre_s = float(net.residual_energy[sender])
                        pre_r = float(net.residual_energy[receiver])
                        ok = (pre_s >= tx_cost) & (pre_r >= rx_tot)

                        # Debit attempt costs
                        net.residual_energy[sender] -= tx_cost
                        net.residual_energy[receiver] -= rx_tot

                        if ok:
                            pkt_hops += int(pkt_cnt[i])
                            q_tx[i] += float(pkt_cnt[i])
                            pkt_cnt[p] += int(pkt_cnt[i])
                            rep_w[p] += float(rep_w[i])

                        pkt_cnt[i] = 0
                        rep_w[i] = 0.0
                    else:
                        # Unreachable: attempt direct to sink if possible
                        d = float(net.dists_to_sink[sender])
                        tx_cost = float(pkt_cnt[i]) * float(radio.tx_energy(L, d))
                        pre = float(net.residual_energy[sender])
                        ok = pre >= tx_cost
                        net.residual_energy[sender] -= tx_cost
                        if ok:
                            pkts_to_sink += int(pkt_cnt[i])
                            pkt_hops += int(pkt_cnt[i])
                            q_tx[i] += float(pkt_cnt[i])
                            delivered_reports += int(round(float(rep_w[i])))
                        pkt_cnt[i] = 0
                        rep_w[i] = 0.0

                # Round-level multi-hop diagnostics
                finite_hops = hop_len[np.isfinite(hop_len)]
                if finite_hops.size > 0:
                    mh_avg_path_hops = float(np.mean(finite_hops))
                # Relay/hotspot metrics on successful transmissions
                if q_tx.size > 0:
                    mh_q_max = float(np.max(q_tx))
                    sum_q = float(np.sum(q_tx))
                    sum_q2 = float(np.sum(q_tx ** 2))
                    if sum_q > 0.0 and sum_q2 > 0.0:
                        mh_jain_q = float((sum_q ** 2) / (float(k) * sum_q2))
                    else:
                        mh_jain_q = float("nan")

        # If r_tx <= 0, multi-hop is disabled effectively and no CH->sink delivery occurs.

    # -------------------------
    # 4) Update liveness + stats
    # -------------------------
    net.alive_mask &= net.residual_energy > 0.0  

    return {
        "alive": int(np.sum(net.alive_mask)),
        "total_energy": float(np.sum(np.maximum(net.residual_energy, 0.0))),
        "pkts_to_sink": int(pkts_to_sink),
        # Multi-hop traffic: total successful packet-hops (CH->CH + CH->sink). Equals pkts_to_sink in single-hop.
        "pkt_hops": int(pkt_hops),
        "mh_avg_path_hops": float(mh_avg_path_hops),
        "mh_q_max": float(mh_q_max),
        "mh_jain_q": float(mh_jain_q),
        "n_ch": int(ch.size),
        "delivered_reports": int(delivered_reports),
    }
