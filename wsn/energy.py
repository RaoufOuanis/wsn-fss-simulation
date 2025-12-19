from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np

from .models import Network


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
    # 3) CH -> sink
    # -------------------------
    ch_alive0 = alive0[ch]
    d_sink = net.dists_to_sink[ch]
    e_tx_sink = radio.tx_energy_vec(radio.l_data, d_sink)

    e_pre_tx = net.residual_energy[ch]
    can_tx = ch_alive0 & (e_pre_tx >= e_tx_sink)
    pkts_to_sink = int(np.sum(can_tx))

    # Debit attempt for all alive CHs (prevents "immortal CH" freezing LND)
    net.residual_energy[ch[ch_alive0]] -= e_tx_sink[ch_alive0]

    delivered_reports = int(np.sum(member_counts[ch[can_tx]]) + pkts_to_sink)

    # -------------------------
    # 4) Update liveness + stats
    # -------------------------
    net.alive_mask &= net.residual_energy > 0.0  

    return {
        "alive": int(np.sum(net.alive_mask)),
        "total_energy": float(np.sum(np.maximum(net.residual_energy, 0.0))),
        "pkts_to_sink": int(pkts_to_sink),
        "n_ch": int(ch.size),
        "delivered_reports": int(delivered_reports),
    }
