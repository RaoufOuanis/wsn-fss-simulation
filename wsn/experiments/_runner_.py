from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple
from time import perf_counter

import numpy as np
import pandas as pd

from ..models import Network
from ..energy import RadioParams, apply_round_energy
from ..fitness import FitnessParams
from ..repair import (
    repair_ch_set_to_cover_all_alive,
    repair_ch_set_to_cover_and_connect_to_sink,
    RepairParams,
)
from ..multihop import MultiHopParams

from ..algorithms.fss_wsn import run_fss_wsn, FSSParams
from ..algorithms.pso_wsn import run_pso_wsn, PSOParams
from ..algorithms.gwo_wsn import run_gwo_wsn, GWOParams
from ..algorithms.abc_wsn import run_abc_wsn, ABCParams
from ..algorithms.protocols import (
    run_leach_wsn, LEACHParams, LEACHState,
    run_heed_wsn, HEEDParams,
    run_sep_wsn, SEPParams, SEPState,
    run_greedy_wsn, GreedyParams,
)


# -----------------------------------------------------------------------------
# Scenario
# -----------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    n_nodes: int
    heterogenous: bool
    area_size: float = 100.0
    adv_fraction: float = 0.2
    e0: float = 0.5     # initial energy of normal nodes (J)
    e_adv: float = 1.0  # initial energy of advanced nodes (J)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _make_fss_params_for_algo(algo_name: str, seed: int) -> FSSParams:
    """Return FSSParams for the selected FSS variant."""
    params = FSSParams(seed=seed)

    if algo_name == "FSS":
        params.use_phase2 = True
    elif algo_name == "FSS_noPhase2":
        params.use_phase2 = False
    elif algo_name == "FSS_noEnergy":
        params.use_phase2 = True
        params.theta = 0.0
    elif algo_name == "FSS_noLS":
        params.use_phase2 = True
        params.Lmax = 0
    else:
        raise ValueError(f"Unknown FSS variant: {algo_name}")

    return params


def _ensure_nonempty_ch(net: Network, ch_indices: np.ndarray) -> np.ndarray:
    """Ensure we have a valid non-empty CH set among alive nodes."""
    ch = np.asarray(ch_indices, dtype=int).reshape(-1)
    alive_idx = net.get_alive_indices()
    if alive_idx.size == 0:
        return np.array([], dtype=int)

    ch = ch[(ch >= 0) & (ch < net.n_nodes)]
    if ch.size > 0:
        ch = np.unique(ch[net.alive_mask[ch]])
        if ch.size > 0:
            return ch

    best = int(alive_idx[int(np.argmax(net.residual_energy[alive_idx]))])
    return np.array([best], dtype=int)


def _sink_position(area_size: float, bs_mode: str) -> Tuple[float, float]:
    L = float(area_size)
    if bs_mode == "corner":
        return (0.0, 0.0)
    return (L / 2.0, L / 2.0)


def _repair_ch_set(
    net: Network,
    ch: np.ndarray,
    rc: float,
    r_tx: float,
    multihop: bool,
    repair_params: RepairParams = RepairParams(),
) -> np.ndarray:
    """Deterministic Repair_t(H0) enforcing strict Rc-coverage (paper Section 3.5).

    This wrapper exists to keep the simulation code stable while ensuring the
    simulator uses the same repair logic as the fitness evaluation.
    """
    if bool(multihop):
        return repair_ch_set_to_cover_and_connect_to_sink(
            net,
            ch0=ch,
            rc=float(rc),
            r_tx=float(r_tx),
            params=repair_params,
        )
    return repair_ch_set_to_cover_all_alive(net, ch0=ch, rc=float(rc), params=repair_params)

# -----------------------------------------------------------------------------
# Core simulation
# -----------------------------------------------------------------------------

def simulate_lifetime(
    scenario: Scenario,
    algo_name: str,
    seed: int,
    max_rounds: int = 5000,
    bs_mode: str = "center",
) -> Tuple[Dict, pd.DataFrame]:
    """Simulate network lifetime for one (scenario, algorithm, seed)."""

    sink_pos = _sink_position(scenario.area_size, bs_mode)

    net = Network.random_network(
        n_nodes=scenario.n_nodes,
        area_size=scenario.area_size,
        heterogenous=scenario.heterogenous,
        adv_fraction=scenario.adv_fraction,
        e0=scenario.e0,
        e_adv=scenario.e_adv,
        seed=seed,
        sink_pos=sink_pos,
    )

    radio = RadioParams()
    fit_params = FitnessParams(rc=25.0)

    # Protocol states
    leach_state = LEACHState.initialize(net.n_nodes)
    sep_state = SEPState.initialize(net.n_nodes)

    leach_params = LEACHParams(seed=seed, p_opt=0.05, min_ch=1, max_ch=None)
    heed_params = HEEDParams(seed=seed, p_init=0.05, c_min=0.02, n_iter=3)
    sep_params = SEPParams(
        seed=seed,
        p_opt=0.05,
        min_ch=1,
        max_ch=None,
        e0=scenario.e0,
        e_adv=scenario.e_adv,
        adv_fraction=scenario.adv_fraction,
    )
    greedy_params = GreedyParams(seed=seed, n_ch=0)

    n0 = int(scenario.n_nodes)

    # Lifetime metrics
    fnd = None
    hnd = None
    lnd = None

    # Per-round histories
    rounds: List[int] = []
    alive_history: List[int] = []
    energy_history: List[float] = []

    delivered_cum_history: List[int] = []
    pkts_to_sink_cum_history: List[int] = []
    delivered_round_history: List[int] = []
    pkts_to_sink_round_history: List[int] = []
    n_ch_round_history: List[int] = []

    delivered_cum = 0
    pkts_to_sink_cum = 0
    pkt_hops_cum = 0

    # Energy spent (floored-at-zero total energy tracking)
    total_energy_prev = float(np.sum(np.maximum(net.initial_energy, 0.0)))
    energy_spent_round_history: List[float] = []

    nfe_per_round: List[float] = []
    cpu_times: List[float] = []

    # Multi-hop diagnostics (per round; NaN for single-hop)
    mh_avg_path_hops_history: List[float] = []
    mh_q_max_history: List[float] = []
    mh_jain_q_history: List[float] = []
    pkt_hops_cum_history: List[int] = []
    pkt_hops_round_history: List[int] = []

    # Operational lifetime: last round with delivered_reports_round > 0
    r_last = -1

    for r in range(int(max_rounds)):
        alive_indices = net.get_alive_indices()
        if alive_indices.size == 0:
            lnd = r
            break

        if fnd is None and alive_indices.size < n0:
            fnd = r
        if hnd is None and alive_indices.size <= n0 / 2:
            hnd = r

        # ---------------- optimization (choose CHs) ----------------
        t0 = perf_counter()

        if algo_name.startswith("FSS"):
            fss_params = _make_fss_params_for_algo(algo_name, seed=seed + r)
            res = run_fss_wsn(net, fit_params, fss_params)
            nfe = float(getattr(res, "nfe", 0.0))

        elif algo_name == "PSO":
            params = PSOParams(seed=seed + r)
            res = run_pso_wsn(net, fit_params, params)
            nfe = float(params.n_iter * params.n_particles)

        elif algo_name == "GWO":
            params = GWOParams(seed=seed + r)
            res = run_gwo_wsn(net, fit_params, params)
            nfe = float(params.n_iter * params.n_wolves)

        elif algo_name == "ABC":
            params = ABCParams(seed=seed + r)
            res = run_abc_wsn(net, fit_params, params)
            nfe = float(params.n_iter * params.n_food_sources)

        elif algo_name == "LEACH":
            res = run_leach_wsn(net, fit_params, leach_params, round_idx=r, state=leach_state)
            nfe = 0.0

        elif algo_name == "HEED":
            res = run_heed_wsn(net, fit_params, heed_params, round_idx=r)
            nfe = 0.0

        elif algo_name == "SEP":
            res = run_sep_wsn(net, fit_params, sep_params, round_idx=r, state=sep_state)
            nfe = 0.0

        elif algo_name == "Greedy":
            res = run_greedy_wsn(net, fit_params, greedy_params, round_idx=r)
            nfe = 0.0

        else:
            raise ValueError(f"Unknown algorithm {algo_name}")

        t1 = perf_counter()

        # ---------------- apply radio model ----------------
        ch_indices = _ensure_nonempty_ch(net, res.best_ch_indices)

        # Option A + repair: force Rc coverage of ALL alive nodes (+ CH->sink connectivity in multi-hop mode)
        ch_indices = _repair_ch_set(
            net,
            ch_indices,
            rc=fit_params.rc,
            r_tx=getattr(fit_params, "r_tx", fit_params.rc),
            multihop=bool(getattr(fit_params, "multihop", False)),
            repair_params=fit_params.repair,
        )
        assignments, _, in_radius = net.assign_clusters(ch_indices, rc=fit_params.rc)
        assignments = np.where(in_radius, assignments, -1).astype(int)
        mh = None
        if bool(getattr(fit_params, "multihop", False)):
            mh = MultiHopParams(r_tx=float(getattr(fit_params, "r_tx", fit_params.rc)))
        stats = apply_round_energy(net, ch_indices, assignments, radio, multihop=mh)


        pkts_to_sink_round = int(stats.get("pkts_to_sink", 0))
        pkt_hops_round = int(stats.get("pkt_hops", 0))
        delivered_round = int(stats.get("delivered_reports", pkts_to_sink_round))
        n_ch_round = int(stats.get("n_ch", int(np.asarray(ch_indices).size)))

        pkts_to_sink_cum += pkts_to_sink_round
        pkt_hops_cum += pkt_hops_round
        delivered_cum += delivered_round

        if delivered_round > 0:
            r_last = r

        rounds.append(int(r))
        alive_history.append(int(stats.get("alive", 0)))
        total_energy_now = float(stats.get("total_energy", 0.0))
        energy_history.append(total_energy_now)

        spent = float(max(0.0, total_energy_prev - total_energy_now))
        energy_spent_round_history.append(spent)
        total_energy_prev = total_energy_now

        delivered_round_history.append(int(delivered_round))
        pkts_to_sink_round_history.append(int(pkts_to_sink_round))
        pkt_hops_round_history.append(int(pkt_hops_round))
        n_ch_round_history.append(int(n_ch_round))

        delivered_cum_history.append(int(delivered_cum))
        pkts_to_sink_cum_history.append(int(pkts_to_sink_cum))
        pkt_hops_cum_history.append(int(pkt_hops_cum))

        mh_avg_path_hops_history.append(float(stats.get("mh_avg_path_hops", float("nan"))))
        mh_q_max_history.append(float(stats.get("mh_q_max", float("nan"))))
        mh_jain_q_history.append(float(stats.get("mh_jain_q", float("nan"))))

        nfe_per_round.append(float(nfe))
        cpu_times.append(float(t1 - t0))

        if int(stats.get("alive", 0)) == 0:
            lnd = r + 1
            break

    if lnd is None:
        lnd = int(max_rounds)

    R_last = int(r_last if r_last >= 0 else 0)

    total_energy_spent = float(np.sum(np.asarray(energy_spent_round_history, dtype=float))) if energy_spent_round_history else 0.0
    energy_per_report = float(total_energy_spent / float(delivered_cum)) if int(delivered_cum) > 0 else float("nan")

    def _nanmean(xs: List[float]) -> float:
        arr = np.asarray(xs, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else float("nan")

    def _nanmax(xs: List[float]) -> float:
        arr = np.asarray(xs, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.max(arr)) if arr.size else float("nan")

    summary = {
        "scenario": scenario.name,
        "algo": algo_name,
        "seed": int(seed),

        "FND": int(fnd if fnd is not None else max_rounds),
        "HND": int(hnd if hnd is not None else max_rounds),
        "LND": int(lnd),

        # Operational lifetime
        "R_last": int(R_last),

        # Primary throughput for paper: WSN-classique
        "throughput": int(delivered_cum),

        # Secondary: aggregated CH->sink packets
        "throughput_pkts_to_sink": int(pkts_to_sink_cum),

        # Multi-hop traffic (packet-hops): CH->CH + CH->sink successful transmissions
        "pkt_hops": int(pkt_hops_cum),

        # Energy efficiency per delivered report
        "energy_per_report": float(energy_per_report),

        # Multi-hop diagnostics (NaN when not in multi-hop mode)
        "mh_avg_path_hops": float(_nanmean(mh_avg_path_hops_history)),
        "mh_q_max": float(_nanmax(mh_q_max_history)),
        "mh_jain_q": float(_nanmean(mh_jain_q_history)),

        "avg_alive": float(np.mean(alive_history)) if alive_history else 0.0,
        "avg_energy": float(np.mean(energy_history)) if energy_history else 0.0,
        "avg_nfe": float(np.mean(nfe_per_round)) if nfe_per_round else 0.0,
        "avg_cpu_time": float(np.mean(cpu_times)) if cpu_times else 0.0,

        "avg_delivered_reports_round": float(np.mean(delivered_round_history)) if delivered_round_history else 0.0,
        "avg_pkts_to_sink_round": float(np.mean(pkts_to_sink_round_history)) if pkts_to_sink_round_history else 0.0,
        "avg_pkt_hops_round": float(np.mean(pkt_hops_round_history)) if pkt_hops_round_history else 0.0,
        "avg_n_ch_round": float(np.mean(n_ch_round_history)) if n_ch_round_history else 0.0,

        "bs_mode": bs_mode,
        "sink_x": float(sink_pos[0]),
        "sink_y": float(sink_pos[1]),
    }

    history_df = pd.DataFrame(
        {
            "round": rounds,
            "alive": alive_history,
            "total_energy": energy_history,

            "energy_spent_round": energy_spent_round_history,

            "throughput_cum": delivered_cum_history,
            "delivered_reports_round": delivered_round_history,

            "pkts_to_sink_cum": pkts_to_sink_cum_history,
            "pkts_to_sink_round": pkts_to_sink_round_history,

            "pkt_hops_cum": pkt_hops_cum_history,
            "pkt_hops_round": pkt_hops_round_history,

            "mh_avg_path_hops": mh_avg_path_hops_history,
            "mh_q_max": mh_q_max_history,
            "mh_jain_q": mh_jain_q_history,

            "n_ch_round": n_ch_round_history,
        }
    )

    history_df["scenario"] = scenario.name
    history_df["algo"] = algo_name
    history_df["seed"] = int(seed)
    history_df["bs_mode"] = bs_mode
    history_df["sink_x"] = float(sink_pos[0])
    history_df["sink_y"] = float(sink_pos[1])

    return summary, history_df


def run_experiments(
    scenarios: List[Scenario],
    algos: List[str],
    n_runs: int = 30,
    max_rounds: int = 5000,
    base_seed: int = 0,
    save_prefix: str | None = None,
    bs_mode: str = "center",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run all experiments and return (summary_df, history_df)."""

    summaries: List[Dict] = []
    histories: List[pd.DataFrame] = []

    for sc in scenarios:
        for algo in algos:
            for run in range(int(n_runs)):
                seed = int(base_seed + run)
                print(f"[run_experiments] {sc.name} - {algo} - run {run}")
                summary, hist_df = simulate_lifetime(sc, algo, seed, max_rounds=max_rounds, bs_mode=bs_mode)
                summaries.append(summary)
                histories.append(hist_df)

    summary_df = pd.DataFrame(summaries)
    history_df = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()

    if save_prefix is not None:
        summary_path = f"{save_prefix}_summary.csv"
        history_path = f"{save_prefix}_timeseries.csv"
        summary_df.to_csv(summary_path, index=False)
        history_df.to_csv(history_path, index=False)
        print(f"Saved summary to {summary_path}")
        print(f"Saved timeseries to {history_path}")

    return summary_df, history_df
