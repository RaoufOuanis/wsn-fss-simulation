from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple
from time import perf_counter

import numpy as np
import pandas as pd

from ..models import Network
from ..energy import RadioParams, apply_round_energy
from ..fitness import FitnessParams

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


def _repair_ch_set_to_cover_all_alive(net: Network, ch: np.ndarray, rc: float) -> np.ndarray:
    """
    Option A (strict clustering): ensure every alive node is within Rc of at least one CH.
    This prevents "silent zombies" and makes LND measurable (reduces right-censoring artifacts).
    """
    n = int(net.n_nodes)
    alive = net.alive_mask
    if n <= 0 or not np.any(alive):
        return np.array([], dtype=int)

    ch = np.asarray(ch, dtype=int).reshape(-1)
    ch = ch[(ch >= 0) & (ch < n)]
    ch = np.unique(ch[alive[ch]])

    if ch.size == 0:
        alive_idx = np.where(alive)[0]
        best = int(alive_idx[int(np.argmax(net.residual_energy[alive_idx]))])
        ch = np.array([best], dtype=int)

    cache = net.get_grasp_cache(rc=float(rc), k_nn=1)
    within_rc = cache["within_rc"]  # (N,N) bool, includes self coverage

    is_ch = np.zeros(n, dtype=bool)
    is_ch[ch] = True

    # uncovered alive nodes
    covered = np.any(within_rc[ch], axis=0)
    uncovered = alive & (~covered)

    # incremental coverage counts against current uncovered set
    cover_counts = within_rc[:, uncovered].sum(axis=1).astype(np.int32)
    cover_counts[~alive] = 0
    cover_counts[is_ch] = 0

    while np.any(uncovered):
        cand = np.where((alive & ~is_ch) & (cover_counts > 0))[0]
        if cand.size == 0:
            # Should not happen since within_rc[i,i] is True, but keep as safety.
            u = int(np.where(uncovered)[0][0])
        else:
            best_cov = int(np.max(cover_counts[cand]))
            top = cand[cover_counts[cand] == best_cov]
            u = int(top[int(np.argmax(net.residual_energy[top]))])

        is_ch[u] = True
        ch = np.append(ch, u)

        newly_covered = within_rc[u] & uncovered
        if np.any(newly_covered):
            uncovered[newly_covered] = False
            cover_counts -= within_rc[:, newly_covered].sum(axis=1).astype(np.int32)

        cover_counts[u] = 0

    return np.unique(ch).astype(int)


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

    nfe_per_round: List[float] = []
    cpu_times: List[float] = []

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

        # Option A + repair: force Rc coverage of ALL alive nodes
        ch_indices = _repair_ch_set_to_cover_all_alive(net, ch_indices, rc=fit_params.rc)
        assignments, _, in_radius = net.assign_clusters(ch_indices, rc=fit_params.rc)
        assignments = np.where(in_radius, assignments, -1).astype(int)
        stats = apply_round_energy(net, ch_indices, assignments, radio)


        pkts_to_sink_round = int(stats.get("pkts_to_sink", 0))
        delivered_round = int(stats.get("delivered_reports", pkts_to_sink_round))
        n_ch_round = int(stats.get("n_ch", int(np.asarray(ch_indices).size)))

        pkts_to_sink_cum += pkts_to_sink_round
        delivered_cum += delivered_round

        if delivered_round > 0:
            r_last = r

        rounds.append(int(r))
        alive_history.append(int(stats.get("alive", 0)))
        energy_history.append(float(stats.get("total_energy", 0.0)))

        delivered_round_history.append(int(delivered_round))
        pkts_to_sink_round_history.append(int(pkts_to_sink_round))
        n_ch_round_history.append(int(n_ch_round))

        delivered_cum_history.append(int(delivered_cum))
        pkts_to_sink_cum_history.append(int(pkts_to_sink_cum))

        nfe_per_round.append(float(nfe))
        cpu_times.append(float(t1 - t0))

        if int(stats.get("alive", 0)) == 0:
            lnd = r + 1
            break

    if lnd is None:
        lnd = int(max_rounds)

    R_last = int(r_last if r_last >= 0 else 0)

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

        "avg_alive": float(np.mean(alive_history)) if alive_history else 0.0,
        "avg_energy": float(np.mean(energy_history)) if energy_history else 0.0,
        "avg_nfe": float(np.mean(nfe_per_round)) if nfe_per_round else 0.0,
        "avg_cpu_time": float(np.mean(cpu_times)) if cpu_times else 0.0,

        "avg_delivered_reports_round": float(np.mean(delivered_round_history)) if delivered_round_history else 0.0,
        "avg_pkts_to_sink_round": float(np.mean(pkts_to_sink_round_history)) if pkts_to_sink_round_history else 0.0,
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

            "throughput_cum": delivered_cum_history,
            "delivered_reports_round": delivered_round_history,

            "pkts_to_sink_cum": pkts_to_sink_cum_history,
            "pkts_to_sink_round": pkts_to_sink_round_history,

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
