from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
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
from ..algorithms.so_wsn import run_so_wsn, SOParams
from ..algorithms.gjo_wsn import run_gjo_wsn, GJOParams
from ..algorithms.esogjo_wsn import run_esogjo_wsn, ESOGJOParams
from ..algorithms.emogjo_wsn import run_emogjo_wsn, EMOGJOParams
from ..algorithms.protocols import (
    run_leach_wsn, LEACHParams, LEACHState,
    run_heed_wsn, HEEDParams,
    run_sep_wsn, SEPParams, SEPState,
    run_greedy_wsn, GreedyParams,
)
from ..algorithms.eem_leach_abc_wsn import (
    run_eem_leach_abc_wsn, EEMParams, EEMState,
)


# -----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Experimental budget control
# ----------------------------------------------------------------------------
# Align the number of iterations per round across metaheuristics (budget-matched).
# We use the FSS default as the reference budget.
BUDGET_NITER = FSSParams(seed=0).n_iter

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
    elif algo_name == "FSS_legacy":
        # Ablation: legacy construction score (no coverage-gain weighting)
        params.use_phase2 = True
        params.coverage_gain_auto = False
        params.coverage_gain_weight = 0.0
    elif algo_name == "FSS_cov":
        # Ablation: always-on coverage-gain weighting (corner fix forced everywhere)
        params.use_phase2 = True
        params.coverage_gain_auto = False
        params.coverage_gain_weight = 1.0
    elif algo_name == "FSS_noPhase2":
        params.use_phase2 = False
    elif algo_name == "FSS_noEnergy":
        params.use_phase2 = True
        params.theta = 0.0
    elif algo_name == "FSS_noLS":
        params.use_phase2 = True
        params.Lmax = 0
    elif algo_name == "FSS_noRepairReg":
        # Ablation: disable the repair-dependence regularizer (lambda*P)
        # This is controlled via FitnessParams.lam in the simulator.
        params.use_phase2 = True
    else:
        raise ValueError(f"Unknown FSS variant: {algo_name}")

    return params


def _apply_fss_param_overrides(params: FSSParams, overrides: Dict[str, Any]) -> None:
    """Apply attribute overrides to an FSSParams instance (best-effort type casting)."""

    if not overrides:
        return

    for k, v in overrides.items():
        key = str(k)
        if not hasattr(params, key):
            raise ValueError(f"Unknown FSSParams override: {key}")

        cur = getattr(params, key)
        try:
            if isinstance(cur, bool):
                setattr(params, key, bool(v))
            elif isinstance(cur, int):
                setattr(params, key, int(v))
            elif isinstance(cur, float):
                setattr(params, key, float(v))
            else:
                setattr(params, key, v)
        except Exception as e:
            raise ValueError(f"Invalid override for {key}: {v} ({e})")


def _apply_fitness_param_overrides(params: FitnessParams, overrides: Dict[str, Any]) -> None:
    """Apply attribute overrides to a FitnessParams instance (best-effort type casting)."""

    if not overrides:
        return

    for k, v in overrides.items():
        key = str(k)
        if not hasattr(params, key):
            raise ValueError(f"Unknown FitnessParams override: {key}")

        cur = getattr(params, key)
        try:
            if isinstance(cur, bool):
                setattr(params, key, bool(v))
            elif isinstance(cur, int):
                setattr(params, key, int(v))
            elif isinstance(cur, float):
                setattr(params, key, float(v))
            else:
                setattr(params, key, v)
        except Exception as e:
            raise ValueError(f"Invalid override for {key}: {v} ({e})")


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
    fss_params_overrides: Optional[Dict[str, Any]] = None,
    fitness_params_overrides: Optional[Dict[str, Any]] = None,
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
    # Keep a single source of truth for the radio model (fitness + simulation).
    fit_params.radio = radio

    # Ablation: remove the repair-dependence regularizer term (lambda * P(H0)).
    # We keep repair itself (H+ construction) identical; only the penalty weight is set to 0.
    if algo_name == "FSS_noRepairReg":
        fit_params.lam = 0.0

    if fitness_params_overrides:
        _apply_fitness_param_overrides(fit_params, fitness_params_overrides)

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

    eem_state = EEMState.initialize(net.n_nodes)
    eem_params = EEMParams(seed=seed, cr=0.10, mu=0.70)

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

    # FSS Phase-II diagnostics (per round; defaults for non-FSS)
    fss_best_source_history: List[str] = []
    fss_phase2_guard_triggered_history: List[bool] = []
    fss_phase2_energy_ratio_history: List[float] = []

    # Fitness component diagnostics: phase2 - phase1 (NaN when unavailable)
    fss_d_CE_history: List[float] = []
    fss_d_CeD_history: List[float] = []
    fss_d_CeS_history: List[float] = []
    fss_d_CeL_history: List[float] = []
    fss_d_CeR_history: List[float] = []
    fss_d_P_history: List[float] = []
    fss_d_F_base_history: List[float] = []

    delivered_cum = 0
    pkts_to_sink_cum = 0
    pkt_hops_cum = 0

    # Energy spent (floored-at-zero total energy tracking)
    total_energy_prev = float(np.sum(np.maximum(net.initial_energy, 0.0)))
    energy_spent_round_history: List[float] = []

    nfe_per_round: List[float] = []
    cpu_times: List[float] = []

    # Repair diagnostics: raw CH count vs repaired CH count (per round)
    n_ch_raw_round_history: List[int] = []
    n_ch_added_by_repair_round_history: List[int] = []

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

        # Per-round FSS Phase-II diagnostics (defaults)
        fss_best_source_round = "na"
        fss_guard_triggered_round = False
        fss_energy_ratio_round = float("nan")

        d_CE = float("nan")
        d_CeD = float("nan")
        d_CeS = float("nan")
        d_CeL = float("nan")
        d_CeR = float("nan")
        d_P = float("nan")
        d_F_base = float("nan")

        if algo_name.startswith("FSS"):
            fss_params = _make_fss_params_for_algo(algo_name, seed=seed + r)
            if fss_params_overrides:
                _apply_fss_param_overrides(fss_params, fss_params_overrides)
            res = run_fss_wsn(net, fit_params, fss_params)
            nfe = float(getattr(res, "nfe", 0.0))

            hist = getattr(res, "history", {}) or {}
            fss_best_source_round = str(hist.get("best_source", "unknown"))
            fss_guard_triggered_round = bool(hist.get("phase2_guard_triggered", False))
            e2 = hist.get("phase2_best_energy_est", float("nan"))
            e1 = hist.get("phase1_best_energy_est", float("nan"))
            try:
                e2f = float(e2)
                e1f = float(e1)
                fss_energy_ratio_round = float(e2f / e1f) if (np.isfinite(e2f) and np.isfinite(e1f) and e1f != 0.0) else float("nan")
            except Exception:
                fss_energy_ratio_round = float("nan")

            # Component deltas: phase2 - phase1
            def _d(key: str) -> float:
                try:
                    p2 = float(hist.get(f"phase2_{key}", float("nan")))
                    p1 = float(hist.get(f"phase1_{key}", float("nan")))
                    if np.isfinite(p2) and np.isfinite(p1):
                        return float(p2 - p1)
                except Exception:
                    return float("nan")
                return float("nan")

            d_CE = _d("CE")
            d_CeD = _d("CeD")
            d_CeS = _d("CeS")
            d_CeL = _d("CeL")
            d_CeR = _d("CeR")
            d_P = _d("P")
            d_F_base = _d("F_base")

        elif algo_name == "PSO":
            params = PSOParams(seed=seed + r, n_iter=BUDGET_NITER)
            res = run_pso_wsn(net, fit_params, params)
            nfe = float(getattr(res, "nfe", 0.0))
            if nfe <= 0.0:
                nfe = float(params.n_iter * params.n_particles)

        elif algo_name == "ESOGJO":
            params = ESOGJOParams(seed=seed + r, n_iter=BUDGET_NITER)
            res = run_esogjo_wsn(net, fit_params, params)
            nfe = float(getattr(res, "nfe", 0.0))
            if nfe <= 0.0:
                nfe = float(params.n_iter * params.pop_size)

        elif algo_name == "GWO":
            params = GWOParams(seed=seed + r, n_iter=BUDGET_NITER)
            res = run_gwo_wsn(net, fit_params, params)
            nfe = float(getattr(res, "nfe", 0.0))
            if nfe <= 0.0:
                nfe = float(params.n_iter * params.n_wolves)

        elif algo_name == "ABC":
            params = ABCParams(seed=seed + r, n_iter=BUDGET_NITER)
            res = run_abc_wsn(net, fit_params, params)
            nfe = float(getattr(res, "nfe", 0.0))
            if nfe <= 0.0:
                # Fallback lower-bound: init + employed + onlooker (scouts excluded)
                nfe = float((2 * params.n_iter + 1) * params.n_food_sources)

        elif algo_name == "SO":
            params = SOParams(seed=seed + r, n_iter=BUDGET_NITER, pop_size=30)
            res = run_so_wsn(net, fit_params, params)
            nfe = float(getattr(res, "nfe", 0.0))
            if nfe <= 0.0:
                nfe = float(params.n_iter * params.pop_size)

        elif algo_name == "GJO":
            params = GJOParams(seed=seed + r, n_iter=BUDGET_NITER, pop_size=30)
            res = run_gjo_wsn(net, fit_params, params)
            nfe = float(getattr(res, "nfe", 0.0))
            if nfe <= 0.0:
                nfe = float(params.n_iter * params.pop_size)

        elif algo_name == "EMOGJO":
            params = EMOGJOParams(seed=seed + r, n_iter=BUDGET_NITER, pop_size=30)
            res = run_emogjo_wsn(net, fit_params, params)
            nfe = float(getattr(res, "nfe", 0.0))
            if nfe <= 0.0:
                nfe = float(params.n_iter * params.pop_size)

        elif algo_name == "EMOGJO_paperCH":
            params = EMOGJOParams(seed=seed + r, n_iter=BUDGET_NITER, pop_size=30)
            params.use_paper_ch_fitness = True
            res = run_emogjo_wsn(net, fit_params, params)
            nfe = float(getattr(res, "nfe", 0.0))
            if nfe <= 0.0:
                nfe = float(params.n_iter * params.pop_size)

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

        elif algo_name == "EEM_LEACH_ABC":
            res = run_eem_leach_abc_wsn(
                net, fit_params, eem_params, round_idx=r, state=eem_state,
            )
            nfe = 0.0

        else:
            raise ValueError(f"Unknown algorithm {algo_name}")

        t1 = perf_counter()

        # Store Phase-II diagnostics (kept for all algos; "na"/NaN for non-FSS)
        fss_best_source_history.append(str(fss_best_source_round))
        fss_phase2_guard_triggered_history.append(bool(fss_guard_triggered_round))
        fss_phase2_energy_ratio_history.append(float(fss_energy_ratio_round))

        fss_d_CE_history.append(float(d_CE))
        fss_d_CeD_history.append(float(d_CeD))
        fss_d_CeS_history.append(float(d_CeS))
        fss_d_CeL_history.append(float(d_CeL))
        fss_d_CeR_history.append(float(d_CeR))
        fss_d_P_history.append(float(d_P))
        fss_d_F_base_history.append(float(d_F_base))

        # ---------------- apply radio model ----------------
        ch_indices = _ensure_nonempty_ch(net, res.best_ch_indices)
        n_ch_raw = int(np.asarray(ch_indices).size)

        # Option A + repair: force Rc coverage of ALL alive nodes
        ch_indices = _repair_ch_set(
            net,
            ch_indices,
            rc=fit_params.rc,
            r_tx=getattr(fit_params, "r_tx", fit_params.rc),
            multihop=bool(getattr(fit_params, "multihop", False)),
            repair_params=fit_params.repair,
        )

        n_ch_repaired = int(np.asarray(ch_indices).size)
        n_added = int(max(0, n_ch_repaired - n_ch_raw))
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

        # Energy spent this round, based on floored-at-zero total energy
        spent = float(max(0.0, total_energy_prev - total_energy_now))
        energy_spent_round_history.append(spent)
        total_energy_prev = total_energy_now

        delivered_round_history.append(int(delivered_round))
        pkts_to_sink_round_history.append(int(pkts_to_sink_round))
        pkt_hops_round_history.append(int(pkt_hops_round))
        n_ch_round_history.append(int(n_ch_round))

        n_ch_raw_round_history.append(int(n_ch_raw))
        n_ch_added_by_repair_round_history.append(int(n_added))

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

        # Diagnostics: how much deterministic repair inflates the CH set
        "avg_n_ch_raw_round": float(np.mean(n_ch_raw_round_history)) if n_ch_raw_round_history else 0.0,
        "avg_n_ch_added_by_repair_round": float(np.mean(n_ch_added_by_repair_round_history)) if n_ch_added_by_repair_round_history else 0.0,

        # FSS Phase-II diagnostics (meaningful only when algo startswith 'FSS')
        "fss_frac_best_from_phase2": float(np.mean([s == "phase2" for s in fss_best_source_history])) if fss_best_source_history else float("nan"),
        "fss_frac_phase2_guard_triggered": float(np.mean(np.asarray(fss_phase2_guard_triggered_history, dtype=float))) if fss_phase2_guard_triggered_history else float("nan"),
        "fss_phase2_energy_ratio_mean": float(_nanmean(fss_phase2_energy_ratio_history)) if fss_phase2_energy_ratio_history else float("nan"),

        # Component delta means (phase2 - phase1)
        "fss_d_CE_mean": float(_nanmean(fss_d_CE_history)) if fss_d_CE_history else float("nan"),
        "fss_d_CeD_mean": float(_nanmean(fss_d_CeD_history)) if fss_d_CeD_history else float("nan"),
        "fss_d_CeS_mean": float(_nanmean(fss_d_CeS_history)) if fss_d_CeS_history else float("nan"),
        "fss_d_CeL_mean": float(_nanmean(fss_d_CeL_history)) if fss_d_CeL_history else float("nan"),
        "fss_d_CeR_mean": float(_nanmean(fss_d_CeR_history)) if fss_d_CeR_history else float("nan"),
        "fss_d_P_mean": float(_nanmean(fss_d_P_history)) if fss_d_P_history else float("nan"),
        "fss_d_F_base_mean": float(_nanmean(fss_d_F_base_history)) if fss_d_F_base_history else float("nan"),

        "bs_mode": bs_mode,
        "sink_x": float(sink_pos[0]),
        "sink_y": float(sink_pos[1]),

        # -------------------------------------------------
        # Reproducibility: consolidated config (paper tables)
        # -------------------------------------------------
        "budget_niter": int(BUDGET_NITER),

        # Fitness / surrogate weights
        "fit_rc": float(fit_params.rc),
        "fit_multihop": bool(getattr(fit_params, "multihop", False)),
        "fit_r_tx": float(getattr(fit_params, "r_tx", float("nan"))),
        "fit_w1": float(getattr(fit_params, "w1", float("nan"))),
        "fit_w2": float(getattr(fit_params, "w2", float("nan"))),
        "fit_w3": float(getattr(fit_params, "w3", float("nan"))),
        "fit_lam": float(getattr(fit_params, "lam", float("nan"))),
        "fit_w_relay": float(getattr(fit_params, "w_relay", float("nan"))),

        # Repair parameters (ties are deterministic: smallest index)
        "repair_alpha1": float(getattr(getattr(fit_params, "repair", None), "alpha1", float("nan"))),
        "repair_alpha2": float(getattr(getattr(fit_params, "repair", None), "alpha2", float("nan"))),
        "repair_alpha3": float(getattr(getattr(fit_params, "repair", None), "alpha3", float("nan"))),

        # Radio model constants (first-order)
        "radio_E_elec": float(getattr(radio, "E_elec", float("nan"))),
        "radio_eps_fs": float(getattr(radio, "eps_fs", float("nan"))),
        "radio_eps_mp": float(getattr(radio, "eps_mp", float("nan"))),
        "radio_E_da": float(getattr(radio, "E_da", float("nan"))),
        "radio_l_data": int(getattr(radio, "l_data", 0)),
        "radio_l_ctrl": int(getattr(radio, "l_ctrl", 0)),
        "radio_d0": float(getattr(radio, "d0", float("nan"))),
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

            "n_ch_raw_round": n_ch_raw_round_history,
            "n_ch_added_by_repair_round": n_ch_added_by_repair_round_history,

            "fss_best_source": fss_best_source_history,
            "fss_phase2_guard_triggered": fss_phase2_guard_triggered_history,
            "fss_phase2_energy_ratio": fss_phase2_energy_ratio_history,

            "fss_d_CE": fss_d_CE_history,
            "fss_d_CeD": fss_d_CeD_history,
            "fss_d_CeS": fss_d_CeS_history,
            "fss_d_CeL": fss_d_CeL_history,
            "fss_d_CeR": fss_d_CeR_history,
            "fss_d_P": fss_d_P_history,
            "fss_d_F_base": fss_d_F_base_history,
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
    fss_params_overrides: Optional[Dict[str, Any]] = None,
    fitness_params_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run all experiments and return (summary_df, history_df)."""

    summaries: List[Dict] = []
    histories: List[pd.DataFrame] = []

    for sc in scenarios:
        for algo in algos:
            for run in range(int(n_runs)):
                seed = int(base_seed + run)
                print(f"[run_experiments] {sc.name} - {algo} - run {run}")
                summary, hist_df = simulate_lifetime(
                    sc,
                    algo,
                    seed,
                    max_rounds=max_rounds,
                    bs_mode=bs_mode,
                    fss_params_overrides=fss_params_overrides,
                    fitness_params_overrides=fitness_params_overrides,
                )
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
