import time
import json
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from wsn.models import Network
from wsn.energy import RadioParams, apply_round_energy
from wsn.fitness import FitnessParams, fitness

from wsn.algorithms.fss_wsn import run_fss_wsn, FSSParams
from wsn.algorithms.pso_wsn import run_pso_wsn, PSOParams
from wsn.algorithms.gwo_wsn import run_gwo_wsn, GWOParams
from wsn.algorithms.abc_wsn import run_abc_wsn, ABCParams
from wsn.algorithms.protocols import (
    run_leach_wsn, LEACHParams, LEACHState,
    run_heed_wsn, HEEDParams,
    run_sep_wsn, SEPParams, SEPState,
    run_greedy_wsn, GreedyParams,
)

from wsn.plot_style import (
    algo_color,
    direction_phrase,
    display_algo_name,
    hex_to_rgba,
    is_central_algo,
    metric_display_name,
)


# =========================================================
# Page
# =========================================================

st.set_page_config(page_title="WSN Clustering Simulator", layout="wide")
st.title("WSN Clustering Simulator")
st.caption(
    "Cadre: clustering centralisé, single-hop CH→BS, énergie radio (LEACH-style), métriques FND/HND/LND + throughput."
)

# =========================================================
# Helpers
# =========================================================

ALGO_KEYS = [
    "FSS",
    "FSS_noPhase2",
    "FSS_noEnergy",
    "FSS_noLS",
    "PSO",
    "GWO",
    "ABC",
    "LEACH",
    "HEED",
    "SEP",
    "Greedy",
]


def _algo_display_name(key: str) -> str:
    mapping = {
        "FSS": "FSS-WSN (proposed)",
        "FSS_noPhase2": "FSS – ablation: no Phase II",
        "FSS_noEnergy": "FSS – ablation: no energy filter (θ = 0)",
        "FSS_noLS": "FSS – ablation: no local search",
        "PSO": "PSO",
        "GWO": "GWO",
        "ABC": "ABC",
        "LEACH": "LEACH",
        "HEED": "HEED",
        "SEP": "SEP",
        "Greedy": "Greedy",
    }
    return mapping.get(key, key)


def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _hashable_config(d: Dict) -> str:
    """Stable hash key for caching and session-state comparisons."""
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def _sink_from_choice(area_size: float, choice: str, custom_xy: Tuple[float, float]) -> Tuple[float, float]:
    L = float(area_size)
    if choice == "Center":
        return (L / 2.0, L / 2.0)
    if choice == "Edge (middle)":
        return (L / 2.0, 0.0)
    if choice == "Corner (0,0)":
        return (0.0, 0.0)
    if choice == "Corner (L,0)":
        return (L, 0.0)
    if choice == "Corner (0,L)":
        return (0.0, L)
    if choice == "Corner (L,L)":
        return (L, L)
    return (float(custom_xy[0]), float(custom_xy[1]))


def _ensure_nonempty_ch(net: Network, ch_indices: np.ndarray) -> np.ndarray:
    ch = np.asarray(ch_indices, dtype=int).reshape(-1)
    alive_idx = net.get_alive_indices()
    if alive_idx.size == 0:
        return np.array([], dtype=int)
    if ch.size > 0:
        ch = np.unique(ch[(ch >= 0) & (ch < net.n_nodes)])
        ch = ch[net.alive_mask[ch]]
    if ch.size > 0:
        return ch
    best = int(alive_idx[int(np.argmax(net.residual_energy[alive_idx]))])
    return np.array([best], dtype=int)


def _compute_markers(alive_history: List[int], n0: int, max_rounds: int) -> Tuple[int, int, int]:
    fnd = max_rounds
    hnd = max_rounds
    lnd = max_rounds
    for r, alive in enumerate(alive_history, start=1):
        if fnd == max_rounds and alive < n0:
            fnd = r
        if hnd == max_rounds and alive <= n0 / 2:
            hnd = r
        if alive == 0:
            lnd = r
            break
    return int(fnd), int(hnd), int(lnd)


def _df_to_csv_download(df: pd.DataFrame, filename_prefix: str) -> Tuple[bytes, str]:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{filename_prefix}_{ts}.csv"
    return df.to_csv(index=False).encode("utf-8"), file_name


def _pad_to_length(arr: np.ndarray, length: int, pad_value: float) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.size >= length:
        return arr[:length]
    out = np.empty((length,), dtype=float)
    out[: arr.size] = arr.astype(float)
    out[arr.size :] = float(pad_value)
    return out


# =========================================================
# Sidebar: Scenario & global parameters
# =========================================================

st.sidebar.header("Scenario")

preset = st.sidebar.selectbox(
    "Preset",
    ["Custom", "S1_100 (homogeneous)", "S1_200 (homogeneous)", "S2_100 (heterogeneous)"],
    index=0,
)

# Default values for presets (can still be edited; reset button applies them)
preset_defaults = {
    "S1_100 (homogeneous)": dict(n_nodes=100, area_size=100.0, hetero=False, adv_fraction=0.2, e0=0.01, e_adv=0.05),
    "S1_200 (homogeneous)": dict(n_nodes=200, area_size=100.0, hetero=False, adv_fraction=0.2, e0=0.01, e_adv=0.05),
    "S2_100 (heterogeneous)": dict(n_nodes=100, area_size=100.0, hetero=True, adv_fraction=0.2, e0=0.01, e_adv=0.05),
}

# Apply presets when the selection changes.
if "preset_last" not in st.session_state:
    st.session_state.preset_last = preset
    if preset != "Custom":
        for k, v in preset_defaults[preset].items():
            st.session_state[k] = v
elif preset != st.session_state.preset_last:
    if preset != "Custom":
        for k, v in preset_defaults[preset].items():
            st.session_state[k] = v
    st.session_state.preset_last = preset


n_nodes = st.sidebar.slider("Number of nodes N", 20, 500, int(st.session_state.get("n_nodes", 100)), 10, key="n_nodes")
area_size = st.sidebar.slider("Area size (L)", 50.0, 300.0, float(st.session_state.get("area_size", 100.0)), 10.0, key="area_size")
hetero = st.sidebar.checkbox("Heterogeneous energy", value=bool(st.session_state.get("hetero", False)), key="hetero")
adv_fraction = st.sidebar.slider("Advanced fraction", 0.0, 0.6, float(st.session_state.get("adv_fraction", 0.2)), 0.05, key="adv_fraction")
e0 = st.sidebar.number_input("E0 (normal nodes, J)", min_value=0.0001, max_value=1.0, value=float(st.session_state.get("e0", 0.01)), step=0.001, format="%.4f", key="e0")
e_adv = st.sidebar.number_input("E_adv (advanced nodes, J)", min_value=0.0001, max_value=1.0, value=float(st.session_state.get("e_adv", 0.05)), step=0.001, format="%.4f", key="e_adv")

st.sidebar.subheader("Sink (BS) position")
sink_choice = st.sidebar.selectbox(
    "Sink location",
    ["Center", "Edge (middle)", "Corner (0,0)", "Corner (L,0)", "Corner (0,L)", "Corner (L,L)", "Custom"],
    index=0,
)
sink_x = st.sidebar.number_input("Sink x", min_value=0.0, max_value=float(area_size), value=float(area_size) / 2.0, step=1.0)
sink_y = st.sidebar.number_input("Sink y", min_value=0.0, max_value=float(area_size), value=float(area_size) / 2.0, step=1.0)
sink_pos = _sink_from_choice(area_size, sink_choice, (sink_x, sink_y))

st.sidebar.subheader("Simulation control")
seed = st.sidebar.number_input("Base seed", 0, 10_000_000, 0)
max_rounds_interactive = st.sidebar.slider("Max rounds (interactive)", 10, 2000, 200, 10)

st.sidebar.subheader("Fitness & radio")
rc = st.sidebar.slider("Cluster radius Rc", 5.0, 80.0, 25.0, 1.0)
with st.sidebar.expander("Fitness weights (objective)", expanded=False):
    w1 = st.slider("w1 (energy)", 0.0, 1.0, 0.4, 0.05)
    w2 = st.slider("w2 (distance)", 0.0, 1.0, 0.4, 0.05)
    w3 = st.slider("w3 (load)", 0.0, 1.0, 0.2, 0.05)
    lam = st.slider("λ (radius penalty)", 0.0, 5.0, 1.0, 0.1)

with st.sidebar.expander("Radio parameters", expanded=False):
    rp = RadioParams()
    rp.E_elec = st.number_input("E_elec (J/bit)", min_value=0.0, value=float(rp.E_elec), format="%.2e")
    rp.eps_fs = st.number_input("eps_fs (J/bit/m^2)", min_value=0.0, value=float(rp.eps_fs), format="%.2e")
    rp.eps_mp = st.number_input("eps_mp (J/bit/m^4)", min_value=0.0, value=float(rp.eps_mp), format="%.2e")
    rp.E_da = st.number_input("E_da (J/bit)", min_value=0.0, value=float(rp.E_da), format="%.2e")
    rp.l_data = st.number_input("l_data (bits)", min_value=100, value=int(rp.l_data), step=100)
    rp.l_ctrl = st.number_input("l_ctrl (bits)", min_value=20, value=int(rp.l_ctrl), step=20)
    rp.d0 = st.number_input("d0 (m)", min_value=1.0, value=float(rp.d0), step=1.0)

fit_params = FitnessParams(w1=float(w1), w2=float(w2), w3=float(w3), lam=float(lam), rc=float(rc))

# =========================================================
# Sidebar: Mode & algorithm parameters
# =========================================================

st.sidebar.header("Mode")

mode = st.sidebar.radio("Execution mode", ["Interactive (single algo)", "Compare (multi algo, full runs)"], index=0)

if mode == "Interactive (single algo)":
    algo_single = st.sidebar.selectbox("Algorithm", ALGO_KEYS, format_func=_algo_display_name, index=0)
    algo_selected = [algo_single]
else:
    algo_selected = st.sidebar.multiselect(
        "Algorithms to compare",
        options=ALGO_KEYS,
        default=["FSS", "LEACH", "HEED", "SEP", "Greedy"],
        format_func=_algo_display_name,
    )
    n_runs = st.sidebar.number_input("Runs (paired seeds)", min_value=1, max_value=200, value=10, step=1)
    max_rounds_compare = st.sidebar.slider("Max rounds (comparison)", 10, 5000, 1000, 50)
    stop_when_dead = st.sidebar.checkbox("Stop early when all nodes are dead", value=True)
    show_quantiles = st.sidebar.checkbox("Show quantile bands (25–75%)", value=True)

st.sidebar.subheader("Algorithm hyperparameters")

# Default parameter store
if "algo_params" not in st.session_state:
    st.session_state.algo_params = {}

algo_params: Dict[str, Dict] = st.session_state.algo_params

def _get_algo_params(key: str) -> Dict:
    if key not in algo_params:
        algo_params[key] = {}
    return algo_params[key]

def _param_int(label: str, key: str, v: int, min_v: int, max_v: int, step: int = 1) -> int:
    return int(st.number_input(label, min_value=min_v, max_value=max_v, value=int(v), step=int(step), key=key))

def _param_float(label: str, key: str, v: float, min_v: float, max_v: float, step: float) -> float:
    return float(st.number_input(label, min_value=float(min_v), max_value=float(max_v), value=float(v), step=float(step), key=key))

# Render controls only for selected algorithms to keep sidebar readable
for algo in algo_selected:
    with st.sidebar.expander(f"{_algo_display_name(algo)} params", expanded=False):
        p = _get_algo_params(algo)

        if algo.startswith("FSS"):
            p.setdefault("n_iter", 50)
            p.setdefault("elite_size", 10)
            p.setdefault("tau", 0.6)
            p.setdefault("theta", 0.3)
            p.setdefault("Lmax", 10)
            p.setdefault("use_phase2", True)
            p.setdefault("n_ch", min(10, int(n_nodes)))

            p["n_iter"] = _param_int("n_iter", f"{algo}_n_iter", p["n_iter"], 5, 500, 5)
            p["elite_size"] = _param_int("elite_size", f"{algo}_elite", p["elite_size"], 1, 200, 1)
            p["tau"] = _param_float("tau", f"{algo}_tau", p["tau"], 0.0, 1.0, 0.05)
            p["theta"] = _param_float("theta", f"{algo}_theta", p["theta"], 0.0, 1.0, 0.05)
            p["Lmax"] = _param_int("Lmax (local search)", f"{algo}_lmax", p["Lmax"], 0, 200, 1)
            p["n_ch"] = _param_int("n_ch (target)", f"{algo}_nch", p["n_ch"], 1, int(n_nodes), 1)

            # Apply ablations from algo key
            if algo == "FSS_noPhase2":
                p["use_phase2"] = False
            elif algo == "FSS_noEnergy":
                p["theta"] = 0.0
            elif algo == "FSS_noLS":
                p["Lmax"] = 0

        elif algo == "PSO":
            p.setdefault("n_particles", 30)
            p.setdefault("n_iter", 100)
            p.setdefault("w_inertia", 0.7)
            p.setdefault("c1", 1.5)
            p.setdefault("c2", 1.5)
            p.setdefault("vmax", 4.0)
            p.setdefault("min_ch", 1)
            p.setdefault("max_ch", min(20, int(n_nodes)))

            p["n_particles"] = _param_int("n_particles", f"{algo}_n_particles", p["n_particles"], 5, 200, 1)
            p["n_iter"] = _param_int("n_iter", f"{algo}_n_iter", p["n_iter"], 5, 500, 5)
            p["w_inertia"] = _param_float("w_inertia", f"{algo}_w", p["w_inertia"], 0.0, 1.2, 0.05)
            p["c1"] = _param_float("c1", f"{algo}_c1", p["c1"], 0.0, 3.0, 0.1)
            p["c2"] = _param_float("c2", f"{algo}_c2", p["c2"], 0.0, 3.0, 0.1)
            p["vmax"] = _param_float("vmax", f"{algo}_vmax", p["vmax"], 0.1, 20.0, 0.1)
            p["min_ch"] = _param_int("min_ch", f"{algo}_minch", p["min_ch"], 1, int(n_nodes), 1)
            p["max_ch"] = _param_int("max_ch", f"{algo}_maxch", p["max_ch"], 1, int(n_nodes), 1)

        elif algo == "GWO":
            p.setdefault("n_wolves", 30)
            p.setdefault("n_iter", 100)
            p.setdefault("min_ch", 1)
            p.setdefault("max_ch", min(20, int(n_nodes)))

            p["n_wolves"] = _param_int("n_wolves", f"{algo}_n_wolves", p["n_wolves"], 5, 200, 1)
            p["n_iter"] = _param_int("n_iter", f"{algo}_n_iter", p["n_iter"], 5, 500, 5)
            p["min_ch"] = _param_int("min_ch", f"{algo}_minch", p["min_ch"], 1, int(n_nodes), 1)
            p["max_ch"] = _param_int("max_ch", f"{algo}_maxch", p["max_ch"], 1, int(n_nodes), 1)

        elif algo == "ABC":
            p.setdefault("n_food_sources", 20)
            p.setdefault("n_iter", 100)
            p.setdefault("limit", 10)
            p.setdefault("min_ch", 1)
            p.setdefault("max_ch", min(20, int(n_nodes)))

            p["n_food_sources"] = _param_int("n_food_sources", f"{algo}_n_food", p["n_food_sources"], 5, 200, 1)
            p["n_iter"] = _param_int("n_iter", f"{algo}_n_iter", p["n_iter"], 5, 500, 5)
            p["limit"] = _param_int("limit", f"{algo}_limit", p["limit"], 1, 200, 1)
            p["min_ch"] = _param_int("min_ch", f"{algo}_minch", p["min_ch"], 1, int(n_nodes), 1)
            p["max_ch"] = _param_int("max_ch", f"{algo}_maxch", p["max_ch"], 1, int(n_nodes), 1)

        elif algo == "LEACH":
            p.setdefault("p_opt", 0.05)
            p.setdefault("min_ch", 1)
            p.setdefault("max_ch", None)
            p["p_opt"] = _param_float("p_opt", f"{algo}_popt", p["p_opt"], 0.005, 0.2, 0.005)
            p["min_ch"] = _param_int("min_ch", f"{algo}_minch", p["min_ch"], 1, int(n_nodes), 1)
            max_ch_val = st.text_input("max_ch (blank = no cap)", value="" if p["max_ch"] is None else str(p["max_ch"]), key=f"{algo}_maxch_txt")
            p["max_ch"] = None if max_ch_val.strip() == "" else int(max_ch_val)

        elif algo == "HEED":
            p.setdefault("p_init", 0.05)
            p.setdefault("c_min", 0.02)
            p.setdefault("n_iter", 3)
            p.setdefault("min_ch", 1)
            p.setdefault("max_ch", None)

            p["p_init"] = _param_float("p_init", f"{algo}_pinit", p["p_init"], 0.005, 0.2, 0.005)
            p["c_min"] = _param_float("c_min", f"{algo}_cmin", p["c_min"], 0.0, 0.2, 0.005)
            p["n_iter"] = _param_int("n_iter", f"{algo}_niter", p["n_iter"], 1, 10, 1)
            p["min_ch"] = _param_int("min_ch", f"{algo}_minch", p["min_ch"], 1, int(n_nodes), 1)
            max_ch_val = st.text_input("max_ch (blank = no cap)", value="" if p["max_ch"] is None else str(p["max_ch"]), key=f"{algo}_maxch_txt")
            p["max_ch"] = None if max_ch_val.strip() == "" else int(max_ch_val)

        elif algo == "SEP":
            p.setdefault("p_opt", 0.05)
            p.setdefault("min_ch", 1)
            p.setdefault("max_ch", None)
            p["p_opt"] = _param_float("p_opt", f"{algo}_popt", p["p_opt"], 0.005, 0.2, 0.005)
            p["min_ch"] = _param_int("min_ch", f"{algo}_minch", p["min_ch"], 1, int(n_nodes), 1)
            max_ch_val = st.text_input("max_ch (blank = no cap)", value="" if p["max_ch"] is None else str(p["max_ch"]), key=f"{algo}_maxch_txt")
            p["max_ch"] = None if max_ch_val.strip() == "" else int(max_ch_val)

        elif algo == "Greedy":
            p.setdefault("n_ch", 0)  # 0 => auto
            p.setdefault("w_energy", 0.7)
            p.setdefault("w_sink", 0.3)

            p["n_ch"] = _param_int("n_ch (0 = auto)", f"{algo}_nch", p["n_ch"], 0, int(n_nodes), 1)
            p["w_energy"] = _param_float("w_energy", f"{algo}_w_energy", p["w_energy"], 0.0, 1.0, 0.05)
            p["w_sink"] = _param_float("w_sink", f"{algo}_w_sink", p["w_sink"], 0.0, 1.0, 0.05)


# =========================================================
# Network lifecycle in session_state (interactive mode)
# =========================================================

def _scenario_config_dict() -> Dict:
    return {
        "preset": preset,
        "n_nodes": int(n_nodes),
        "area_size": float(area_size),
        "hetero": bool(hetero),
        "adv_fraction": float(adv_fraction),
        "e0": float(e0),
        "e_adv": float(e_adv),
        "sink_pos": (float(sink_pos[0]), float(sink_pos[1])),
    }

def _create_network(seed_value: int) -> Network:
    cfg = _scenario_config_dict()
    net = Network.random_network(
        n_nodes=cfg["n_nodes"],
        area_size=cfg["area_size"],
        sink_pos=cfg["sink_pos"],
        heterogenous=cfg["hetero"],
        adv_fraction=cfg["adv_fraction"],
        e0=cfg["e0"],
        e_adv=cfg["e_adv"],
        seed=int(seed_value),
    )
    return net

def _init_interactive_state(force: bool = False) -> None:
    cfg = _scenario_config_dict()
    cfg_hash = _hashable_config(cfg)
    if (not force) and st.session_state.get("scenario_hash") == cfg_hash and st.session_state.get("net") is not None:
        return

    net = _create_network(int(seed))

    st.session_state.net = net
    st.session_state.net_initial = net.initial_energy.copy()
    st.session_state.round = 0
    st.session_state.alive_history = []
    st.session_state.energy_history = []
    st.session_state.pkts_history = []
    st.session_state.throughput = 0
    st.session_state.n_initial = int(n_nodes)
    st.session_state.FND = None
    st.session_state.HND = None
    st.session_state.LND = None
    st.session_state.last_ch = np.array([], dtype=int)
    st.session_state.last_assignments = None
    st.session_state.scenario_hash = cfg_hash

    # protocol states (persist across rounds)
    st.session_state.leach_state = LEACHState.initialize(net.n_nodes)
    st.session_state.sep_state = SEPState.initialize(net.n_nodes)

    # remember current algo to detect changes that require protocol re-init
    st.session_state.interactive_algo = None


def _make_algo_objects(algo: str, seed_base: int, round_idx: int, net: Network) -> Tuple[str, object, Optional[object]]:
    """Return (algo, params, state) to pass to run_* functions."""
    # All metaheuristics: seed varies by round for fairness (as in runner).
    # Protocols keep a persistent state (LEACH/SEP).
    p = _get_algo_params(algo)

    if algo.startswith("FSS"):
        params = FSSParams(
            seed=int(seed_base) + int(round_idx),
            n_iter=int(p.get("n_iter", 50)),
            elite_size=int(p.get("elite_size", 10)),
            tau=float(p.get("tau", 0.6)),
            theta=float(p.get("theta", 0.3)),
            Lmax=int(p.get("Lmax", 10)),
            use_phase2=bool(p.get("use_phase2", True)),
            n_ch=int(p.get("n_ch", min(10, net.n_nodes))),
        )
        return algo, params, None

    if algo == "PSO":
        params = PSOParams(
            seed=int(seed_base) + int(round_idx),
            n_particles=int(p.get("n_particles", 30)),
            n_iter=int(p.get("n_iter", 100)),
            w_inertia=float(p.get("w_inertia", 0.7)),
            c1=float(p.get("c1", 1.5)),
            c2=float(p.get("c2", 1.5)),
            vmax=float(p.get("vmax", 4.0)),
            min_ch=int(p.get("min_ch", 1)),
            max_ch=int(p.get("max_ch", min(20, net.n_nodes))),
        )
        return algo, params, None

    if algo == "GWO":
        params = GWOParams(
            seed=int(seed_base) + int(round_idx),
            n_wolves=int(p.get("n_wolves", 30)),
            n_iter=int(p.get("n_iter", 100)),
            min_ch=int(p.get("min_ch", 1)),
            max_ch=int(p.get("max_ch", min(20, net.n_nodes))),
        )
        return algo, params, None

    if algo == "ABC":
        params = ABCParams(
            seed=int(seed_base) + int(round_idx),
            n_food_sources=int(p.get("n_food_sources", 20)),
            n_iter=int(p.get("n_iter", 100)),
            limit=int(p.get("limit", 10)),
            min_ch=int(p.get("min_ch", 1)),
            max_ch=int(p.get("max_ch", min(20, net.n_nodes))),
        )
        return algo, params, None

    if algo == "LEACH":
        params = LEACHParams(
            seed=int(seed_base),
            p_opt=float(p.get("p_opt", 0.05)),
            min_ch=int(p.get("min_ch", 1)),
            max_ch=p.get("max_ch", None),
        )
        return algo, params, st.session_state.get("leach_state", None)

    if algo == "HEED":
        params = HEEDParams(
            seed=int(seed_base),
            p_init=float(p.get("p_init", 0.05)),
            c_min=float(p.get("c_min", 0.02)),
            n_iter=int(p.get("n_iter", 3)),
            min_ch=int(p.get("min_ch", 1)),
            max_ch=p.get("max_ch", None),
        )
        return algo, params, None

    if algo == "SEP":
        params = SEPParams(
            seed=int(seed_base),
            p_opt=float(p.get("p_opt", 0.05)),
            min_ch=int(p.get("min_ch", 1)),
            max_ch=p.get("max_ch", None),
            e0=float(e0),
            e_adv=float(e_adv),
            adv_fraction=float(adv_fraction),
        )
        return algo, params, st.session_state.get("sep_state", None)

    if algo == "Greedy":
        params = GreedyParams(
            seed=int(seed_base),
            n_ch=int(p.get("n_ch", 0)),
            w_energy=float(p.get("w_energy", 0.7)),
            w_sink=float(p.get("w_sink", 0.3)),
        )
        return algo, params, None

    raise ValueError(f"Unknown algorithm: {algo}")


def _choose_ch(net: Network, algo: str, round_idx: int) -> np.ndarray:
    algo, params, state = _make_algo_objects(algo, int(seed), int(round_idx), net)

    if algo.startswith("FSS"):
        res = run_fss_wsn(net, fit_params, params)  # type: ignore[arg-type]
        return res.best_ch_indices

    if algo == "PSO":
        res = run_pso_wsn(net, fit_params, params)  # type: ignore[arg-type]
        return res.best_ch_indices

    if algo == "GWO":
        res = run_gwo_wsn(net, fit_params, params)  # type: ignore[arg-type]
        return res.best_ch_indices

    if algo == "ABC":
        res = run_abc_wsn(net, fit_params, params)  # type: ignore[arg-type]
        return res.best_ch_indices

    if algo == "LEACH":
        res = run_leach_wsn(net, fit_params, params, round_idx=round_idx, state=state)  # type: ignore[arg-type]
        return res.best_ch_indices

    if algo == "HEED":
        res = run_heed_wsn(net, fit_params, params, round_idx=round_idx)  # type: ignore[arg-type]
        return res.best_ch_indices

    if algo == "SEP":
        res = run_sep_wsn(net, fit_params, params, round_idx=round_idx, state=state)  # type: ignore[arg-type]
        return res.best_ch_indices

    if algo == "Greedy":
        res = run_greedy_wsn(net, fit_params, params, round_idx=round_idx)  # type: ignore[arg-type]
        return res.best_ch_indices

    raise ValueError(f"Unknown algorithm: {algo}")


def _interactive_one_round(algo: str) -> None:
    net: Network = st.session_state.net
    if net is None:
        return

    # If switching algorithm mid-session, reset protocol states (LEACH/SEP epochs).
    if st.session_state.get("interactive_algo") != algo:
        st.session_state.leach_state = LEACHState.initialize(net.n_nodes)
        st.session_state.sep_state = SEPState.initialize(net.n_nodes)
        st.session_state.interactive_algo = algo

    r = int(st.session_state.round)
    ch = _choose_ch(net, algo, r)
    ch = _ensure_nonempty_ch(net, ch)
    st.session_state.last_ch = ch

    assignments, _, _ = net.assign_clusters(ch, rc=float(fit_params.rc))
    st.session_state.last_assignments = assignments
    stats = apply_round_energy(net, ch, assignments, rp)

    st.session_state.round += 1
    st.session_state.alive_history.append(int(stats["alive"]))
    st.session_state.energy_history.append(float(stats["total_energy"]))
    st.session_state.pkts_history.append(int(stats["pkts_to_sink"]))
    st.session_state.throughput += int(stats["pkts_to_sink"])

    alive = int(stats["alive"])
    n0 = int(st.session_state.n_initial)
    if st.session_state.FND is None and alive < n0:
        st.session_state.FND = int(st.session_state.round)
    if st.session_state.HND is None and alive <= n0 / 2:
        st.session_state.HND = int(st.session_state.round)
    if st.session_state.LND is None and alive == 0:
        st.session_state.LND = int(st.session_state.round)


# =========================================================
# Comparison mode: full runs (multi algo)
# =========================================================

def _simulate_one(
    net_init: Network,
    algo: str,
    seed_value: int,
    max_rounds: int,
    stop_when_dead: bool,
    fit_params_local: FitnessParams,
    radio_local: RadioParams,
    algo_params_store: Dict[str, Dict],
    scenario_energy: Dict[str, float],
) -> Tuple[Dict, pd.DataFrame]:
    """Run a full lifetime simulation on a fresh copy of the initial network.

    This function is deliberately self-contained (no dependency on module globals),
    so it can be used safely inside Streamlit caches.
    """
    net = Network(
        positions=net_init.get_positions_array().copy(),
        sink_position=net_init.sink_position(),
        area_size=float(net_init.area_size),
        initial_energy=net_init.initial_energy.copy(),
        rng=np.random.default_rng(int(seed_value)),
    )

    # protocol states for this simulation
    leach_state = LEACHState.initialize(net.n_nodes)
    sep_state = SEPState.initialize(net.n_nodes)

    alive_hist: List[int] = []
    energy_hist: List[float] = []
    pkts_hist: List[int] = []
    throughput_cum = 0

    nfe_total = 0.0
    cpu_opt_total = 0.0

    p = algo_params_store.get(algo, {})

    for r in range(int(max_rounds)):
        if stop_when_dead and net.get_alive_indices().size == 0:
            break

        t0 = time.perf_counter()

        if algo.startswith("FSS"):
            fss_params = FSSParams(
                seed=int(seed_value) + int(r),
                n_iter=int(p.get("n_iter", 50)),
                elite_size=int(p.get("elite_size", 10)),
                tau=float(p.get("tau", 0.6)),
                theta=float(p.get("theta", 0.3)),
                Lmax=int(p.get("Lmax", 10)),
                use_phase2=bool(p.get("use_phase2", True)),
                n_ch=int(p.get("n_ch", min(10, net.n_nodes))),
            )
            if algo == "FSS_noPhase2":
                fss_params.use_phase2 = False
            elif algo == "FSS_noEnergy":
                fss_params.theta = 0.0
            elif algo == "FSS_noLS":
                fss_params.Lmax = 0
            res = run_fss_wsn(net, fit_params_local, fss_params)
            nfe_total += float(fss_params.n_iter)

        elif algo == "PSO":
            params = PSOParams(
                seed=int(seed_value) + int(r),
                n_particles=int(p.get("n_particles", 30)),
                n_iter=int(p.get("n_iter", 100)),
                w_inertia=float(p.get("w_inertia", 0.7)),
                c1=float(p.get("c1", 1.5)),
                c2=float(p.get("c2", 1.5)),
                vmax=float(p.get("vmax", 4.0)),
                min_ch=int(p.get("min_ch", 1)),
                max_ch=int(p.get("max_ch", min(20, net.n_nodes))),
            )
            res = run_pso_wsn(net, fit_params_local, params)
            nfe_total += float(params.n_iter * params.n_particles)

        elif algo == "GWO":
            params = GWOParams(
                seed=int(seed_value) + int(r),
                n_wolves=int(p.get("n_wolves", 30)),
                n_iter=int(p.get("n_iter", 100)),
                min_ch=int(p.get("min_ch", 1)),
                max_ch=int(p.get("max_ch", min(20, net.n_nodes))),
            )
            res = run_gwo_wsn(net, fit_params_local, params)
            nfe_total += float(params.n_iter * params.n_wolves)

        elif algo == "ABC":
            params = ABCParams(
                seed=int(seed_value) + int(r),
                n_food_sources=int(p.get("n_food_sources", 20)),
                n_iter=int(p.get("n_iter", 100)),
                limit=int(p.get("limit", 10)),
                min_ch=int(p.get("min_ch", 1)),
                max_ch=int(p.get("max_ch", min(20, net.n_nodes))),
            )
            res = run_abc_wsn(net, fit_params_local, params)
            nfe_total += float(params.n_iter * params.n_food_sources)

        elif algo == "LEACH":
            params = LEACHParams(
                seed=int(seed_value),
                p_opt=float(p.get("p_opt", 0.05)),
                min_ch=int(p.get("min_ch", 1)),
                max_ch=p.get("max_ch", None),
            )
            res = run_leach_wsn(net, fit_params_local, params, round_idx=r, state=leach_state)

        elif algo == "HEED":
            params = HEEDParams(
                seed=int(seed_value),
                p_init=float(p.get("p_init", 0.05)),
                c_min=float(p.get("c_min", 0.02)),
                n_iter=int(p.get("n_iter", 3)),
                min_ch=int(p.get("min_ch", 1)),
                max_ch=p.get("max_ch", None),
            )
            res = run_heed_wsn(net, fit_params_local, params, round_idx=r)

        elif algo == "SEP":
            params = SEPParams(
                seed=int(seed_value),
                p_opt=float(p.get("p_opt", 0.05)),
                min_ch=int(p.get("min_ch", 1)),
                max_ch=p.get("max_ch", None),
                e0=float(scenario_energy["e0"]),
                e_adv=float(scenario_energy["e_adv"]),
                adv_fraction=float(scenario_energy["adv_fraction"]),
            )
            res = run_sep_wsn(net, fit_params_local, params, round_idx=r, state=sep_state)

        elif algo == "Greedy":
            params = GreedyParams(
                seed=int(seed_value),
                n_ch=int(p.get("n_ch", 0)),
                w_energy=float(p.get("w_energy", 0.7)),
                w_sink=float(p.get("w_sink", 0.3)),
            )
            res = run_greedy_wsn(net, fit_params_local, params, round_idx=r)

        else:
            raise ValueError(f"Unknown algorithm: {algo}")

        cpu_opt_total += float(time.perf_counter() - t0)

        ch = _ensure_nonempty_ch(net, res.best_ch_indices)
        assignments, _, _ = net.assign_clusters(ch, rc=float(fit_params_local.rc))
        stats = apply_round_energy(net, ch, assignments, radio_local)

        alive_hist.append(int(stats["alive"]))
        energy_hist.append(float(stats["total_energy"]))
        pkts_hist.append(int(stats["pkts_to_sink"]))
        throughput_cum += int(stats["pkts_to_sink"])

        if stop_when_dead and int(stats["alive"]) == 0:
            break

    n0 = int(net.n_nodes)
    fnd, hnd, lnd = _compute_markers(alive_hist, n0=n0, max_rounds=int(max_rounds))

    summary = {
        "algo": algo,
        "seed": int(seed_value),
        "n_nodes": int(net.n_nodes),
        "area_size": float(net.area_size),
        "hetero": bool(hetero),
        "adv_fraction": float(scenario_energy["adv_fraction"]),
        "e0": float(scenario_energy["e0"]),
        "e_adv": float(scenario_energy["e_adv"]),
        "sink_x": float(net.sink_position()[0]),
        "sink_y": float(net.sink_position()[1]),
        "rc": float(fit_params_local.rc),
        "FND": int(fnd),
        "HND": int(hnd),
        "LND": int(lnd),
        "throughput": int(throughput_cum),
        "cpu_opt_s": float(cpu_opt_total),
        "nfe": float(nfe_total),
    }

    ts = pd.DataFrame(
        {
            "round": np.arange(1, len(alive_hist) + 1, dtype=int),
            "algo": algo,
            "seed": int(seed_value),
            "alive": np.asarray(alive_hist, dtype=int),
            "total_energy": np.asarray(energy_hist, dtype=float),
            "pkts_to_sink": np.asarray(pkts_hist, dtype=int),
        }
    )
    ts["throughput_cum"] = ts["pkts_to_sink"].cumsum()
    return summary, ts


@st.cache_data(show_spinner=False)
def _run_compare_cached(config_key: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cached comparison run (config_key is a stable JSON string)."""
    cfg = json.loads(config_key)

    net_init = Network.random_network(
        n_nodes=int(cfg["scenario"]["n_nodes"]),
        area_size=float(cfg["scenario"]["area_size"]),
        sink_pos=tuple(cfg["scenario"]["sink_pos"]),
        heterogenous=bool(cfg["scenario"]["hetero"]),
        adv_fraction=float(cfg["scenario"]["adv_fraction"]),
        e0=float(cfg["scenario"]["e0"]),
        e_adv=float(cfg["scenario"]["e_adv"]),
        seed=int(cfg["scenario"]["seed"]),
    )

    fit_params_local = FitnessParams(
        w1=float(cfg["fitness"]["w1"]),
        w2=float(cfg["fitness"]["w2"]),
        w3=float(cfg["fitness"]["w3"]),
        lam=float(cfg["fitness"]["lam"]),
        rc=float(cfg["fitness"]["rc"]),
    )
    radio_local = RadioParams(**cfg["radio"])

    algo_params_store = cfg["algo_params"]
    scenario_energy = {
        "e0": float(cfg["scenario"]["e0"]),
        "e_adv": float(cfg["scenario"]["e_adv"]),
        "adv_fraction": float(cfg["scenario"]["adv_fraction"]),
    }

    summaries: List[Dict] = []
    timeseries: List[pd.DataFrame] = []

    algos = list(cfg["algos"])
    seeds = [int(cfg["scenario"]["seed"]) + int(i) for i in range(int(cfg["n_runs"]))]

    for s in seeds:
        for algo in algos:
            summ, ts = _simulate_one(
                net_init,
                algo=algo,
                seed_value=s,
                max_rounds=int(cfg["max_rounds"]),
                stop_when_dead=bool(cfg["stop_when_dead"]),
                fit_params_local=fit_params_local,
                radio_local=radio_local,
                algo_params_store=algo_params_store,
                scenario_energy=scenario_energy,
            )
            summaries.append(summ)
            timeseries.append(ts)

    df_sum = pd.DataFrame(summaries)
    df_ts = pd.concat(timeseries, ignore_index=True) if timeseries else pd.DataFrame()
    return df_sum, df_ts


# =========================================================
# Main layout

# =========================================================

tabs = st.tabs(["Interactive", "Compare", "Export / Reproducibility"])

# ---------------------------------------------------------
# Interactive tab
# ---------------------------------------------------------

with tabs[0]:
    st.subheader("Interactive (single algorithm)")

    if mode != "Interactive (single algo)":
        st.info("Passez en mode 'Interactive' dans la sidebar pour piloter round par round.")
    else:
        _init_interactive_state(force=False)
        net: Network = st.session_state.net

        colA, colB, colC = st.columns([1, 1, 1])

        with colA:
            if st.button("Rebuild network (apply scenario)", type="primary"):
                _init_interactive_state(force=True)

        with colB:
            if st.button("Reset energies (same topology)"):
                net.reset_energies()
                st.session_state.round = 0
                st.session_state.alive_history = []
                st.session_state.energy_history = []
                st.session_state.pkts_history = []
                st.session_state.throughput = 0
                st.session_state.FND = None
                st.session_state.HND = None
                st.session_state.LND = None
                st.session_state.last_ch = np.array([], dtype=int)
                st.session_state.last_assignments = None
                st.session_state.leach_state = LEACHState.initialize(net.n_nodes)
                st.session_state.sep_state = SEPState.initialize(net.n_nodes)

        with colC:
            if st.button("Advance 1 round"):
                if net.get_alive_indices().size > 0:
                    _interactive_one_round(algo_single)
                else:
                    st.warning("All nodes are dead. Reset to restart.")

        run_all = st.button("Run to max rounds", help="Runs until max rounds or until all nodes are dead.", use_container_width=True)

        if run_all:
            if net.get_alive_indices().size == 0:
                st.warning("All nodes are dead. Reset to restart.")
            else:
                progress_bar = st.progress(0.0, text="Simulation in progress...")
                status_text = st.empty()
                start_time = time.time()
                target = int(max_rounds_interactive)

                for i in range(target):
                    if net.get_alive_indices().size == 0:
                        break
                    _interactive_one_round(algo_single)

                    frac = min(max((i + 1) / float(target), 0.0), 1.0)
                    elapsed = time.time() - start_time
                    eta = elapsed * (1.0 - frac) / max(frac, 1e-12)
                    eta_int = int(eta)
                    eta_min = eta_int // 60
                    eta_sec = eta_int % 60
                    eta_str = f"{eta_min:02d}:{eta_sec:02d}"
                    current_round = int(st.session_state.round)
                    progress_bar.progress(frac, text=f"Round {current_round}/{target} | ETA ~ {eta_str}")
                    status_text.markdown(f"**Round:** {current_round} | ETA ~ {eta_str}")

                progress_bar.progress(1.0, text="Done")
                status_text.markdown(f"Simulation done. Last round: {st.session_state.round}.")

        # --- display ---
        left, right = st.columns([1.1, 1.0])

        with left:
            st.markdown(f"**Algorithm:** {_algo_display_name(algo_single)}")
            alive_now = int(net.get_alive_indices().size)
            st.markdown(f"**Alive:** {alive_now} / {int(st.session_state.n_initial)}")

            coords = net.get_positions_array()
            xs = coords[:, 0]
            ys = coords[:, 1]

            energies = net.residual_energy.copy()
            max_e = float(np.max(energies)) if energies.size > 0 else 1.0
            norm_e = energies / max(max_e, 1e-12)

            ch_vis = np.asarray(st.session_state.last_ch, dtype=int)
            is_ch = np.zeros(net.n_nodes, dtype=bool)
            if ch_vis.size > 0:
                is_ch[ch_vis] = True

            fig = go.Figure()

            # Non-CH nodes
            fig.add_trace(
                go.Scatter(
                    x=xs[~is_ch],
                    y=ys[~is_ch],
                    mode="markers",
                    name="Nodes",
                    marker=dict(
                        size=8,
                        color=norm_e[~is_ch],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Energy (norm)"),
                    ),
                    text=[f"i={i}<br>E={energies[i]:.4f} J" for i in range(net.n_nodes) if not is_ch[i]],
                    hoverinfo="text",
                )
            )

            # CH nodes
            if ch_vis.size > 0:
                fig.add_trace(
                    go.Scatter(
                        x=xs[is_ch],
                        y=ys[is_ch],
                        mode="markers",
                        name="Cluster-Heads",
                        marker=dict(size=12, symbol="diamond", color="red", line=dict(width=1, color="black")),
                        text=[f"CH i={i}<br>E={energies[i]:.4f} J" for i in ch_vis],
                        hoverinfo="text",
                    )
                )

            # Sink
            sx, sy = net.sink_position()
            fig.add_trace(
                go.Scatter(
                    x=[sx],
                    y=[sy],
                    mode="markers",
                    name="Sink",
                    marker=dict(size=14, symbol="x", color="black"),
                )
            )

            fig.update_layout(
                xaxis=dict(range=[0, area_size]),
                yaxis=dict(range=[0, area_size]),
                height=620,
                title="Topology (nodes, CHs, sink)",
                legend=dict(orientation="h"),
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show current fitness details for last CH set
            if ch_vis.size > 0 and net.get_alive_indices().size > 0:
                f_val, details = fitness(net, ch_vis, fit_params)
                with st.expander("Fitness details (current round)", expanded=False):
                    st.write({"F": float(f_val), **{k: float(v) for k, v in details.items()}})

        with right:
            st.markdown("### Metrics")

            rounds = np.arange(len(st.session_state.alive_history))
            if len(rounds) == 0:
                st.info("Run at least one round to see curves.")
            else:
                import matplotlib.pyplot as plt

                fig2, ax2 = plt.subplots()
                ax2.plot(rounds + 1, st.session_state.alive_history, label="Alive")
                ax2.set_xlabel("Round")
                ax2.set_ylabel("Alive nodes")
                ax2.legend()
                ax2.text(
                    0.99,
                    0.99,
                    direction_phrase("alive"),
                    transform=ax2.transAxes,
                    ha="right",
                    va="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.9),
                )
                st.pyplot(fig2, use_container_width=True)

                fig3, ax3 = plt.subplots()
                ax3.plot(rounds + 1, st.session_state.energy_history, label="Total residual energy")
                ax3.set_xlabel("Round")
                ax3.set_ylabel("Energy (J)")
                ax3.legend()
                ax3.text(
                    0.99,
                    0.99,
                    direction_phrase("total_energy"),
                    transform=ax3.transAxes,
                    ha="right",
                    va="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.9),
                )
                st.pyplot(fig3, use_container_width=True)

                throughput_total = int(st.session_state.throughput)
                st.markdown(f"**Throughput total (pkts → BS):** {throughput_total}")

                fnd = st.session_state.FND if st.session_state.FND is not None else "–"
                hnd = st.session_state.HND if st.session_state.HND is not None else "–"
                lnd = st.session_state.LND if st.session_state.LND is not None else "–"
                st.markdown(f"**FND:** {fnd}")
                st.markdown(f"**HND:** {hnd}")
                st.markdown(f"**LND:** {lnd}")

                st.markdown(
                    f"**E_min:** {float(net.residual_energy.min()):.4f} J  |  "
                    f"**E_mean:** {float(net.residual_energy.mean()):.4f} J"
                )

# ---------------------------------------------------------
# Compare tab
# ---------------------------------------------------------

with tabs[1]:
    st.subheader("Compare algorithms (paired seeds, full runs)")

    if mode != "Compare (multi algo, full runs)":
        st.info("Passez en mode 'Compare' dans la sidebar pour exécuter des runs complets multi-algos.")
    elif len(algo_selected) == 0:
        st.warning("Select at least one algorithm.")
    else:
        cfg = {
            "scenario": {
                **_scenario_config_dict(),
                "seed": int(seed),
            },
            "fitness": {"w1": float(w1), "w2": float(w2), "w3": float(w3), "lam": float(lam), "rc": float(rc)},
            "radio": asdict(rp),
            "algos": list(algo_selected),
            "algo_params": algo_params,
            "n_runs": int(n_runs),
            "max_rounds": int(max_rounds_compare),
            "stop_when_dead": bool(stop_when_dead),
        }
        cfg_key = _hashable_config(cfg)

        run_compare = st.button("Run comparison", type="primary", use_container_width=True)
        if run_compare:
            with st.spinner("Running simulations..."):
                df_sum, df_ts = _run_compare_cached(cfg_key)
            st.session_state.compare_summary = df_sum
            st.session_state.compare_timeseries = df_ts
            st.success(f"Done. Rows: summary={len(df_sum)}, timeseries={len(df_ts)}")

        df_sum = st.session_state.get("compare_summary")
        df_ts = st.session_state.get("compare_timeseries")

        if isinstance(df_sum, pd.DataFrame) and not df_sum.empty:
            st.markdown("### Summary (per seed)")
            st.dataframe(
                df_sum.sort_values(["algo", "seed"])[
                    ["algo", "seed", "FND", "HND", "LND", "throughput", "cpu_opt_s", "nfe"]
                ],
                use_container_width=True,
            )

            st.markdown("### Aggregated (mean ± std)")
            agg = df_sum.groupby("algo").agg(
                runs=("seed", "count"),
                FND_mean=("FND", "mean"),
                FND_std=("FND", "std"),
                HND_mean=("HND", "mean"),
                HND_std=("HND", "std"),
                LND_mean=("LND", "mean"),
                LND_std=("LND", "std"),
                thr_mean=("throughput", "mean"),
                thr_std=("throughput", "std"),
                cpu_opt_s_mean=("cpu_opt_s", "mean"),
            ).reset_index()
            st.dataframe(agg.sort_values("FND_mean", ascending=False), use_container_width=True)

            if isinstance(df_ts, pd.DataFrame) and not df_ts.empty:
                st.markdown("### Curves (mean across seeds)")

                metric = st.selectbox(
                    "Metric",
                    ["alive", "total_energy", "throughput_cum"],
                    index=0,
                    format_func=metric_display_name,
                )

                max_r = int(max_rounds_compare)
                fig = go.Figure()

                # Metric direction hint for intuitive reading
                metric_hint = direction_phrase(metric)

                # Keep central method first in legend
                algo_order = sorted(df_ts["algo"].unique(), key=lambda a: (0 if is_central_algo(a) else 1, str(a)))

                for algo in algo_order:
                    sub = df_ts[df_ts["algo"] == algo]
                    # Build matrix seeds x rounds
                    series = []
                    for s in sorted(sub["seed"].unique()):
                        ss = sub[sub["seed"] == s].sort_values("round")[metric].to_numpy()
                        pad_val = 0.0 if metric != "total_energy" else float(ss[-1]) if ss.size > 0 else 0.0
                        series.append(_pad_to_length(ss, max_r, pad_val))
                    mat = np.vstack(series) if series else np.zeros((0, max_r))
                    if mat.size == 0:
                        continue
                    mean = mat.mean(axis=0)
                    q25 = np.quantile(mat, 0.25, axis=0)
                    q75 = np.quantile(mat, 0.75, axis=0)

                    x = np.arange(1, max_r + 1, dtype=int)
                    color = algo_color(algo)
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=mean,
                            mode="lines",
                            name=_algo_display_name(algo),
                            line=dict(
                                color=color,
                                width=(4 if is_central_algo(algo) else 2),
                                dash=("solid" if is_central_algo(algo) else "dash"),
                            ),
                        )
                    )

                    if show_quantiles and mat.shape[0] >= 3:
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=q75,
                                mode="lines",
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=q25,
                                mode="lines",
                                line=dict(width=0),
                                fill="tonexty",
                                showlegend=False,
                                hoverinfo="skip",
                                fillcolor=hex_to_rgba(color, 0.18),
                            )
                        )

                fig.update_layout(
                    height=520,
                    xaxis_title="Round",
                    yaxis_title=metric_display_name(metric),
                    legend=dict(orientation="h"),
                    annotations=[
                        dict(
                            text=metric_hint,
                            x=1.0,
                            y=1.08,
                            xref="paper",
                            yref="paper",
                            xanchor="right",
                            yanchor="top",
                            showarrow=False,
                            font=dict(size=12),
                        )
                    ],
                )
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Export tab
# ---------------------------------------------------------

with tabs[2]:
    st.subheader("Export / Reproducibility")

    st.markdown("This tab is intended to help you export results in the same spirit as the batch runner.")

    # Export interactive session
    if mode == "Interactive (single algo)" and st.session_state.get("net") is not None:
        st.markdown("### Export current interactive session")
        rounds = np.arange(len(st.session_state.alive_history)) + 1
        if rounds.size == 0:
            st.info("Run at least one round to enable export.")
        else:
            df_session = pd.DataFrame(
                {
                    "round": rounds.astype(int),
                    "algo": algo_single,
                    "seed": int(seed),
                    "alive": st.session_state.alive_history,
                    "total_energy": st.session_state.energy_history,
                    "pkts_to_sink": st.session_state.pkts_history,
                }
            )
            df_session["throughput_cum"] = df_session["pkts_to_sink"].cumsum()

            csv_bytes, file_name = _df_to_csv_download(df_session, f"interactive_{algo_single}")
            st.download_button(
                label="Download timeseries (CSV)",
                data=csv_bytes,
                file_name=file_name,
                mime="text/csv",
            )

    # Export comparison results
    df_sum = st.session_state.get("compare_summary")
    df_ts = st.session_state.get("compare_timeseries")

    st.markdown("### Export comparison (summary + timeseries)")
    if isinstance(df_sum, pd.DataFrame) and not df_sum.empty:
        b1, fn1 = _df_to_csv_download(df_sum, "compare_summary")
        st.download_button("Download compare_summary.csv", data=b1, file_name=fn1, mime="text/csv")
    else:
        st.info("Run a comparison first to export the summary.")

    if isinstance(df_ts, pd.DataFrame) and not df_ts.empty:
        b2, fn2 = _df_to_csv_download(df_ts, "compare_timeseries")
        st.download_button("Download compare_timeseries.csv", data=b2, file_name=fn2, mime="text/csv")
    else:
        st.info("Run a comparison first to export the timeseries.")

    st.markdown("### Configuration snapshot")
    cfg_snapshot = {
        "scenario": _scenario_config_dict(),
        "fitness": asdict(fit_params),
        "radio": asdict(rp),
        "algo_params": algo_params,
        "seed": int(seed),
        "mode": mode,
    }
    st.code(json.dumps(cfg_snapshot, indent=2, sort_keys=True), language="json")
