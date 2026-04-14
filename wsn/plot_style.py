from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


# Central method (proposed)
CENTRAL_ALGO_KEYS = {"FSS", "FSS-WSN", "FSS_WSN"}


# Plotly/Matplotlib-friendly color palette (distinct, colorblind-aware-ish)
# (D3 category10-like)
_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


# Fixed mapping for known algorithms in this repo (keeps colors stable across runs)
ALGO_COLOR_MAP: Dict[str, str] = {
    "FSS": "#d62728",  # red (central)
    "PSO": "#1f77b4",
    "GWO": "#2ca02c",
    "ABC": "#ff7f0e",
    "SO": "#17becf",
    "GJO": "#bcbd22",
    "EMOGJO": "#e377c2",
    "ESOGJO": "#9edae5",
    "LEACH": "#9467bd",
    "HEED": "#8c564b",
    "SEP": "#2f4b7c",
    "Greedy": "#7f7f7f",
    # ablations (kept distinct but treated as baselines visually)
    "FSS_noPhase2": "#bcbd22",
    "FSS_noEnergy": "#e377c2",
    "FSS_noLS": "#9edae5",
    "FSS_noRepairReg": "#9edae5",
}


def normalize_algo_key(algo: str) -> str:
    return str(algo).strip()


def is_central_algo(algo: str) -> bool:
    key = normalize_algo_key(algo)
    return key in CENTRAL_ALGO_KEYS


def display_algo_name(algo: str) -> str:
    key = normalize_algo_key(algo)
    if key == "FSS":
        return "FSS-WSN"
    return key


def algo_color(algo: str) -> str:
    key = normalize_algo_key(algo)
    if key in ALGO_COLOR_MAP:
        return ALGO_COLOR_MAP[key]

    # Stable fallback: hash into palette
    idx = (abs(hash(key)) % len(_PALETTE)) if _PALETTE else 0
    return _PALETTE[idx]


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = str(hex_color).lstrip("#")
    if len(h) != 6:
        return f"rgba(0,0,0,{float(alpha)})"
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{float(alpha)})"


@dataclass(frozen=True)
class MetricDirection:
    direction: str  # "higher" or "lower"
    assumed: bool
    note: str = ""


def _norm_metric_name(metric: str) -> str:
    """Normalize metric names across CSV/Excel/UI.

    Examples:
    - "CPU time per round" -> "cpu_time_per_round"
    - "Cumulative CH→BS traffic (pkts)" -> "cumulative_ch_bs_traffic_pkts"
    - "pkts_to_sink_r" -> "pkts_to_sink_r"
    """
    m = str(metric).strip().lower()
    m = (
        m.replace("→", "_")
        .replace("->", "_")
        .replace("/", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace("{", "_")
        .replace("}", "_")
        .replace("%", "pct")
        .replace("±", "pm")
    )
    # collapse any non-alnum/underscore to underscore
    out = []
    prev_us = False
    for ch in m:
        is_ok = ch.isalnum() or ch == "_"
        if is_ok:
            out.append(ch)
            prev_us = (ch == "_")
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    norm = "".join(out).strip("_")
    while "__" in norm:
        norm = norm.replace("__", "_")
    return norm


# Explicit metric directions (user-provided + repo keys)
_HIGHER_IS_BETTER = {
    # lifetime
    "fnd",
    "hnd",
    "lnd",
    "rlast",
    "r_last",
    "last_round",
    "last_useful_round",
    # throughput / delivered reports
    "throughput",
    "throughput_reports",
    "throughput_cum",
    "cumulative_delivered_reports",
    "cumulative_delivered_reports_reports",
    "cumulative_delivered_reports_sum",
    "delivered_reports",
    "delivered_reports_r",
    "delivered_reports_cum",
    "reports",
    "reports_cum",
    # CH->BS traffic
    "cumulative_ch_bs_traffic_pkts",
    "ch_bs_traffic_pkts",
    "pkts_to_sink",
    "pkts_to_sink_r",
    "pkts_to_sink_cum",
    # Multi-hop fairness / quality (diagnostics)
    "mh_jain_q",
    # energy
    "total_energy",
    "total_residual_energy",
    "total_residual_energy_over_time",
    # sometimes plotted as a curve metric
    "alive",
}

_LOWER_IS_BETTER = {
    # compute cost
    "cpu_opt_s",
    "cpu_time",
    "cpu_time_s",
    "cpu_time_per_round",
    "runtime",
    "time",
    # proxy cost
    "nfe",
    "nfe_per_round",
    "number_of_fitness_evaluations",
    "fitness_evaluations",

    # Multi-hop cost diagnostics
    "pkt_hops",
    "pkt_hops_cum",
    "pkt_hops_round",
    "avg_pkt_hops_round",
    "mh_avg_path_hops",
    "mh_q_max",
    "energy_per_report",
    "energy_spent_round",
}


def metric_direction(metric: str) -> MetricDirection:
    raw = str(metric)
    m = _norm_metric_name(raw)

    if m in _HIGHER_IS_BETTER:
        return MetricDirection(direction="higher", assumed=False)
    if m in _LOWER_IS_BETTER:
        # document note: CPU primary, NFE secondary
        if "cpu" in m or m in {"time", "runtime"}:
            return MetricDirection(direction="lower", assumed=False, note="primary cost")
        if m.startswith("nfe") or "fitness" in m:
            return MetricDirection(direction="lower", assumed=False, note="secondary cost")
        return MetricDirection(direction="lower", assumed=False)

    # Explicit pattern rules for paper-style labels
    # - cumulative delivered reports / throughput
    if "cumulative" in m and ("deliver" in m or "report" in m or "throughput" in m):
        return MetricDirection(direction="higher", assumed=False)
    # - CH->BS traffic in packets
    if ("traffic" in m or "pkts" in m or "packet" in m) and ("sink" in m or "ch_bs" in m or "ch" in m):
        return MetricDirection(direction="higher", assumed=False)
    # - total residual energy over time
    if "residual" in m and "energy" in m:
        return MetricDirection(direction="higher", assumed=False)

    # - NFE phrasing (explicit secondary cost)
    if "nfe" in m or ("fitness" in m and "evaluat" in m):
        return MetricDirection(direction="lower", assumed=False, note="secondary cost")

    # heuristics for unknown names
    if any(tok in m for tok in ["loss", "latency", "delay", "error", "rmse", "mae", "mse"]):
        return MetricDirection(direction="lower", assumed=True)
    if any(tok in m for tok in ["cpu", "time", "runtime", "nfe", "fitness"]):
        return MetricDirection(direction="lower", assumed=True)
    return MetricDirection(direction="higher", assumed=True)


def direction_phrase(metric: str) -> str:
    md = metric_direction(metric)
    base = "higher is better" if md.direction == "higher" else "lower is better"
    if md.note:
        base = f"{base} ({md.note})"
    if md.assumed:
        return f"{base} (assumed)"
    return base


_METRIC_DISPLAY_MAP: Dict[str, str] = {
    # lifetime
    "fnd": "FND (rounds)",
    "hnd": "HND (rounds)",
    "lnd": "LND (rounds)",
    "rlast": "Rlast (round)",
    "r_last": "Rlast (round)",
    "last_round": "Rlast (round)",
    "last_useful_round": "Rlast (round)",
    # throughput / delivered reports
    "throughput": "Cumulative delivered reports (reports)",
    "throughput_reports": "Cumulative delivered reports (reports)",
    "throughput_cum": "Cumulative delivered reports (reports)",
    "cumulative_delivered_reports": "Cumulative delivered reports (reports)",
    "delivered_reports": "Delivered reports",
    "delivered_reports_r": "Delivered reports per round",
    "delivered_reports_cum": "Cumulative delivered reports (reports)",
    # CH->BS traffic
    "pkts_to_sink": "CH->BS traffic (pkts)",
    "pkts_to_sink_r": "CH->BS traffic per round (pkts)",
    "pkts_to_sink_cum": "Cumulative CH->BS traffic (pkts)",
    "cumulative_ch_bs_traffic_pkts": "Cumulative CH->BS traffic (pkts)",

    # Multi-hop traffic + diagnostics
    "pkt_hops": "Inter-CH/Sink hop-traffic (packet-hops)",
    "pkt_hops_round": "Inter-CH/Sink hop-traffic per round (packet-hops)",
    "pkt_hops_cum": "Cumulative inter-CH/Sink hop-traffic (packet-hops)",
    "avg_pkt_hops_round": "Inter-CH/Sink hop-traffic per round (packet-hops)",
    "mh_avg_path_hops": "Avg CH→sink path length (hops)",
    "mh_q_max": "Max relay load (packet transmissions)",
    "mh_jain_q": "Jain fairness of relay load (q)",
    "energy_per_report": "Energy per delivered report (J/report)",
    "energy_spent_round": "Energy spent per round (J)",
    # energy
    "total_energy": "Total residual energy (J)",
    "total_residual_energy": "Total residual energy (J)",
    "total_residual_energy_over_time": "Total residual energy (J)",
    # misc curves
    "alive": "Alive nodes",
    # compute cost
    "cpu_opt_s": "CPU time per round (s)",
    "cpu_time": "CPU time per round",
    "cpu_time_s": "CPU time per round (s)",
    "cpu_time_per_round": "CPU time per round",
    "runtime": "Runtime",
    "time": "Time",
    # NFE
    "nfe": "NFE per round",
    "nfe_per_round": "NFE per round",
    "number_of_fitness_evaluations": "NFE per round",
    "fitness_evaluations": "NFE per round",
}


def metric_display_name(metric: str) -> str:
    """User-facing label for metrics without affecting internal column keys."""
    raw = str(metric).strip()
    key = _norm_metric_name(raw)
    return _METRIC_DISPLAY_MAP.get(key, raw)
