#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wsn.plot_style import (
    algo_color,
    direction_phrase,
    display_algo_name,
    is_central_algo,
    metric_display_name,
)


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(s))


def _load_sheets(excel: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = pd.read_excel(excel, sheet_name="Meta")
    stats = pd.read_excel(excel, sheet_name="SummaryStats")
    wil = pd.read_excel(excel, sheet_name="Wilcoxon")
    return meta, stats, wil


def _meta_value(meta: pd.DataFrame, key: str) -> Optional[str]:
    try:
        v = meta.loc[meta["key"] == key, "value"].iloc[0]
        s = str(v).strip()
        return s or None
    except Exception:
        return None


def _resolve_meta_path(excel: Path, raw_path: str) -> Path:
    def _find_project_root(start: Path) -> Optional[Path]:
        for parent in [start, *start.parents]:
            if (parent / "requirements.txt").exists() or (parent / "README.md").exists() or (parent / "wsn").is_dir():
                return parent
        return None

    raw_path = str(raw_path).strip()
    if not raw_path:
        raise FileNotFoundError("Meta path not found: (empty)")

    p = Path(raw_path)
    attempted: List[Path] = []

    # 1) Exact path as stored in Meta (works for correct absolute paths).
    attempted.append(p)
    if p.exists():
        return p

    # 2) Interpret relative paths against common bases.
    if not p.is_absolute():
        for base in [excel.parent, Path.cwd()]:
            cand = (base / p)
            attempted.append(cand)
            if cand.exists():
                return cand

    # 3) Same filename next to the Excel (legacy behavior).
    p2 = excel.parent / p.name
    attempted.append(p2)
    if p2.exists():
        return p2

    # 4) Workspace/project root locations.
    root = _find_project_root(excel.parent)
    if root is not None:
        cand = root / p
        attempted.append(cand)
        if cand.exists():
            return cand

        cand2 = root / p.name
        attempted.append(cand2)
        if cand2.exists():
            return cand2

        # 5) Search under simulations/** by filename (common layout in this repo).
        simulations_dir = root / "simulations"
        if simulations_dir.is_dir():
            matches = list(simulations_dir.rglob(p.name))
            matches = [m for m in matches if m.is_file()]
            if matches:
                # Prefer the match closest to the Excel location; fall back to newest mtime.
                excel_parent = excel.parent.resolve()

                def _score(match: Path) -> tuple[int, float]:
                    try:
                        mp = match.resolve()
                        common = len(set(excel_parent.parents).intersection(set(mp.parents)))
                    except Exception:
                        common = 0
                    try:
                        mtime = match.stat().st_mtime
                    except Exception:
                        mtime = 0.0
                    return (common, mtime)

                best = max(matches, key=_score)
                return best

    tried = "\n".join(f" - {t}" for t in attempted)
    raise FileNotFoundError(
        "Meta path not found: "
        + raw_path
        + "\nTried:" \
        + ("\n" + tried if tried else " (no candidates)")
        + "\nHint: ensure Meta.timeseries_csv points to an existing CSV (absolute path, relative to the Excel, or under simulations/**)."
    )


def _load_summary_csv(excel: Path, meta: pd.DataFrame) -> pd.DataFrame:
    raw = _meta_value(meta, "summary_csv")
    if not raw:
        raise ValueError("Meta.summary_csv manquant: requis pour box/hist")
    return pd.read_csv(_resolve_meta_path(excel, raw))


def _load_timeseries_csv(excel: Path, meta: pd.DataFrame) -> pd.DataFrame:
    raw = _meta_value(meta, "timeseries_csv")
    if not raw:
        raise ValueError("Meta.timeseries_csv manquant: requis pour line")
    return pd.read_csv(_resolve_meta_path(excel, raw))


def _resolve_timeseries_metric_column(ts: pd.DataFrame, metric: str) -> tuple[str, str]:
    """Return (column_to_use, label_metric).

    The CLI metrics are often stored with suffixes in timeseries CSVs, e.g.
    - throughput -> throughput_cum
    - pkts_to_sink -> pkts_to_sink_cum / pkts_to_sink_round
    """

    if metric in ts.columns:
        return metric, metric

    aliases = {
        "throughput": "throughput_cum",
    }
    aliased = aliases.get(metric)
    if aliased and aliased in ts.columns:
        return aliased, metric

    candidates = [
        f"{metric}_cum",
        f"{metric}_round",
        f"{metric}__cum",
        f"{metric}__round",
    ]
    for c in candidates:
        if c in ts.columns:
            return c, metric

    cols = list(map(str, ts.columns))
    preview = ", ".join(cols[:30]) + (" …" if len(cols) > 30 else "")
    raise ValueError(
        f"Metric introuvable dans timeseries CSV: {metric}\n"
        f"Colonnes disponibles: {preview}"
    )


def _pick_scenario_label(excel: Path, meta: pd.DataFrame) -> str:
    """Prefer label derived from summary_csv in Meta, else excel filename."""
    try:
        summary_csv = str(meta.loc[meta["key"] == "summary_csv", "value"].iloc[0])
        name = Path(summary_csv).name
        if name.endswith("_ALL_summary.csv"):
            return name[: -len("_ALL_summary.csv")]
    except Exception:
        pass

    base = excel.name
    if base.endswith("_report.xlsx"):
        return base[: -len("_report.xlsx")]
    return excel.stem


def _extract_metric_by_algo(stats: pd.DataFrame, metric: str) -> pd.DataFrame:
    mean_col = metric + "_mean"
    std_col = metric + "_std"
    if mean_col not in stats.columns:
        raise ValueError(f"Metric introuvable dans SummaryStats: {metric}")

    cols = ["algo", mean_col] + ([std_col] if std_col in stats.columns else [])
    out = stats[cols].copy()
    out = out.rename(columns={mean_col: "mean", std_col: "std"} if std_col in out.columns else {mean_col: "mean"})
    if "std" not in out.columns:
        out["std"] = np.nan
    out["algo"] = out["algo"].astype(str)
    return out


def plot_scenario_compare_bar(
    excels: List[Path],
    out_dir: Path,
    metric: str,
    algos: List[str],
    title: str | None,
) -> Path:
    rows: List[Dict] = []

    scenario_labels: List[str] = []
    for excel in excels:
        meta, stats, _wil = _load_sheets(excel)
        scenario = _pick_scenario_label(excel, meta)
        scenario_labels.append(scenario)

        df = _extract_metric_by_algo(stats, metric)
        for a in algos:
            sub = df[df["algo"] == a]
            if sub.empty:
                continue
            rows.append(
                {
                    "scenario": scenario,
                    "algo": a,
                    "mean": float(sub["mean"].iloc[0]),
                    "std": float(sub["std"].iloc[0]) if pd.notna(sub["std"].iloc[0]) else np.nan,
                }
            )

    if not rows:
        raise ValueError("Aucune donnée trouvée: vérifiez --metric et --algos")

    data = pd.DataFrame(rows)

    # Sort scenarios by mean for central algo if present; else keep input order
    central = next((a for a in algos if is_central_algo(a)), None)
    if central is not None and (data["algo"] == central).any():
        # Aggregate in case we have duplicated scenario labels (e.g., same scenario label across files)
        pivot_c = data[data["algo"] == central].groupby("scenario", dropna=False)["mean"].mean()
        scenario_order = pivot_c.sort_values(
            ascending=("lower is better" in direction_phrase(metric))
        ).index.tolist()
    else:
        scenario_order = list(dict.fromkeys(scenario_labels))

    data["scenario"] = pd.Categorical(data["scenario"], categories=scenario_order, ordered=True)

    # Build grouped bar plot
    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(scenario_order)), 5.2))
    n_sc = len(scenario_order)
    n_a = max(len(algos), 1)

    x = np.arange(n_sc)
    total_width = 0.82
    bar_w = total_width / n_a
    offsets = (np.arange(n_a) - (n_a - 1) / 2.0) * bar_w

    for i, algo in enumerate(algos):
        sub = data[data["algo"] == algo].sort_values("scenario")

        # align to scenario_order
        by_sc = sub.set_index("scenario").reindex(scenario_order)
        y = by_sc["mean"].to_numpy(dtype=float)
        yerr = by_sc["std"].to_numpy(dtype=float)
        color = algo_color(algo)

        ax.bar(
            x + offsets[i],
            y,
            width=bar_w,
            color=color,
            yerr=yerr if np.isfinite(yerr).any() else None,
            capsize=3,
            label=display_algo_name(algo),
            edgecolor=("black" if is_central_algo(algo) else "none"),
            linewidth=(1.5 if is_central_algo(algo) else 0.0),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_order, rotation=30, ha="right")
    ax.set_ylabel(metric_display_name(metric))

    t = title or f"Scenario comparison — {metric_display_name(metric)}"
    ax.set_title(t)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    # Direction annotation
    ax.text(
        0.99,
        0.99,
        direction_phrase(metric),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.9),
    )

    fig.tight_layout()

    out = out_dir / f"scenarios__{_safe_name(metric)}__bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_scenario_compare_heatmap(
    excels: List[Path],
    out_dir: Path,
    metric: str,
    algos: List[str],
    title: str | None,
) -> Path:
    rows: List[Dict] = []
    scenario_labels: List[str] = []
    for excel in excels:
        meta, stats, _wil = _load_sheets(excel)
        scenario = _pick_scenario_label(excel, meta)
        scenario_labels.append(scenario)
        df = _extract_metric_by_algo(stats, metric)
        for a in algos:
            sub = df[df["algo"] == a]
            if sub.empty:
                continue
            rows.append({"scenario": scenario, "algo": a, "mean": float(sub["mean"].iloc[0])})

    if not rows:
        raise ValueError("Aucune donnée trouvée: vérifiez --metric et --algos")

    data = pd.DataFrame(rows)
    scenario_order = list(dict.fromkeys(scenario_labels))
    algo_order = list(algos)
    mat = data.pivot(index="algo", columns="scenario", values="mean").reindex(index=algo_order, columns=scenario_order)

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(scenario_order)), max(4, 0.45 * len(algo_order))))
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(scenario_order)))
    ax.set_xticklabels(scenario_order, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(algo_order)))
    ax.set_yticklabels([display_algo_name(a) for a in algo_order])
    fig.colorbar(im, ax=ax, label=metric_display_name(metric))

    ax.set_title(title or f"Scenario comparison — {metric_display_name(metric)} (heatmap)")
    ax.text(
        0.99,
        0.99,
        direction_phrase(metric),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.9),
    )
    fig.tight_layout()

    out = out_dir / f"scenarios__{_safe_name(metric)}__heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_scenario_compare_scatter(
    excels: List[Path],
    out_dir: Path,
    metric: str,
    algos: List[str],
    title: str | None,
) -> Path:
    rows: List[Dict] = []
    scenario_labels: List[str] = []
    for excel in excels:
        meta, stats, _wil = _load_sheets(excel)
        scenario = _pick_scenario_label(excel, meta)
        scenario_labels.append(scenario)
        df = _extract_metric_by_algo(stats, metric)
        for a in algos:
            sub = df[df["algo"] == a]
            if sub.empty:
                continue
            rows.append({"scenario": scenario, "algo": a, "mean": float(sub["mean"].iloc[0])})

    if not rows:
        raise ValueError("Aucune donnée trouvée: vérifiez --metric et --algos")

    data = pd.DataFrame(rows)
    scenario_order = list(dict.fromkeys(scenario_labels))

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(scenario_order)), 5.2))
    x = np.arange(len(scenario_order))
    n_a = max(len(algos), 1)
    offsets = (np.arange(n_a) - (n_a - 1) / 2.0) * 0.09

    for i, algo in enumerate(algos):
        sub = data[data["algo"] == algo].set_index("scenario").reindex(scenario_order)
        y = sub["mean"].to_numpy(dtype=float)
        ax.scatter(
            x + offsets[i],
            y,
            s=(70 if is_central_algo(algo) else 45),
            color=algo_color(algo),
            edgecolor=("black" if is_central_algo(algo) else "none"),
            linewidth=(1.2 if is_central_algo(algo) else 0.0),
            label=display_algo_name(algo),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_order, rotation=30, ha="right")
    ax.set_ylabel(metric_display_name(metric))
    ax.set_title(title or f"Scenario comparison — {metric_display_name(metric)} (scatter)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax.text(
        0.99,
        0.99,
        direction_phrase(metric),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.9),
    )
    fig.tight_layout()

    out = out_dir / f"scenarios__{_safe_name(metric)}__scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_scenario_compare_box(
    excels: List[Path],
    out_dir: Path,
    metric: str,
    algos: List[str],
) -> List[Path]:
    # One figure per algo: scenarios on x-axis, distribution per scenario.
    scenario_labels: List[str] = []
    per_scenario: Dict[str, pd.DataFrame] = {}
    for excel in excels:
        meta, _stats, _wil = _load_sheets(excel)
        scenario = _pick_scenario_label(excel, meta)
        scenario_labels.append(scenario)
        df = _load_summary_csv(excel, meta)
        if metric not in df.columns:
            raise ValueError(f"Metric introuvable dans summary CSV: {metric}")
        per_scenario[scenario] = df

    scenario_order = list(dict.fromkeys(scenario_labels))
    written: List[Path] = []

    for algo in algos:
        groups = []
        for sc in scenario_order:
            df = per_scenario[sc]
            vals = df.loc[df["algo"].astype(str) == algo, metric].astype(float).dropna().to_numpy()
            groups.append(vals)

        if not any(len(g) for g in groups):
            continue

        fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(scenario_order)), 5.2))
        bp = ax.boxplot(groups, patch_artist=True, showfliers=False)
        for box in bp["boxes"]:
            box.set_facecolor(algo_color(algo))
            box.set_alpha(0.55)
            if is_central_algo(algo):
                box.set_edgecolor("black")
                box.set_linewidth(1.5)
            else:
                box.set_edgecolor("none")

        ax.set_xticks(np.arange(1, len(scenario_order) + 1))
        ax.set_xticklabels(scenario_order, rotation=30, ha="right")
        ax.set_ylabel(metric_display_name(metric))
        ax.set_title(f"Scenario comparison — {metric_display_name(metric)} (box) — {display_algo_name(algo)}")
        ax.text(
            0.99,
            0.99,
            direction_phrase(metric),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.9),
        )
        fig.tight_layout()

        out = out_dir / f"scenarios__{_safe_name(metric)}__box__{_safe_name(algo)}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)

    if not written:
        raise ValueError("Aucune donnée trouvée pour box: vérifiez --algos et --metric")
    return written


def plot_scenario_compare_hist(
    excels: List[Path],
    out_dir: Path,
    metric: str,
    algos: List[str],
) -> List[Path]:
    scenario_labels: List[str] = []
    per_scenario: Dict[str, pd.DataFrame] = {}
    for excel in excels:
        meta, _stats, _wil = _load_sheets(excel)
        scenario = _pick_scenario_label(excel, meta)
        scenario_labels.append(scenario)
        df = _load_summary_csv(excel, meta)
        if metric not in df.columns:
            raise ValueError(f"Metric introuvable dans summary CSV: {metric}")
        per_scenario[scenario] = df

    scenario_order = list(dict.fromkeys(scenario_labels))
    written: List[Path] = []

    for algo in algos:
        fig, ax = plt.subplots(figsize=(10, 5.2))
        any_data = False
        for sc in scenario_order:
            df = per_scenario[sc]
            vals = df.loc[df["algo"].astype(str) == algo, metric].astype(float).dropna().to_numpy()
            if len(vals) == 0:
                continue
            any_data = True
            ax.hist(vals, bins=18, alpha=0.35, label=sc)

        if not any_data:
            plt.close(fig)
            continue

        ax.set_xlabel(metric_display_name(metric))
        ax.set_ylabel("count")
        ax.set_title(f"Scenario comparison — {metric_display_name(metric)} (hist) — {display_algo_name(algo)}")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
        ax.text(
            0.99,
            0.99,
            direction_phrase(metric),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.9),
        )
        fig.tight_layout()

        out = out_dir / f"scenarios__{_safe_name(metric)}__hist__{_safe_name(algo)}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)

    if not written:
        raise ValueError("Aucune donnée trouvée pour hist: vérifiez --algos et --metric")
    return written


def plot_scenario_compare_line(
    excels: List[Path],
    out_dir: Path,
    metric: str,
    algos: List[str],
) -> List[Path]:
    scenario_labels: List[str] = []
    per_scenario_ts: Dict[str, pd.DataFrame] = {}
    per_scenario_metric_col: Dict[str, str] = {}
    label_metric = metric
    for excel in excels:
        meta, _stats, _wil = _load_sheets(excel)
        scenario = _pick_scenario_label(excel, meta)
        scenario_labels.append(scenario)
        ts = _load_timeseries_csv(excel, meta)
        col, label_metric = _resolve_timeseries_metric_column(ts, metric)
        per_scenario_metric_col[scenario] = col
        per_scenario_ts[scenario] = ts

    scenario_order = list(dict.fromkeys(scenario_labels))
    written: List[Path] = []

    # One figure per algo: scenarios as different linestyles.
    linestyles = ["-", "--", ":", "-."]

    for algo in algos:
        fig, ax = plt.subplots(figsize=(10, 5.2))
        any_data = False
        for i, sc in enumerate(scenario_order):
            ts = per_scenario_ts[sc]
            metric_col = per_scenario_metric_col[sc]
            sub = ts.loc[ts["algo"].astype(str) == algo, ["round", metric_col]].copy()
            if sub.empty:
                continue
            any_data = True
            agg = (
                sub.groupby("round", dropna=False)[metric_col]
                .mean(numeric_only=True)
                .reset_index()
                .sort_values("round")
            )
            ax.plot(
                agg["round"].to_numpy(dtype=float),
                agg[metric_col].to_numpy(dtype=float),
                color=algo_color(algo),
                linestyle=linestyles[i % len(linestyles)],
                linewidth=(3.0 if is_central_algo(algo) else 1.8),
                label=sc,
            )

        if not any_data:
            plt.close(fig)
            continue

        ax.set_xlabel("round")
        ax.set_ylabel(metric_display_name(label_metric))
        ax.set_title(f"Scenario comparison — {metric_display_name(label_metric)} (line) — {display_algo_name(algo)}")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
        ax.text(
            0.99,
            0.99,
            direction_phrase(label_metric),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.9),
        )
        fig.tight_layout()

        out = out_dir / f"scenarios__{_safe_name(label_metric)}__line__{_safe_name(algo)}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)

    if not written:
        raise ValueError("Aucune donnée trouvée pour line: vérifiez --algos et --metric")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Compare multiple scenario Excel reports on a chosen metric.")
    p.add_argument(
        "--excels",
        nargs="+",
        required=True,
        help="One or more paths to *_report.xlsx (each is a scenario)",
    )
    p.add_argument("--out_dir", default=".", help="Output directory")
    p.add_argument("--metric", required=True, help="Metric base name (e.g., FND, throughput, avg_cpu_time)")
    p.add_argument(
        "--algos",
        default="FSS",
        help="Comma-separated list of algorithms to show (default: FSS)",
    )
    p.add_argument(
        "--title",
        default="",
        help="Optional plot title override",
    )

    p.add_argument(
        "--plots",
        default="bar",
        help="Comma-separated list: scatter,bar,box,line,hist,heatmap",
    )

    args = p.parse_args()

    excels = [Path(x) for x in args.excels]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    algos = [a.strip() for a in str(args.algos).split(",") if a.strip()]

    plots = [p.strip().lower() for p in str(args.plots).split(",") if p.strip()]
    written: List[Path] = []
    title = (str(args.title).strip() or None)

    for pt in plots:
        if pt == "bar":
            written.append(
                plot_scenario_compare_bar(
                    excels=excels,
                    out_dir=out_dir,
                    metric=str(args.metric),
                    algos=algos,
                    title=title,
                )
            )
        elif pt == "scatter":
            written.append(
                plot_scenario_compare_scatter(
                    excels=excels,
                    out_dir=out_dir,
                    metric=str(args.metric),
                    algos=algos,
                    title=title,
                )
            )
        elif pt == "heatmap":
            written.append(
                plot_scenario_compare_heatmap(
                    excels=excels,
                    out_dir=out_dir,
                    metric=str(args.metric),
                    algos=algos,
                    title=title,
                )
            )
        elif pt == "box":
            written.extend(
                plot_scenario_compare_box(
                    excels=excels,
                    out_dir=out_dir,
                    metric=str(args.metric),
                    algos=algos,
                )
            )
        elif pt == "hist":
            written.extend(
                plot_scenario_compare_hist(
                    excels=excels,
                    out_dir=out_dir,
                    metric=str(args.metric),
                    algos=algos,
                )
            )
        elif pt == "line":
            written.extend(
                plot_scenario_compare_line(
                    excels=excels,
                    out_dir=out_dir,
                    metric=str(args.metric),
                    algos=algos,
                )
            )
        else:
            raise SystemExit(f"Unknown plot type: {pt}")

    print("Saved plots:")
    for w in written:
        print(f" - {w}")


if __name__ == "__main__":
    main()
