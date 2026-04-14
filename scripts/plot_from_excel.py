#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wsn.plot_style import (
    algo_color,
    display_algo_name,
    direction_phrase,
    is_central_algo,
    metric_display_name,
)


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)


def _infer_direction(metric: str) -> str:
    # kept for backward-compat; prefer wsn.plot_style.direction_phrase/metric_direction
    return "lower" if "lower is better" in direction_phrase(metric) else "higher"


def _annotate_direction(ax: plt.Axes, metric: str) -> None:
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


def _load_sheets(excel: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    """Resolve a path stored in Meta.

    The Excel may have been moved; try:
    1) the raw path as-is
    2) same directory as the Excel, using basename of raw path
    """

    p = Path(raw_path)
    if p.exists():
        return p

    # If report moved, the CSVs are typically alongside it.
    p2 = excel.parent / p.name
    if p2.exists():
        return p2

    raise FileNotFoundError(f"Meta path not found: {raw_path}")


def _load_summary_csv(meta: pd.DataFrame, excel: Path) -> pd.DataFrame:
    raw = _meta_value(meta, "summary_csv")
    if not raw:
        raise ValueError("Meta.summary_csv manquant: requis pour box/hist")
    p = _resolve_meta_path(excel, raw)
    return pd.read_csv(p)


def _load_timeseries_csv(meta: pd.DataFrame, excel: Path) -> pd.DataFrame:
    raw = _meta_value(meta, "timeseries_csv")
    if not raw:
        raise ValueError("Meta.timeseries_csv manquant: requis pour line")
    p = _resolve_meta_path(excel, raw)
    return pd.read_csv(p)


def _pick_prefix(meta: pd.DataFrame, excel: Path) -> str:
    try:
        summary_csv = str(meta.loc[meta["key"] == "summary_csv", "value"].iloc[0])
        name = Path(summary_csv).name
        if name.endswith("_ALL_summary.csv"):
            return name[: -len("_ALL_summary.csv")]
    except Exception:
        pass

    # fallback from filename
    base = excel.name
    if base.endswith("_report.xlsx"):
        return base[: -len("_report.xlsx")]
    return excel.stem


def plot_mean_bar(stats: pd.DataFrame, out_dir: Path, prefix: str, metric: str) -> Path:
    mean_col = metric + "_mean"
    std_col = metric + "_std"
    if mean_col not in stats.columns:
        raise ValueError(f"Metric introuvable dans SummaryStats: {metric}")

    df = stats[["algo", mean_col] + ([std_col] if std_col in stats.columns else [])].copy()
    df = df.sort_values(mean_col, ascending=(_infer_direction(metric) == "lower"))

    x = np.arange(len(df))
    y = df[mean_col].to_numpy(dtype=float)
    yerr = df[std_col].to_numpy(dtype=float) if std_col in df.columns else None

    fig, ax = plt.subplots(figsize=(10, 4.8))

    algos = df["algo"].astype(str).tolist()
    colors = [algo_color(a) for a in algos]
    edgecolors = ["black" if is_central_algo(a) else "none" for a in algos]
    linewidths = [1.5 if is_central_algo(a) else 0.0 for a in algos]

    ax.bar(x, y, yerr=yerr, capsize=3, color=colors, edgecolor=edgecolors, linewidth=linewidths)
    ax.set_xticks(x)
    ax.set_xticklabels([display_algo_name(a) for a in algos], rotation=30, ha="right")
    plt.ylabel(metric_display_name(metric))
    plt.title(f"{prefix} - {metric_display_name(metric)} (mean±std)")

    # Legend (required): one entry per algorithm
    from matplotlib.patches import Patch

    handles = [
        Patch(
            facecolor=algo_color(a),
            edgecolor=("black" if is_central_algo(a) else "none"),
            label=display_algo_name(a),
        )
        for a in algos
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    _annotate_direction(ax, metric)
    fig.tight_layout()

    out = out_dir / f"{prefix}__mean_bar__{_safe_name(metric)}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_means_heatmap(stats: pd.DataFrame, out_dir: Path, prefix: str) -> Path:
    metric_cols = [c for c in stats.columns if c.endswith("_mean")]
    if not metric_cols:
        raise ValueError("Aucune colonne *_mean dans SummaryStats")

    mat = stats.set_index("algo")[metric_cols].copy()
    # Z-score per metric for readability
    mat = (mat - mat.mean(axis=0)) / (mat.std(axis=0) + 1e-12)

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * mat.shape[1]), max(4, 0.4 * mat.shape[0])))
    im = ax.imshow(mat.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels([display_algo_name(a) for a in mat.index.astype(str).tolist()])
    plt.xticks(np.arange(mat.shape[1]), [c[:-5] for c in metric_cols], rotation=45, ha="right")
    fig.colorbar(im, ax=ax, label="z-score")
    plt.title(f"{prefix} - Means heatmap (z-score)")
    ax.text(
        0.99,
        0.01,
        "Note: interpretation depends on per-metric direction (higher is better / lower is better)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.9),
    )
    fig.tight_layout()

    out = out_dir / f"{prefix}__means_heatmap.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_rank_heatmap(stats: pd.DataFrame, out_dir: Path, prefix: str) -> Path:
    metric_cols = [c for c in stats.columns if c.endswith("_mean")]
    if not metric_cols:
        raise ValueError("Aucune colonne *_mean dans SummaryStats")

    mean = stats.set_index("algo")[metric_cols].copy()

    ranks = pd.DataFrame(index=mean.index)
    for c in metric_cols:
        metric = c[:-5]
        asc = (_infer_direction(metric) == "lower")
        ranks[metric] = mean[c].rank(ascending=asc, method="min")

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * ranks.shape[1]), max(4, 0.4 * ranks.shape[0])))
    im = ax.imshow(ranks.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(ranks.shape[0]))
    ax.set_yticklabels([display_algo_name(a) for a in ranks.index.astype(str).tolist()])
    plt.xticks(np.arange(ranks.shape[1]), ranks.columns.tolist(), rotation=45, ha="right")
    fig.colorbar(im, ax=ax, label="rank (1=best)")
    plt.title(f"{prefix} - Rank heatmap")
    ax.text(
        0.99,
        0.01,
        "Ranking uses per-metric direction: higher is better / lower is better",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.9),
    )
    fig.tight_layout()

    out = out_dir / f"{prefix}__rank_heatmap.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_pvalue_heatmap(wil: pd.DataFrame, out_dir: Path, prefix: str, use_holm: bool = True) -> Path:
    if wil.empty:
        raise ValueError("Feuille Wilcoxon vide")

    pcol = "p_holm" if use_holm and "p_holm" in wil.columns else "p_value"
    mat = wil.pivot(index="algo", columns="metric", values=pcol)

    # transform for visibility
    mat2 = -np.log10(mat.astype(float).clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * mat2.shape[1]), max(4, 0.4 * mat2.shape[0])))
    im = ax.imshow(mat2.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(mat2.shape[0]))
    ax.set_yticklabels([display_algo_name(a) for a in mat2.index.astype(str).tolist()])
    plt.xticks(np.arange(mat2.shape[1]), mat2.columns.astype(str).tolist(), rotation=45, ha="right")
    fig.colorbar(im, ax=ax, label=f"-log10({pcol})")
    if "ref_algo" in wil.columns and wil["ref_algo"].nunique(dropna=True) == 1:
        ref = str(wil["ref_algo"].dropna().iloc[0])
        plt.title(f"{prefix} - P-values heatmap ({display_algo_name(ref)} vs baselines)")
    else:
        plt.title(f"{prefix} - P-values heatmap")
    fig.tight_layout()

    out = out_dir / f"{prefix}__pvalue_heatmap__{pcol}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_effect_bar(wil: pd.DataFrame, out_dir: Path, prefix: str, metric: str) -> Path:
    df = wil[wil["metric"] == metric].copy()
    if df.empty:
        raise ValueError(f"Aucune ligne Wilcoxon pour metric={metric}")

    df = df.sort_values("r_rb")

    x = np.arange(len(df))
    y = df["r_rb"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    algos = df["algo"].astype(str).tolist()
    colors = [algo_color(a) for a in algos]
    ax.barh(x, y, color=colors)
    ax.set_yticks(x)
    ax.set_yticklabels([display_algo_name(a) for a in algos])
    ax.set_xlabel("r_rb (rank-biserial)")
    if "ref_algo" in wil.columns and wil["ref_algo"].nunique(dropna=True) == 1:
        ref = str(wil["ref_algo"].dropna().iloc[0])
        ax.set_title(f"{prefix} - Effect size ({display_algo_name(ref)} − baseline) ({metric_display_name(metric)})")
    else:
        ax.set_title(f"{prefix} - Effect size vs baseline ({metric_display_name(metric)})")

    from matplotlib.patches import Patch

    handles = [Patch(facecolor=algo_color(a), label=display_algo_name(a)) for a in algos]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    _annotate_direction(ax, metric)
    fig.tight_layout()

    out = out_dir / f"{prefix}__effect_bar__{_safe_name(metric)}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_volcano(wil: pd.DataFrame, out_dir: Path, prefix: str, metric: str) -> Path:
    df = wil[wil["metric"] == metric].copy()
    if df.empty:
        raise ValueError(f"Aucune ligne Wilcoxon pour metric={metric}")

    pcol = "p_holm" if "p_holm" in df.columns and df["p_holm"].notna().any() else "p_value"
    x = df["r_rb"].to_numpy(dtype=float)
    y = -np.log10(df[pcol].astype(float).clip(lower=1e-300).to_numpy())

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    algos = df["algo"].astype(str).tolist()
    colors = [algo_color(a) for a in algos]
    ax.scatter(x, y, c=colors)
    for i, algo in enumerate(algos):
        ax.annotate(display_algo_name(algo), (x[i], y[i]), fontsize=8, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("r_rb")
    ax.set_ylabel(f"-log10({pcol})")
    if "ref_algo" in wil.columns and wil["ref_algo"].nunique(dropna=True) == 1:
        ref = str(wil["ref_algo"].dropna().iloc[0])
        ax.set_title(f"{prefix} - Volcano ({display_algo_name(ref)} − baseline) ({metric_display_name(metric)})")
    else:
        ax.set_title(f"{prefix} - Volcano ({metric_display_name(metric)})")
    _annotate_direction(ax, metric)
    fig.tight_layout()

    out = out_dir / f"{prefix}__volcano__{_safe_name(metric)}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_box(summary_df: pd.DataFrame, out_dir: Path, prefix: str, metric: str) -> Path:
    if metric not in summary_df.columns:
        raise ValueError(f"Metric introuvable dans summary CSV: {metric}")
    if "algo" not in summary_df.columns:
        raise ValueError("Colonne 'algo' manquante dans summary CSV")

    df = summary_df[["algo", metric]].copy()
    df["algo"] = df["algo"].astype(str)

    # Put central algo first, then others for readability
    algos = df["algo"].dropna().unique().tolist()
    algos = sorted(algos, key=lambda a: (not is_central_algo(a), a))

    groups = [df.loc[df["algo"] == a, metric].astype(float).dropna().to_numpy() for a in algos]
    if not any(len(g) for g in groups):
        raise ValueError("Aucune donnée pour boxplot")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bp = ax.boxplot(groups, patch_artist=True, showfliers=False)

    for i, box in enumerate(bp["boxes"]):
        a = algos[i]
        box.set_facecolor(algo_color(a))
        if is_central_algo(a):
            box.set_edgecolor("black")
            box.set_linewidth(1.5)
        else:
            box.set_edgecolor("none")

    ax.set_xticks(np.arange(1, len(algos) + 1))
    ax.set_xticklabels([display_algo_name(a) for a in algos], rotation=30, ha="right")
    ax.set_ylabel(metric_display_name(metric))
    ax.set_title(f"{prefix} - {metric_display_name(metric)} (box)")

    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=algo_color(a), edgecolor=("black" if is_central_algo(a) else "none"), label=display_algo_name(a))
        for a in algos
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    _annotate_direction(ax, metric)
    fig.tight_layout()

    out = out_dir / f"{prefix}__box__{_safe_name(metric)}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_hist(summary_df: pd.DataFrame, out_dir: Path, prefix: str, metric: str) -> Path:
    if metric not in summary_df.columns:
        raise ValueError(f"Metric introuvable dans summary CSV: {metric}")
    if "algo" not in summary_df.columns:
        raise ValueError("Colonne 'algo' manquante dans summary CSV")

    df = summary_df[["algo", metric]].copy()
    df["algo"] = df["algo"].astype(str)

    algos = df["algo"].dropna().unique().tolist()
    algos = sorted(algos, key=lambda a: (not is_central_algo(a), a))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for a in algos:
        vals = df.loc[df["algo"] == a, metric].astype(float).dropna().to_numpy()
        if len(vals) == 0:
            continue
        ax.hist(
            vals,
            bins=18,
            alpha=(0.6 if is_central_algo(a) else 0.35),
            color=algo_color(a),
            label=display_algo_name(a),
            edgecolor=("black" if is_central_algo(a) else "none"),
            linewidth=(1.0 if is_central_algo(a) else 0.0),
        )

    ax.set_xlabel(metric_display_name(metric))
    ax.set_ylabel("count")
    ax.set_title(f"{prefix} - {metric_display_name(metric)} (hist)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    _annotate_direction(ax, metric)
    fig.tight_layout()

    out = out_dir / f"{prefix}__hist__{_safe_name(metric)}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_line(timeseries_df: pd.DataFrame, out_dir: Path, prefix: str, metric: str) -> Path:
    # Allow common aliases: some metrics differ between summary and timeseries CSVs
    def _timeseries_alias(m: str) -> Optional[str]:
        aliases = {
            # summary vs timeseries naming
            "throughput": "throughput_cum",
            "pkts_to_sink": "pkts_to_sink_cum",
            "pkt_hops": "pkt_hops_cum",
            # average-of-round metrics map to per-round columns in timeseries
            "avg_delivered_reports_round": "delivered_reports_round",
            "avg_pkts_to_sink_round": "pkts_to_sink_round",
            "avg_pkt_hops_round": "pkt_hops_round",
            "avg_n_ch_round": "n_ch_round",
            # energy/alive naming differences
            "avg_energy": "total_energy",
            "avg_alive": "alive",
        }
        return aliases.get(m)

    col = metric
    if col not in timeseries_df.columns:
        alias = _timeseries_alias(metric)
        if alias and alias in timeseries_df.columns:
            col = alias
        else:
            raise ValueError(f"Metric introuvable dans timeseries CSV: {metric}")
    if "algo" not in timeseries_df.columns or "round" not in timeseries_df.columns:
        raise ValueError("Colonnes requises manquantes dans timeseries CSV: 'round' et/ou 'algo'")

    df = timeseries_df[["round", "algo", col]].copy()
    df["algo"] = df["algo"].astype(str)

    # Mean across seeds per round
    agg = df.groupby(["algo", "round"], dropna=False)[col].mean(numeric_only=True).reset_index()

    algos = agg["algo"].dropna().unique().tolist()
    algos = sorted(algos, key=lambda a: (not is_central_algo(a), a))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for a in algos:
        sub = agg[agg["algo"] == a].sort_values("round")
        ax.plot(
            sub["round"].to_numpy(dtype=float),
            sub[col].to_numpy(dtype=float),
            color=algo_color(a),
            linewidth=(3.0 if is_central_algo(a) else 1.6),
            label=display_algo_name(a),
        )

    ax.set_xlabel("round")
    ax.set_ylabel(metric_display_name(metric))
    ax.set_title(f"{prefix} - {metric_display_name(metric)} (line)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    _annotate_direction(ax, metric)
    fig.tight_layout()

    out = out_dir / f"{prefix}__line__{_safe_name(metric)}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create plots from an Excel report.")
    parser.add_argument("--excel", required=True, help="Path to *_report.xlsx")
    parser.add_argument("--out_dir", default=".", help="Output directory")
    parser.add_argument("--metric", default="throughput", help="Metric for metric-specific plots")
    parser.add_argument(
        "--plots",
        default="mean_bar",
        help="Comma-separated list: mean_bar,means_heatmap,rank_heatmap,pvalue_heatmap,effect_bar,volcano,box,hist,line",
    )

    args = parser.parse_args()

    excel = Path(args.excel)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta, stats, wil = _load_sheets(excel)
    prefix = _pick_prefix(meta, excel)

    plots = [p.strip() for p in str(args.plots).split(",") if p.strip()]

    written: List[Path] = []
    for p in plots:
        if p == "mean_bar":
            written.append(plot_mean_bar(stats, out_dir, prefix, str(args.metric)))
        elif p == "means_heatmap":
            written.append(plot_means_heatmap(stats, out_dir, prefix))
        elif p == "rank_heatmap":
            written.append(plot_rank_heatmap(stats, out_dir, prefix))
        elif p == "pvalue_heatmap":
            written.append(plot_pvalue_heatmap(wil, out_dir, prefix, use_holm=True))
        elif p == "effect_bar":
            written.append(plot_effect_bar(wil, out_dir, prefix, str(args.metric)))
        elif p == "volcano":
            written.append(plot_volcano(wil, out_dir, prefix, str(args.metric)))
        elif p == "box":
            summary_df = _load_summary_csv(meta, excel)
            written.append(plot_box(summary_df, out_dir, prefix, str(args.metric)))
        elif p == "hist":
            summary_df = _load_summary_csv(meta, excel)
            written.append(plot_hist(summary_df, out_dir, prefix, str(args.metric)))
        elif p == "line":
            ts_df = _load_timeseries_csv(meta, excel)
            written.append(plot_line(ts_df, out_dir, prefix, str(args.metric)))
        else:
            raise SystemExit(f"Unknown plot type: {p}")

    print("Saved plots:")
    for w in written:
        print(f" - {w}")


if __name__ == "__main__":
    main()
