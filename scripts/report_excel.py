#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


KEY_COLS = {"scenario", "algo", "seed", "bs_mode", "sink_x", "sink_y"}


def _numeric_metric_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c in KEY_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return p

    order = np.argsort(p)
    p_sorted = p[order]

    adj_sorted = np.empty_like(p_sorted)
    running_max = 0.0
    for i, pv in enumerate(p_sorted):
        k = m - i
        adj = min(1.0, k * pv)
        running_max = max(running_max, adj)
        adj_sorted[i] = running_max

    out = np.empty_like(adj_sorted)
    out[order] = adj_sorted
    return out


def _wilcoxon_stats(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, int]:
    """Return (W_plus, p_two_sided, r_rb, n_eff)."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    d = x[mask] - y[mask]

    # Remove zeros for rank-biserial computation
    d_nz = d[d != 0]
    n_eff = int(d_nz.size)
    if n_eff == 0:
        return 0.0, 1.0, 0.0, 0

    # p-value from scipy
    try:
        w_stat, p = wilcoxon(d_nz, alternative="two-sided", zero_method="wilcox")
    except TypeError:
        # Older scipy
        w_stat, p = wilcoxon(d_nz)

    # Rank-biserial effect size from signed ranks
    abs_d = np.abs(d_nz)
    ranks = pd.Series(abs_d).rank(method="average").to_numpy(dtype=float)

    w_plus = float(ranks[d_nz > 0].sum())
    w_minus = float(ranks[d_nz < 0].sum())
    denom = w_plus + w_minus
    r_rb = float((w_plus - w_minus) / denom) if denom > 0 else 0.0

    # Return W+ (more interpretable than min(W+,W-))
    return w_plus, float(p), r_rb, n_eff


def build_summary_stats(summary_df: pd.DataFrame) -> pd.DataFrame:
    metrics = _numeric_metric_columns(summary_df)
    if not metrics:
        return pd.DataFrame()

    g = summary_df.groupby("algo", dropna=False)

    mean_df = g[metrics].mean(numeric_only=True).add_suffix("_mean")
    std_df = g[metrics].std(numeric_only=True).add_suffix("_std")
    n_df = g.size().rename("n")

    out = pd.concat([n_df, mean_df, std_df], axis=1).reset_index()
    return out


def build_wilcoxon_table(summary_df: pd.DataFrame, baseline_algo: str, correction: str) -> pd.DataFrame:
    metrics = _numeric_metric_columns(summary_df)
    if not metrics:
        return pd.DataFrame()

    if "algo" not in summary_df.columns:
        raise ValueError("Colonne 'algo' manquante")

    algos = sorted(a for a in summary_df["algo"].dropna().unique().tolist())
    if baseline_algo not in algos:
        # fallback to first algo
        baseline_algo = algos[0] if algos else baseline_algo

    # Pair by seed within the same scenario/bs_mode if present
    pair_keys = [c for c in ["scenario", "seed", "bs_mode", "sink_x", "sink_y"] if c in summary_df.columns]

    base_df = summary_df[summary_df["algo"] == baseline_algo].copy()

    rows: List[Dict] = []

    for algo in algos:
        if algo == baseline_algo:
            continue

        other_df = summary_df[summary_df["algo"] == algo].copy()

        if pair_keys:
            merged = base_df[pair_keys + metrics].merge(
                other_df[pair_keys + metrics],
                on=pair_keys,
                how="inner",
                suffixes=("_base", "_other"),
            )
        else:
            # Fallback: align by row order
            min_len = min(len(base_df), len(other_df))
            merged = pd.DataFrame()
            for m in metrics:
                merged[m + "_base"] = base_df[m].iloc[:min_len].to_numpy()
                merged[m + "_other"] = other_df[m].iloc[:min_len].to_numpy()

        for metric in metrics:
            x = merged[metric + "_other"].to_numpy(dtype=float)
            y = merged[metric + "_base"].to_numpy(dtype=float)
            w_plus, p, r_rb, n_eff = _wilcoxon_stats(x, y)
            rows.append(
                {
                    "metric": metric,
                    "baseline": baseline_algo,
                    "algo": algo,
                    "n": n_eff,
                    "W_plus": w_plus,
                    "p_value": p,
                    "r_rb": r_rb,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    if correction == "holm":
        out["p_holm"] = _holm_adjust(out["p_value"].to_numpy(dtype=float))
    else:
        out["p_holm"] = np.nan

    return out.sort_values(["metric", "p_value"], ascending=[True, True]).reset_index(drop=True)


def build_wilcoxon_table_ref_vs_each(summary_df: pd.DataFrame, ref_algo: str, correction: str) -> pd.DataFrame:
    """Wilcoxon signed-rank with a fixed compared algorithm and a varying baseline.

    For each baseline B != ref_algo, compute paired differences:
        d_i = metric(ref_algo, seed_i) - metric(B, seed_i)
    and run Wilcoxon on {d_i} (two-sided).

    Output keeps the existing shape expected by plotting scripts:
      - column 'algo' contains the baseline algorithm B (one row per baseline)
      - column 'baseline' repeats B (explicit semantics)
      - column 'ref_algo' is constant
    """

    metrics = _numeric_metric_columns(summary_df)
    if not metrics:
        return pd.DataFrame()

    if "algo" not in summary_df.columns:
        raise ValueError("Colonne 'algo' manquante")

    algos = sorted(a for a in summary_df["algo"].dropna().unique().tolist())
    if ref_algo not in algos:
        # fallback to first algo
        ref_algo = algos[0] if algos else ref_algo

    # Pair by seed within the same scenario/bs_mode if present
    pair_keys = [c for c in ["scenario", "seed", "bs_mode", "sink_x", "sink_y"] if c in summary_df.columns]

    ref_df = summary_df[summary_df["algo"] == ref_algo].copy()
    rows: List[Dict] = []

    for baseline in algos:
        if baseline == ref_algo:
            continue

        base_df = summary_df[summary_df["algo"] == baseline].copy()

        if pair_keys:
            merged = ref_df[pair_keys + metrics].merge(
                base_df[pair_keys + metrics],
                on=pair_keys,
                how="inner",
                suffixes=("_ref", "_base"),
            )
        else:
            min_len = min(len(ref_df), len(base_df))
            merged = pd.DataFrame()
            for m in metrics:
                merged[m + "_ref"] = ref_df[m].iloc[:min_len].to_numpy()
                merged[m + "_base"] = base_df[m].iloc[:min_len].to_numpy()

        for metric in metrics:
            x = merged[metric + "_ref"].to_numpy(dtype=float)
            y = merged[metric + "_base"].to_numpy(dtype=float)
            w_plus, p, r_rb, n_eff = _wilcoxon_stats(x, y)
            rows.append(
                {
                    "metric": metric,
                    "ref_algo": ref_algo,
                    "baseline": baseline,
                    "algo": baseline,
                    "n": n_eff,
                    "W_plus": w_plus,
                    "p_value": p,
                    "r_rb": r_rb,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    if correction == "holm":
        out["p_holm"] = _holm_adjust(out["p_value"].to_numpy(dtype=float))
    else:
        out["p_holm"] = np.nan

    return out.sort_values(["metric", "p_value"], ascending=[True, True]).reset_index(drop=True)


@dataclass(frozen=True)
class ReportPaths:
    summary: Path
    timeseries: Path
    out: Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Excel report (summary stats + Wilcoxon/Holm + r_rb).")
    parser.add_argument("--summary", required=True, help="Path to *_ALL_summary.csv")
    parser.add_argument("--timeseries", required=True, help="Path to *_ALL_timeseries.csv (kept for traceability)")
    parser.add_argument("--out", required=True, help="Output .xlsx")
    parser.add_argument(
        "--wilcoxon_mode",
        choices=["baseline_fixed", "ref_vs_each"],
        default="baseline_fixed",
        help="Wilcoxon mode. 'ref_vs_each' uses d_i = metric(ref) - metric(baseline) for each baseline.",
    )
    parser.add_argument("--baseline", default="Greedy", help="Baseline algorithm for Wilcoxon (baseline_fixed mode)")
    parser.add_argument("--ref_algo", default="FSS", help="Reference algorithm for Wilcoxon (ref_vs_each mode)")
    parser.add_argument("--correction", choices=["holm", "none"], default="holm")

    args = parser.parse_args()

    paths = ReportPaths(summary=Path(args.summary), timeseries=Path(args.timeseries), out=Path(args.out))

    if not paths.summary.exists():
        raise SystemExit(f"Missing summary CSV: {paths.summary}")
    if not paths.timeseries.exists():
        raise SystemExit(f"Missing timeseries CSV: {paths.timeseries}")

    summary_df = pd.read_csv(paths.summary)
    ts_df = pd.read_csv(paths.timeseries)

    stats_df = build_summary_stats(summary_df)
    if str(args.wilcoxon_mode) == "ref_vs_each":
        wil_df = build_wilcoxon_table_ref_vs_each(summary_df, ref_algo=str(args.ref_algo), correction=str(args.correction))
    else:
        wil_df = build_wilcoxon_table(summary_df, baseline_algo=str(args.baseline), correction=str(args.correction))

    # Write Excel
    paths.out.parent.mkdir(parents=True, exist_ok=True)

    try:
        with pd.ExcelWriter(paths.out, engine="openpyxl") as writer:
            meta = pd.DataFrame(
                {
                    "key": [
                        "summary_csv",
                        "timeseries_csv",
                        "wilcoxon_mode",
                        "baseline",
                        "ref_algo",
                        "correction",
                    ],
                    "value": [
                        str(paths.summary),
                        str(paths.timeseries),
                        str(args.wilcoxon_mode),
                        str(args.baseline),
                        str(args.ref_algo),
                        str(args.correction),
                    ],
                }
            )
            meta.to_excel(writer, sheet_name="Meta", index=False)

            stats_df.to_excel(writer, sheet_name="SummaryStats", index=False)
            wil_df.to_excel(writer, sheet_name="Wilcoxon", index=False)

            # Also store raw column lists for convenience
            pd.DataFrame({"summary_columns": summary_df.columns.tolist()}).to_excel(writer, sheet_name="Columns", index=False)

    except ModuleNotFoundError as e:
        raise SystemExit(
            "openpyxl est requis pour écrire .xlsx. Ajoutez-le via 'pip install openpyxl' ou mettez à jour requirements.txt"
        ) from e

    print(f"Wrote Excel: {paths.out}")
    print(f"  SummaryStats rows: {len(stats_df)}")
    print(f"  Wilcoxon rows: {len(wil_df)}")
    print(f"  Timeseries rows (loaded): {len(ts_df)}")


if __name__ == "__main__":
    main()
