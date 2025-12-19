#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
from typing import List, Tuple, Optional

import pandas as pd


EXCLUDE_TOKENS = (
    "_ALL_",          # outputs: *_ALL_summary.csv
    "merged_ALL",     # outputs: merged_ALL_summary.csv
    "_by_algo",       # *_by_algo.csv
    "_per_run",       # *_per_run*.csv
    "_augmented",     # *_augmented.csv
)

SUMMARY_SUFFIX = "_summary.csv"
TS_SUFFIX = "_timeseries.csv"


def _is_excluded(path: str) -> bool:
    name = os.path.basename(path)
    return any(tok in name for tok in EXCLUDE_TOKENS)


def _find_matching(prefix: str, suffix: str) -> List[str]:
    pattern = f"{prefix}_*{suffix}"
    files = sorted(glob.glob(pattern))
    return [f for f in files if not _is_excluded(f)]


def _read_csvs(files: List[str]) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["_source_file"] = os.path.basename(f)
        df["_mtime"] = os.path.getmtime(f)
        df["_non_null"] = df.notna().sum(axis=1).astype(int)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True, sort=False)


def _dedup_best(df: pd.DataFrame, preferred_keys: List[str]) -> pd.DataFrame:
    """
    Dédoublonnage robuste :
      - conserve par clé la ligne la plus complète (max _non_null),
      - puis la plus récente (max _mtime).
    """
    if df.empty:
        return df

    keys = [k for k in preferred_keys if k in df.columns]
    if not keys:
        df = df.drop_duplicates()
        return df.drop(columns=["_non_null", "_mtime"], errors="ignore").reset_index(drop=True)

    # Sort so that the "best" row per key ends up last
    df = df.sort_values(
        keys + ["_non_null", "_mtime"],
        ascending=[True] * len(keys) + [True, True],
    )
    df = df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    return df.drop(columns=["_non_null", "_mtime"], errors="ignore")


def _sort(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if df.empty:
        return df

    if kind == "summary":
        cols = [c for c in ["scenario", "bs_mode", "algo", "seed"] if c in df.columns]
    else:
        cols = [c for c in ["scenario", "bs_mode", "algo", "seed", "round"] if c in df.columns]

    return df.sort_values(cols).reset_index(drop=True) if cols else df


def merge_one_prefix(prefix: str, out_prefix: Optional[str] = None, dry_run: bool = False) -> Tuple[str, str]:
    if out_prefix is None:
        out_prefix = f"{prefix}_ALL"

    summary_files = _find_matching(prefix, SUMMARY_SUFFIX)
    ts_files = _find_matching(prefix, TS_SUFFIX)

    print(f"[merge] prefix={prefix}")
    print(f"  Found {len(summary_files)} summary files")
    for f in summary_files:
        print(f"   - {f}")
    print(f"  Found {len(ts_files)} timeseries files")
    for f in ts_files:
        print(f"   - {f}")

    if not summary_files and not ts_files:
        raise SystemExit(f"[merge] No matching files for prefix '{prefix}'")

    summary_df = _read_csvs(summary_files)
    ts_df = _read_csvs(ts_files)

    summary_key = ["scenario", "algo", "seed", "bs_mode", "sink_x", "sink_y"]
    ts_key = ["scenario", "algo", "seed", "round", "bs_mode", "sink_x", "sink_y"]

    summary_df = _dedup_best(summary_df, summary_key)
    ts_df = _dedup_best(ts_df, ts_key)

    summary_df = _sort(summary_df, "summary")
    ts_df = _sort(ts_df, "timeseries")

    summary_out = f"{out_prefix}{SUMMARY_SUFFIX}"
    ts_out = f"{out_prefix}{TS_SUFFIX}"

    print(f"  Merged summary rows: {len(summary_df)}")
    print(f"  Merged timeseries rows: {len(ts_df)}")

    if dry_run:
        print("  Dry run: no files written.")
        return summary_out, ts_out

    summary_df.to_csv(summary_out, index=False)
    ts_df.to_csv(ts_out, index=False)
    print(f"  Wrote: {summary_out}")
    print(f"  Wrote: {ts_out}")
    return summary_out, ts_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge generated WSN CSVs without duplicates.")
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Merge ONLY this campaign prefix (e.g., paper_S2_100_corner_R2500).",
    )
    parser.add_argument(
        "--center_prefix",
        type=str,
        default=None,
        help="Prefix for center campaign (optional).",
    )
    parser.add_argument(
        "--corner_prefix",
        type=str,
        default=None,
        help="Prefix for corner campaign (optional).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Output directory (default: current dir).",
    )
    parser.add_argument("--dry_run", action="store_true", help="Show what would be merged, do not write files")
    parser.add_argument("--also_merge_global", action="store_true", help="If center+corner provided, merge both into one")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Case 1: single-prefix merge (what you want for S2 corner)
    if args.prefix:
        out_prefix = os.path.join(args.out_dir, f"{args.prefix}_ALL")
        merge_one_prefix(args.prefix, out_prefix=out_prefix, dry_run=args.dry_run)
        return

    # Case 2: center + corner merge
    if not args.center_prefix or not args.corner_prefix:
        raise SystemExit("Provide either --prefix OR both --center_prefix and --corner_prefix.")

    center_out_prefix = os.path.join(args.out_dir, f"{args.center_prefix}_ALL")
    corner_out_prefix = os.path.join(args.out_dir, f"{args.corner_prefix}_ALL")

    center_summary, center_ts = merge_one_prefix(args.center_prefix, out_prefix=center_out_prefix, dry_run=args.dry_run)
    corner_summary, corner_ts = merge_one_prefix(args.corner_prefix, out_prefix=corner_out_prefix, dry_run=args.dry_run)

    if args.also_merge_global and not args.dry_run:
        global_prefix = os.path.join(args.out_dir, "paper_S2_100_R2500_ALL")
        print(f"[merge] Creating global merge: {global_prefix}")

        s_df = pd.concat([pd.read_csv(center_summary), pd.read_csv(corner_summary)], ignore_index=True, sort=False)
        t_df = pd.concat([pd.read_csv(center_ts), pd.read_csv(corner_ts)], ignore_index=True, sort=False)

        s_df["_mtime"] = 0
        s_df["_non_null"] = s_df.notna().sum(axis=1).astype(int)
        t_df["_mtime"] = 0
        t_df["_non_null"] = t_df.notna().sum(axis=1).astype(int)

        s_df = _dedup_best(s_df, ["scenario", "algo", "seed", "bs_mode", "sink_x", "sink_y"])
        t_df = _dedup_best(t_df, ["scenario", "algo", "seed", "round", "bs_mode", "sink_x", "sink_y"])

        s_df = _sort(s_df, "summary")
        t_df = _sort(t_df, "timeseries")

        s_out = f"{global_prefix}{SUMMARY_SUFFIX}"
        t_out = f"{global_prefix}{TS_SUFFIX}"

        s_df.to_csv(s_out, index=False)
        t_df.to_csv(t_out, index=False)
        print(f"  Wrote: {s_out}")
        print(f"  Wrote: {t_out}")


if __name__ == "__main__":
    main()


    #   python -u -m scripts.merge_csvs --prefix paper_S1_100_center_R2200 --out_dir .

    #   python -u -m scripts.merge_csvs --prefix paper_S1_100_corner_R2200 --out_dir .
    
    #   python -u -m scripts.merge_csvs --prefix paper_S2_100_center_R2200 --out_dir .

    #   python -u -m scripts.merge_csvs --prefix paper_S2_100_corner_R2200 --out_dir .

    