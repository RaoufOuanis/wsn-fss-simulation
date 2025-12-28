from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from typing import List

from wsn.plot_style import algo_color, direction_phrase, display_algo_name, is_central_algo, metric_display_name


def compare_algorithms(df: pd.DataFrame, metric: str, ref_algo: str = "FSS") -> pd.DataFrame:
    """
    For each other algo, run paired Wilcoxon with reference algo.
    """
    algos = df["algo"].unique().tolist()
    algos = [a for a in algos if a != ref_algo]
    rows = []
    for algo in algos:
        df_ref = df[df["algo"] == ref_algo].sort_values(by=["scenario", "seed"])
        df_cmp = df[df["algo"] == algo].sort_values(by=["scenario", "seed"])
        x = df_ref[metric].values
        y = df_cmp[metric].values
        stat, p = wilcoxon(x, y)
        rows.append({"metric": metric, "algo": algo, "p_value": p})
    return pd.DataFrame(rows)


def boxplot_metric(df: pd.DataFrame, metric: str, scenarios: List[str] | None = None):
    if scenarios is None:
        scenarios = df["scenario"].unique().tolist()
    for sc in scenarios:
        sub = df[df["scenario"] == sc]
        # Keep central algo first in order
        algo_order = sorted(sub["algo"].astype(str).unique().tolist(), key=lambda a: (0 if is_central_algo(a) else 1, str(a)))
        sub = sub.copy()
        sub["algo"] = pd.Categorical(sub["algo"].astype(str), categories=algo_order, ordered=True)

        fig, ax = plt.subplots(figsize=(10, 4.8))
        sub.boxplot(column=metric, by="algo", ax=ax, patch_artist=True)

        # Color each box consistently
        for i, box in enumerate(ax.artists):
            a = algo_order[i] if i < len(algo_order) else ""
            box.set_facecolor(algo_color(a))
            if is_central_algo(a):
                box.set_edgecolor("black")
                box.set_linewidth(2.0)

        ax.set_title(f"{metric_display_name(metric)} - {sc}")
        fig.suptitle("")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel(metric_display_name(metric))
        ax.set_xticklabels([display_algo_name(a) for a in algo_order], rotation=30, ha="right")

        # Legend (required)
        from matplotlib.patches import Patch

        handles = [
            Patch(
                facecolor=algo_color(a),
                edgecolor=("black" if is_central_algo(a) else "none"),
                label=display_algo_name(a),
            )
            for a in algo_order
        ]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

        # Explicit direction hint
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
        plt.show()
