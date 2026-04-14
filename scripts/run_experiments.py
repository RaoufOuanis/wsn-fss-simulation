import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import argparse
from wsn.experiments.runner import Scenario, run_experiments


def _parse_kv_overrides(raw: str) -> dict:
    """Parse overrides of form 'k1=v1,k2=v2' with best-effort numeric casting."""
    out: dict = {}
    s = str(raw or "").strip()
    if not s:
        return out
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        if "=" not in p:
            raise ValueError(f"Invalid override '{p}' (expected key=value)")
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"Invalid override '{p}' (empty key)")

        # Best-effort casting: bool, int, float, else string
        vl = v.lower()
        if vl in {"true", "false"}:
            out[k] = (vl == "true")
        else:
            try:
                if any(ch in v for ch in [".", "e", "E"]):
                    out[k] = float(v)
                else:
                    out[k] = int(v)
            except Exception:
                out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="results")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--max_rounds", type=int, default=5000)
    parser.add_argument(
        "--base_seed",
        type=int,
        default=0,
        help="Base seed for experiments (run k => seed = base_seed + k)",
    )
    parser.add_argument(
        "--algos",
        type=str,
        default="FSS,PSO,GWO,ABC,LEACH,HEED,SEP,Greedy,EEM_LEACH_ABC",
        help="Comma-separated list of algorithms to run.",
    )

    # Controlled revision experiments
    parser.add_argument(
        "--fss_kmax",
        type=int,
        default=None,
        help="If set, overrides FSSParams.Kmax (e.g., 20 to harmonize CH caps).",
    )
    parser.add_argument(
        "--fss_overrides",
        type=str,
        default="",
        help="Extra FSS overrides as 'k=v,...' (e.g., 'n_iter=60,Lmax=0').",
    )
    parser.add_argument(
        "--fit_overrides",
        type=str,
        default="",
        help="Fitness overrides as 'k=v,...' (e.g., 'r_tx=50,lam=1.0,w_relay=0.05').",
    )

    parser.add_argument("--only_s1_100", action="store_true")
    parser.add_argument("--only_s1_200", action="store_true")
    parser.add_argument("--only_s2_100", action="store_true")
    
    parser.add_argument(
        "--bs",
        type=str,
        default="center",
        choices=["center", "corner"],
        help="Base Station position mode (center or corner). Default: center",
    )



    args = parser.parse_args()

    # Scenarios
    s1_100 = Scenario(name="S1_100", n_nodes=100, heterogenous=False)
    s1_200 = Scenario(name="S1_200", n_nodes=200, heterogenous=False)
    s2_100 = Scenario(name="S2_100", n_nodes=100, heterogenous=True)

    if args.only_s1_100:
        scenarios = [s1_100]
    elif args.only_s1_200:
        scenarios = [s1_200]
    elif args.only_s2_100:
        scenarios = [s2_100]
    else:
        scenarios = [s1_100, s1_200, s2_100]

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]

    fss_over = _parse_kv_overrides(args.fss_overrides)
    if args.fss_kmax is not None:
        fss_over["Kmax"] = int(args.fss_kmax)

    fit_over = _parse_kv_overrides(args.fit_overrides)

    summary_df, history_df = run_experiments(
        scenarios=scenarios,
        algos=algos,
        n_runs=args.runs,
        max_rounds=args.max_rounds,
        base_seed=args.base_seed,
        save_prefix=args.prefix,
        bs_mode=args.bs,
        fss_params_overrides=(fss_over or None),
        fitness_params_overrides=(fit_over or None),
    )

    print("Summary:")
    print(summary_df.head())
    print("\nTimeseries:")
    print(history_df.head())


if __name__ == "__main__":
    main()
