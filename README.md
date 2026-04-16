# WSN Clustering Simulator

Wireless Sensor Network (WSN) clustering simulator with per-round cluster-head (CH) selection via optimization algorithms, and a reproducible CSV → Excel → plots reporting pipeline.

This repo supports both single-hop and multi-hop CH→sink delivery (enabled by default in the fitness function).

## Installation (Windows)

```powershell
cd C:\wsn-project
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run (GUI)

```powershell
python runner.py
```

GUI default algorithm set (S1 and S2):

`FSS,PSO,GWO,ABC,SO,GJO,EMOGJO,ESOGJO,LEACH,HEED,SEP`

## Run (CLI)

```powershell
python -m scripts.run_experiments --prefix results --runs 30 --max_rounds 2500 --bs center
python -m scripts.merge_csvs --prefix results --out_dir .
python -m scripts.report_excel --summary results_ALL_summary.csv --timeseries results_ALL_timeseries.csv --out results_report.xlsx
python -m scripts.plot_from_excel --excel results_report.xlsx --out_dir plots --metric throughput --plots mean_bar
```

CLI notes:

- Scenarios supported by `scripts.run_experiments`: `S1_100`, `S1_200`, `S2_100`.
- Filter to a single scenario: `--only_s1_100`, `--only_s1_200`, `--only_s2_100`.
- Choose algorithms: `--algos <comma-separated>`.
  - Recommended (same as GUI defaults): `FSS,PSO,GWO,ABC,SO,GJO,EMOGJO,ESOGJO,LEACH,HEED,SEP`
  - Optional extras / ablations (when needed): `Greedy`, `EMOGJO_paperCH`, `FSS_noPhase2`, `FSS_noEnergy`, `FSS_noLS`

---

## Repository layout

- `wsn/`: simulator core (network model, energy, fitness, repair)
- `wsn/algorithms/`: optimizers and protocol baselines
  - optimizers: FSS, PSO, GWO, ABC, SO, GJO, EMOGJO, ESOGJO
  - protocols: LEACH / HEED / SEP / EEM-LEACH-ABC
- `wsn/experiments/runner.py`: experiment loop (produces CSV)
- `scripts/`: CLI tools (merge, Excel report, plots)
- `runner.py`: Tkinter GUI (end-to-end orchestration)
- `app/`: Streamlit UI (optional)

---

## Pipeline (what each step produces)

1) `scripts.run_experiments`
    - writes `<prefix>_summary.csv` (one row per seed × algorithm)
    - writes `<prefix>_timeseries.csv` (one row per round)

2) `scripts.merge_csvs`
    - merges suffix-split runs into `*_ALL_summary.csv` and `*_ALL_timeseries.csv`
    - de-duplicates and keeps the most complete/most recent row

3) `scripts.report_excel`
    - loads the “ALL” CSV files
    - writes `*_report.xlsx` (stats + Wilcoxon + stable tables)

4) `scripts.plot_from_excel`
    - generates PNG figures from the Excel report

---

## Outputs

### CSV

- `*_summary.csv`: one row per `(scenario, algo, seed)`
- `*_timeseries.csv`: per-round time series
- `*_ALL_summary.csv` / `*_ALL_timeseries.csv`: merged versions (recommended for reporting)

### Excel

The report includes (main sheets):

- `Meta`: provenance (CSV paths, Wilcoxon options)
- `SummaryStats`: mean/std per algorithm for all numeric columns
- `MainTable`: stable “core” table
- `MultiHopTable`: stable “multi-hop diagnostics” table
- `Wilcoxon`: Wilcoxon signed-rank + optional Holm correction
- `MetricCatalog`: label + direction (higher/lower is better)

---

## Metrics

### Core metrics (kept comparable)

- `throughput`: delivered reports (primary metric)
- `R_last`: last round with at least 1 delivered report
- `FND`, `HND`, `LND`: first/half/last node death
- `avg_energy`: mean residual energy over simulated rounds
- `avg_cpu_time`: average CPU time per round (CH selection)
- `avg_nfe`: average number of fitness evaluations per round

### Multi-hop diagnostics (additive)

Added to analyze multi-hop behavior without breaking core metrics:

- `pkt_hops`: total successful packet-hops (CH→CH + CH→sink)
- `mh_avg_path_hops`: average CH→sink hop count (Dijkstra paths)
- `mh_q_max`: max relay load hotspot
- `mh_jain_q`: Jain fairness of relay load
- `energy_per_report`: energy spent per delivered report

Compatibility: `pkts_to_sink*` is preserved as a legacy single-hop proxy.

---

## Multi-hop vs single-hop

Multi-hop is enabled by default via `FitnessParams.multihop = True` (see `wsn/fitness.py`).

- Repair enforces CH→sink connectivity under `r_tx`.
- CH→sink routing uses Dijkstra on a CH-only graph + sink (see `wsn/multihop.py`).

To run strict single-hop experiments, disable multi-hop consistently in your experiment protocol.

---

## Fairness and NFE (important)

For “strict fairness” comparisons in this repo, population-based optimizers are evaluated under:

- the same discrete decoding rule (`decode_topk_bounded`),
- the same objective (`fitness()`), which includes deterministic `Repair_t`,
- a budget-matched iteration count (`n_iter`) and consistent population sizes.

`avg_nfe` is a useful cost proxy, but its definition differs:

- PSO/GWO/SO/GJO/EMOGJO/ESOGJO: ~ `n_iter * pop_size` (1 `fitness()` call per individual per iteration)
- ABC: instrumented to count real `fitness()` calls
- FSS: counts effective evaluations (cache misses); it also exposes `nfe_requests` (hits + misses)

---

## Tests

```powershell
python -m pytest -q
```

---

## Streamlit (optional web UI)

The Streamlit app can:

- run an interactive simulation round-by-round (**Interactive**)
- compare multiple algorithms with matched seeds (**Compare**)
- export CSV outputs and a reproducibility snapshot

### Run

```powershell
python -m scripts.run_app
```

This launches `streamlit run app/app_streamlit.py` with `PYTHONPATH` configured so that `wsn/` is importable.

Default URL: `http://localhost:8501`.

### Key sidebar parameters

- **Scenario**: preset (S1_100/S1_200/S2_100) or Custom
- **Simulation control**: base seed, max rounds
- **Fitness & radio**: `Rc`, weights (w1,w2,w3, λ), radio parameters
- **Multi-hop CH→sink**: enable/disable, `r_tx`, `w_relay`

### Interactive mode

- Controls: *Rebuild network*, *Reset energies*, *Advance 1 round*, *Run to max rounds*
- Displays: topology + selected CHs + sink
- Curves: `alive`, `total_energy`
- Counters: throughput, legacy `pkts_to_sink`, multi-hop `pkt_hops`, and `FND/HND/LND` markers

### Compare mode

- Select algorithms, `n_runs` (matched seeds), `max_rounds`, optional early stop
- Produces per-seed summary and mean±std aggregates, plus mean curves across seeds

### Export / reproducibility

- Exports CSV for interactive and compare runs
- Saves a JSON configuration snapshot (scenario + fitness + radio + optimizer parameters)

