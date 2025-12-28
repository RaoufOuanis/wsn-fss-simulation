#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
import time
import queue
import threading
import subprocess
import difflib
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd

from wsn.plot_style import ALGO_COLOR_MAP


WORKDIR_DEFAULT = Path(r"C:\wsn-project")

RUNS_FIXED = 5

# Keep a constrained dropdown list as requested.
ROUND_VALUES = [str(r) for r in range(200, 5001, 100)]

GUI_EXTRA_ALGOS = ["SO", "GJO", "EMOGJO", "EMOGJO_paperCH", "ESOGJO"]

S1_ALGOS_DEFAULT = "FSS,PSO,GWO,ABC,SO,GJO,EMOGJO,ESOGJO,LEACH,HEED,SEP"
S2_ALGOS_DEFAULT = "FSS,PSO,GWO,ABC,SO,GJO,EMOGJO,ESOGJO,LEACH,HEED,SEP"

JOB_SUFFIXES: List[Tuple[str, int]] = [
    ("s00_04", 0),
    ("s05_09", 5),
    ("s10_14", 10),
    ("s15_19", 15),
    ("s20_24", 20),
    ("s25_29", 25),
]

RUN_LINE_RE = re.compile(r"^\[run_experiments\]\s+(?P<scenario>[^-]+?)\s+-\s+(?P<algo>[^-]+?)\s+-\s+run\s+(?P<run>\d+)\s*$")


def _system_time_str() -> str:
    # Local system time, formatted for logs.
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _campaign_prefix_base(scenario_tag: str, bs_mode: str, rounds: int) -> str:
    # Keep exactly the naming style used in your files: paper_S1_100_center_R2200
    return f"paper_{scenario_tag}_{bs_mode}_R{rounds}"


def _make_jobs(prefix_base: str) -> List[Tuple[str, int]]:
    jobs: List[Tuple[str, int]] = []
    for suffix, base_seed in JOB_SUFFIXES:
        jobs.append((f"{prefix_base}_{suffix}", base_seed))
    return jobs


def _list_all_prefixes(workdir: Path) -> List[str]:
    # Detect prefixes that already have merged files.
    # Example: paper_S2_100_corner_R2200_ALL_summary.csv => prefix=paper_S2_100_corner_R2200
    out: set[str] = set()
    for p in workdir.glob("*_ALL_summary.csv"):
        name = p.name
        if name.endswith("_ALL_summary.csv"):
            out.add(name[: -len("_ALL_summary.csv")])
    return sorted(out)


def _list_excel_reports(workdir: Path) -> List[Path]:
    return sorted(workdir.glob("*_report.xlsx"))


def _ps_quote(s: str) -> str:
    # Single-quote for PowerShell. Escape ' by doubling.
    return "'" + s.replace("'", "''") + "'"


def _build_process_args(shell_kind: str, workdir: Path, argv: List[str]) -> List[str]:
    """Return a subprocess args list that runs `argv` inside chosen shell."""

    if shell_kind == "powershell":
        # powershell.exe -Command "& { Set-Location 'C:\wsn-project'; & 'python.exe' -u -m ... }"
        ps_cmd = "& { Set-Location " + _ps_quote(str(workdir)) + "; "
        # Use explicit quoting for each argument.
        # Ensure the PowerShell process exits with the child exit code.
        ps_cmd += "& " + " ".join(_ps_quote(a) for a in argv) + "; exit $LASTEXITCODE }"
        return [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            ps_cmd,
        ]

    if shell_kind == "cmd":
        # cmd.exe /c "cd /d C:\wsn-project && python -u -m ..."
        def cmd_quote(a: str) -> str:
            if not a or any(ch in a for ch in ' \t"'):
                return '"' + a.replace('"', '\\"') + '"'
            return a

        cmdline = f"cd /d {cmd_quote(str(workdir))} && " + " ".join(cmd_quote(a) for a in argv)
        return ["cmd.exe", "/c", cmdline]

    raise ValueError(f"Unknown shell_kind: {shell_kind}")


def _popen_capture(shell_kind: str, workdir: Path, argv: List[str]) -> subprocess.Popen:
    creationflags = 0
    startupinfo = None

    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW  # hide new console windows
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    args = _build_process_args(shell_kind, workdir, argv)

    return subprocess.Popen(
        args,
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        creationflags=creationflags,
        startupinfo=startupinfo,
    )


class RunnerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("WSN Runner")
        # The UI contains 2 stacked control panels (Simulation + Graphiques) plus logs.
        # A taller default prevents the 'Graphiques' menu from being clipped on startup.
        self.root.geometry("900x760")

        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None
        self._log_queue: queue.Queue[str] = queue.Queue()

        self.workdir_var = tk.StringVar(value=str(WORKDIR_DEFAULT))

        self.mode_var = tk.StringVar(value="simulate")  # simulate | excel | plots
        self.shell_var = tk.StringVar(value="powershell")  # powershell | cmd

        self.scenario_var = tk.StringVar(value="S1_100")
        self.bs_var = tk.StringVar(value="center")

        self.rounds_var = tk.StringVar(value="2200")
        self.algos_var = tk.StringVar(value=S1_ALGOS_DEFAULT)

        # Wilcoxon reference algorithm (protocol: compare FSS vs each baseline)
        self.baseline_algo_var = tk.StringVar(value="FSS")
        self.correction_var = tk.StringVar(value="holm")  # holm | none

        self.excel_prefix_var = tk.StringVar(value="")
        self.excel_file_var = tk.StringVar(value="")

        # Scenario comparison (multi-Excel)
        self.compare_excels: List[str] = []
        self.compare_excels_var = tk.StringVar(value="")
        self.compare_algos_var = tk.StringVar(value="FSS")
        self.compare_algos_help_var = tk.StringVar(value="")
        self._compare_algos_available: List[str] = []

        # Plot metrics
        # radio: either plot with one selected metric, or plot for ALL metrics found in the Excel.
        self.metric_mode_var = tk.StringVar(value="one")  # one | all | select
        self.metric_var = tk.StringVar(value="throughput")
        self._metric_values: List[str] = [
            "throughput",
            "FND",
            "HND",
            "LND",
            "delivered",
            "pkt_hops",
            "pkts_to_sink",
            "energy_per_report",
            "mh_avg_path_hops",
            "mh_q_max",
            "mh_jain_q",
            "avg_cpu_time",
            "nfe",
            "avg_pkt_hops_round",
        ]
        self._metric_radio_buttons: List[ttk.Radiobutton] = []
        self._metric_checkbuttons: List[ttk.Checkbutton] = []
        self._metric_select_vars: dict[str, tk.BooleanVar] = {}

        # Plot types (exactly as requested): Scatter, Bar, Box, Line, Hist, Heatmap
        self.plot_scatter = tk.BooleanVar(value=False)
        self.plot_bar = tk.BooleanVar(value=True)
        self.plot_box = tk.BooleanVar(value=False)
        self.plot_line = tk.BooleanVar(value=False)
        self.plot_hist = tk.BooleanVar(value=False)
        self.plot_heatmap = tk.BooleanVar(value=False)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="0%")
        self.eta_var = tk.StringVar(value="ETA: --")

        self._progress_total_units: int = 0
        self._progress_done_units: int = 0
        self._progress_t0: float = 0.0
        self._progress_lock = threading.Lock()

        self._build_ui()
        self._refresh_dynamic_lists()
        self._refresh_compare_algos_help()
        self._log_queue.put(f"[time] {_system_time_str()} - Application lancée.\n")
        self._tick_ui()

    @staticmethod
    def _split_csv_names(s: str) -> List[str]:
        parts = [p.strip() for p in str(s).split(",")]
        return [p for p in parts if p]

    @staticmethod
    def _format_short_list(items: List[str], max_items: int = 14) -> str:
        items = [str(x) for x in items if str(x).strip()]
        if len(items) <= max_items:
            return ", ".join(items)
        shown = items[:max_items]
        return ", ".join(shown) + f" … (+{len(items) - max_items})"

    def _refresh_compare_algos_help(self) -> None:
        # Prefer algos detected from selected Excel files; fallback to known repo keys.
        base_known = set(ALGO_COLOR_MAP.keys()) | set(GUI_EXTRA_ALGOS)
        base = sorted(base_known, key=lambda x: (x != "FSS", x.lower()))

        if self._compare_algos_available:
            disp = self._format_short_list(self._compare_algos_available)
            self.compare_algos_help_var.set(f"Disponibles (dans Excel sélectionnés): {disp}  |  Saisir séparés par virgules")
        else:
            disp = self._format_short_list(base)
            self.compare_algos_help_var.set(f"Noms courants: {disp}  |  Saisir séparés par virgules")

    def _detect_algos_from_excels(self, excels: List[str]) -> List[str]:
        found: set[str] = set()
        for f in excels:
            try:
                df = pd.read_excel(f, sheet_name="SummaryStats")
                if "algo" not in df.columns:
                    continue
                for a in df["algo"].dropna().astype(str).tolist():
                    a = a.strip()
                    if a:
                        found.add(a)
            except Exception:
                continue

        # Stable ordering: prefer known algos first, then others alpha.
        known = set(ALGO_COLOR_MAP.keys()) | set(GUI_EXTRA_ALGOS)
        known_part = sorted([a for a in found if a in known], key=lambda x: (x != "FSS", x.lower()))
        other_part = sorted([a for a in found if a not in known], key=lambda x: x.lower())
        return known_part + other_part

    def _validate_compare_algos_input(self, raw: str) -> str:
        """Validate and return a normalized comma-separated algos string."""

        parsed = self._split_csv_names(raw)
        if not parsed:
            raise ValueError("Liste d'algorithmes vide")

        # Deduplicate while preserving order (common user mistake: typing FSS twice)
        seen: set[str] = set()
        algos: List[str] = []
        for a in parsed:
            if a not in seen:
                algos.append(a)
                seen.add(a)

        allowed = set(self._compare_algos_available) if self._compare_algos_available else (set(ALGO_COLOR_MAP.keys()) | set(GUI_EXTRA_ALGOS))
        unknown = [a for a in algos if a not in allowed]
        if unknown:
            # Suggestions: closest matches among allowed.
            suggestions: List[str] = []
            allowed_list = sorted(allowed, key=lambda x: x.lower())
            for u in unknown:
                sugg = difflib.get_close_matches(u, allowed_list, n=3, cutoff=0.55)
                if sugg:
                    suggestions.append(f"{u} → {', '.join(sugg)}")
                else:
                    suggestions.append(f"{u} → (aucune suggestion)")

            avail_text = self._format_short_list(sorted(allowed, key=lambda x: (x != "FSS", x.lower())))
            msg = (
                "Algorithme(s) inconnu(s) dans 'Algorithmes (scénarios)':\n"
                + " - "
                + "\n - ".join(unknown)
                + "\n\nSuggestions:\n - "
                + "\n - ".join(suggestions)
                + "\n\nDisponibles:\n"
                + avail_text
            )
            raise ValueError(msg)

        # Preserve user order, but normalize spacing.
        return ",".join(algos)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Dossier de travail:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.workdir_var, width=70).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(top, text="Parcourir...", command=self._choose_workdir).grid(row=0, column=2, sticky="e")

        ttk.Label(top, text="Shell:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        shell_frame = ttk.Frame(top)
        shell_frame.grid(row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Radiobutton(shell_frame, text="PowerShell", value="powershell", variable=self.shell_var).pack(side="left")
        ttk.Radiobutton(shell_frame, text="CMD", value="cmd", variable=self.shell_var).pack(side="left", padx=(12, 0))

        ttk.Label(top, text="Mode:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        mode_frame = ttk.Frame(top)
        mode_frame.grid(row=2, column=1, sticky="w", pady=(8, 0))
        ttk.Radiobutton(mode_frame, text="Simulation → Merge → Excel → Graphiques", value="simulate", variable=self.mode_var, command=self._update_mode_visibility).pack(side="left")
        ttk.Radiobutton(mode_frame, text="Excel depuis CSV *_ALL_*", value="excel", variable=self.mode_var, command=self._update_mode_visibility).pack(side="left", padx=(12, 0))
        ttk.Radiobutton(mode_frame, text="Graphiques depuis Excel", value="plots", variable=self.mode_var, command=self._update_mode_visibility).pack(side="left", padx=(12, 0))

        top.columnconfigure(1, weight=1)

        # Main frames
        self.sim_frame = ttk.LabelFrame(self.root, text="Simulation", padding=10)
        self.excel_frame = ttk.LabelFrame(self.root, text="Générer Excel", padding=10)
        self.plots_frame = ttk.LabelFrame(self.root, text="Graphiques", padding=10)

        self.sim_frame.pack(fill="x", padx=10, pady=(0, 8))
        self.excel_frame.pack(fill="x", padx=10, pady=(0, 8))
        self.plots_frame.pack(fill="x", padx=10, pady=(0, 8))

        self._build_sim_frame()
        self._build_excel_frame()
        self._build_plots_frame()

        # Progress + controls
        bottom = ttk.Frame(self.root, padding=10)
        bottom.pack(fill="x")

        ttk.Progressbar(bottom, variable=self.progress_var, maximum=100.0).grid(row=0, column=0, sticky="we")
        ttk.Label(bottom, textvariable=self.progress_text_var, width=10).grid(row=0, column=1, padx=(8, 0))
        ttk.Label(bottom, textvariable=self.eta_var, width=18).grid(row=0, column=2, padx=(8, 0))

        btns = ttk.Frame(bottom)
        btns.grid(row=1, column=0, columnspan=3, sticky="we", pady=(8, 0))
        self.start_btn = ttk.Button(btns, text="Démarrer", command=self._on_start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="Arrêter", command=self._on_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Rafraîchir listes", command=self._refresh_dynamic_lists).pack(side="right")

        bottom.columnconfigure(0, weight=1)

        # Logs
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=10)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        # Keep logs readable but avoid pushing the controls off-screen.
        self.log_text = tk.Text(log_frame, height=8, wrap="none")
        self.log_text.pack(fill="both", expand=True)

        self._update_mode_visibility()

    def _build_sim_frame(self) -> None:
        ttk.Label(self.sim_frame, text="Scenario:").grid(row=0, column=0, sticky="w")
        scenario_cb = ttk.Combobox(self.sim_frame, textvariable=self.scenario_var, values=["S1_100", "S2_100"], state="readonly", width=10)
        scenario_cb.grid(row=0, column=1, sticky="w")
        scenario_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_scenario_change())

        ttk.Label(self.sim_frame, text="BS:").grid(row=0, column=2, sticky="w", padx=(14, 0))
        ttk.Combobox(self.sim_frame, textvariable=self.bs_var, values=["center", "corner"], state="readonly", width=10).grid(row=0, column=3, sticky="w")

        ttk.Label(self.sim_frame, text="Runs (fixé):").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Label(self.sim_frame, text=f"{RUNS_FIXED}  (6 jobs → 30 seeds)").grid(row=1, column=1, sticky="w", pady=(8, 0))

        ttk.Label(self.sim_frame, text="Rounds (200..5000):").grid(row=1, column=2, sticky="w", padx=(14, 0), pady=(8, 0))
        ttk.Combobox(self.sim_frame, textvariable=self.rounds_var, values=ROUND_VALUES, state="readonly", width=10).grid(row=1, column=3, sticky="w", pady=(8, 0))

        ttk.Label(self.sim_frame, text="Algorithmes (CSV):").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(self.sim_frame, textvariable=self.algos_var, width=60).grid(row=2, column=1, columnspan=3, sticky="we", pady=(8, 0))

        ttk.Label(self.sim_frame, text="Reference (Wilcoxon):").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(self.sim_frame, textvariable=self.baseline_algo_var, width=12).grid(row=3, column=1, sticky="w", pady=(8, 0))

        ttk.Label(self.sim_frame, text="Correction:").grid(row=3, column=2, sticky="w", padx=(14, 0), pady=(8, 0))
        ttk.Combobox(self.sim_frame, textvariable=self.correction_var, values=["holm", "none"], state="readonly", width=10).grid(row=3, column=3, sticky="w", pady=(8, 0))

        self.sim_frame.columnconfigure(1, weight=1)

    def _build_excel_frame(self) -> None:
        ttk.Label(self.excel_frame, text="Préfixe détecté (*_ALL_summary.csv):").grid(row=0, column=0, sticky="w")
        self.excel_prefix_cb = ttk.Combobox(self.excel_frame, textvariable=self.excel_prefix_var, values=[], state="readonly", width=45)
        self.excel_prefix_cb.grid(row=0, column=1, sticky="w")

        ttk.Button(self.excel_frame, text="Utiliser ce préfixe", command=self._set_excel_from_prefix).grid(row=0, column=2, padx=(8, 0))

        ttk.Label(self.excel_frame, text="Fichier Excel:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(self.excel_frame, textvariable=self.excel_file_var, width=60).grid(row=1, column=1, sticky="we", pady=(8, 0))
        ttk.Button(self.excel_frame, text="Choisir...", command=self._choose_excel_out).grid(row=1, column=2, padx=(8, 0), pady=(8, 0))

        ttk.Label(self.excel_frame, text="Correction:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(self.excel_frame, textvariable=self.correction_var, values=["holm", "none"], state="readonly", width=10).grid(row=2, column=1, sticky="w", pady=(8, 0))

        ttk.Label(self.excel_frame, text="Reference:").grid(row=2, column=2, sticky="w", padx=(14, 0), pady=(8, 0))
        ttk.Entry(self.excel_frame, textvariable=self.baseline_algo_var, width=12).grid(row=2, column=3, sticky="w", pady=(8, 0))

        self.excel_frame.columnconfigure(1, weight=1)

    def _build_plots_frame(self) -> None:
        ttk.Label(self.plots_frame, text="Métriques:").grid(row=0, column=0, sticky="w")

        metric_mode = ttk.Frame(self.plots_frame)
        metric_mode.grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(metric_mode, text="Une métrique", value="one", variable=self.metric_mode_var, command=self._refresh_metric_state).pack(side="left")
        ttk.Radiobutton(metric_mode, text="Toutes", value="all", variable=self.metric_mode_var, command=self._refresh_metric_state).pack(side="left", padx=(12, 0))
        ttk.Radiobutton(metric_mode, text="Sélection", value="select", variable=self.metric_mode_var, command=self._refresh_metric_state).pack(side="left", padx=(12, 0))

        # Scrollable radio list so Tkinter never "hides" items.
        metric_box = ttk.Frame(self.plots_frame)
        metric_box.grid(row=1, column=0, columnspan=4, sticky="we", pady=(6, 0))
        metric_box.columnconfigure(0, weight=1)

        self._metric_canvas = tk.Canvas(metric_box, height=72, highlightthickness=0)
        self._metric_canvas.grid(row=0, column=0, sticky="we")
        self._metric_scroll = ttk.Scrollbar(metric_box, orient="vertical", command=self._metric_canvas.yview)
        self._metric_scroll.grid(row=0, column=1, sticky="ns")
        self._metric_canvas.configure(yscrollcommand=self._metric_scroll.set)

        self._metric_list_frame = ttk.Frame(self._metric_canvas)
        self._metric_canvas_window = self._metric_canvas.create_window((0, 0), window=self._metric_list_frame, anchor="nw")

        def _on_metric_frame_config(_e=None):
            self._metric_canvas.configure(scrollregion=self._metric_canvas.bbox("all"))

        def _on_metric_canvas_config(e):
            # Keep inner frame width synced to canvas width.
            self._metric_canvas.itemconfigure(self._metric_canvas_window, width=e.width)

        self._metric_list_frame.bind("<Configure>", _on_metric_frame_config)
        self._metric_canvas.bind("<Configure>", _on_metric_canvas_config)

        self._rebuild_metric_widgets()

        checks = ttk.Frame(self.plots_frame)
        checks.grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))

        items = [
            ("Scatter", self.plot_scatter),
            ("Bar", self.plot_bar),
            ("Box", self.plot_box),
            ("Line", self.plot_line),
            ("Hist", self.plot_hist),
            ("Heatmap", self.plot_heatmap),
        ]
        # 2 lignes × 3 colonnes pour éviter que le menu soit caché.
        for i, (label, var) in enumerate(items):
            r = i // 3
            c = i % 3
            ttk.Checkbutton(checks, text=label, variable=var).grid(row=r, column=c, sticky="w", padx=(0 if c == 0 else 14, 0), pady=(0 if r == 0 else 4, 0))

        ttk.Label(self.plots_frame, text="Excel existant:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.excel_exist_cb = ttk.Combobox(self.plots_frame, values=[], state="readonly", width=60)
        self.excel_exist_cb.grid(row=3, column=1, columnspan=2, sticky="we", pady=(8, 0))
        ttk.Button(self.plots_frame, text="Choisir...", command=self._choose_excel_in).grid(row=3, column=3, padx=(8, 0), pady=(8, 0))

        # Multi-Excel scenario comparison
        ttk.Label(self.plots_frame, text="Comparer scénarios (multi-Excel):").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(self.plots_frame, textvariable=self.compare_excels_var, state="readonly", width=60).grid(
            row=4, column=1, columnspan=2, sticky="we", pady=(8, 0)
        )
        ttk.Button(self.plots_frame, text="Choisir...", command=self._choose_excels_multi).grid(row=4, column=3, padx=(8, 0), pady=(8, 0))

        ttk.Label(self.plots_frame, text="Algorithmes (scénarios):").grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(self.plots_frame, textvariable=self.compare_algos_var, width=60).grid(row=5, column=1, columnspan=3, sticky="we", pady=(8, 0))

        ttk.Label(self.plots_frame, textvariable=self.compare_algos_help_var).grid(
            row=6,
            column=1,
            columnspan=3,
            sticky="w",
            pady=(2, 0),
        )

        self.plots_frame.columnconfigure(1, weight=1)

    def _choose_excels_multi(self) -> None:
        workdir = Path(self.workdir_var.get())
        files = filedialog.askopenfilenames(
            initialdir=str(workdir),
            filetypes=[("Excel", "*.xlsx")],
        )
        if not files:
            return

        # Keep stable order as selected by user
        self.compare_excels = [str(f) for f in files]

        self._compare_algos_available = self._detect_algos_from_excels(self.compare_excels)
        self._refresh_compare_algos_help()

        # Display summary (avoid super long strings)
        names = [Path(f).name for f in self.compare_excels]
        disp = "; ".join(names)
        if len(disp) > 160:
            disp = disp[:157] + "..."
        self.compare_excels_var.set(f"{len(names)} fichier(s): {disp}")

    def _rebuild_metric_widgets(self) -> None:
        prev_metric = (self.metric_var.get() or "").strip()
        prev_checked = {k for k, v in self._metric_select_vars.items() if bool(v.get())}

        # Clear old widgets
        for rb in self._metric_radio_buttons:
            try:
                rb.destroy()
            except Exception:
                pass
        self._metric_radio_buttons.clear()

        for cb in self._metric_checkbuttons:
            try:
                cb.destroy()
            except Exception:
                pass
        self._metric_checkbuttons.clear()
        self._metric_select_vars.clear()

        vals = list(self._metric_values)
        if not vals:
            vals = ["throughput"]
            self._metric_values = vals

        # Prefer multi-hop traffic metric when available
        if prev_metric == "pkts_to_sink" and "pkt_hops" in vals:
            prev_metric = "pkt_hops"

        # Use a compact 2-column layout; scrollbar handles overflow.
        ncols = 2
        for i, m in enumerate(vals):
            r = i // ncols
            c = i % ncols

            # Radio for "one" mode
            rb = ttk.Radiobutton(self._metric_list_frame, text=str(m), value=str(m), variable=self.metric_var)
            rb.grid(row=r, column=c, sticky="w", padx=(0 if c == 0 else 18, 0), pady=2)
            self._metric_radio_buttons.append(rb)

            # Checkbutton for "select" mode (same grid position, different row offset)
            v = tk.BooleanVar(value=False)
            self._metric_select_vars[str(m)] = v
            cb = ttk.Checkbutton(self._metric_list_frame, text=str(m), variable=v)
            cb.grid(row=r + 1000, column=c, sticky="w", padx=(0 if c == 0 else 18, 0), pady=2)
            self._metric_checkbuttons.append(cb)

        # Restore previous selection when possible; otherwise choose a sensible default set.
        cur = prev_metric
        if cur not in vals:
            cur = vals[0]
        self.metric_var.set(cur)

        if prev_checked:
            for k in prev_checked:
                if k in self._metric_select_vars:
                    self._metric_select_vars[k].set(True)
        else:
            # Default plot set: core endpoints + multi-hop diagnostics (when present)
            core_defaults = [
                "throughput",
                "FND",
                "HND",
                "LND",
                "avg_cpu_time",
                "nfe",
            ]
            mh_defaults = [
                "pkt_hops" if "pkt_hops" in vals else "pkts_to_sink",
                "energy_per_report",
                "mh_avg_path_hops",
                "mh_q_max",
                "mh_jain_q",
            ]
            for k in core_defaults + mh_defaults:
                if k in self._metric_select_vars:
                    self._metric_select_vars[k].set(True)

            # In multi-hop mode, make pkt_hops the current metric if it exists.
            if "pkt_hops" in vals:
                self.metric_var.set("pkt_hops")

        self._refresh_metric_state()

    def _refresh_metric_state(self) -> None:
        mode = (self.metric_mode_var.get() or "one").strip()
        if mode == "one":
            # Show radios, hide checkboxes
            for rb in self._metric_radio_buttons:
                rb.grid()
                rb.configure(state="normal")
            for cb in self._metric_checkbuttons:
                cb.grid_remove()
        elif mode == "select":
            # Show checkboxes, hide radios
            for rb in self._metric_radio_buttons:
                rb.grid_remove()
            for cb in self._metric_checkbuttons:
                cb.grid()
                cb.configure(state="normal")
        else:
            # all: disable selection UI but keep it visible (radios hidden, checkboxes hidden)
            for rb in self._metric_radio_buttons:
                rb.grid_remove()
            for cb in self._metric_checkbuttons:
                cb.grid_remove()

    def _extract_metrics_from_excel(self, excel_path: Path) -> List[str]:
        try:
            stats = pd.read_excel(excel_path, sheet_name="SummaryStats")
        except Exception:
            return []

        metrics: List[str] = []
        for c in stats.columns:
            if isinstance(c, str) and c.endswith("_mean"):
                metrics.append(c[: -len("_mean")])

        # Keep stable order but unique
        seen: set[str] = set()
        out: List[str] = []
        for m in metrics:
            if m not in seen:
                out.append(m)
                seen.add(m)
        return out

    def _common_metrics_for_excels(self, excels: List[str]) -> List[str]:
        """Return metrics common to all provided Excel reports.

        Uses the metric list inferred from the SummaryStats sheet ("*_mean" columns).
        Order is preserved from the first Excel file.
        """

        if not excels:
            return []

        per_file: List[List[str]] = []
        for f in excels:
            try:
                ms = self._extract_metrics_from_excel(Path(f))
            except Exception:
                ms = []
            per_file.append(ms)

        if not per_file or not per_file[0]:
            return []

        common = set(per_file[0])
        for ms in per_file[1:]:
            common &= set(ms)

        # Preserve order from the first excel.
        return [m for m in per_file[0] if m in common]

    def _refresh_metric_dropdown(self) -> None:
        # Prefer metrics inferred from a selected Excel report (most accurate)
        excel_path = (self.excel_exist_cb.get() or "").strip()
        vals: List[str] = []
        if excel_path:
            p = Path(excel_path)
            if p.exists() and p.suffix.lower() == ".xlsx":
                vals = self._extract_metrics_from_excel(p)

        if not vals:
            vals = list(self._metric_values)

        self._metric_values = vals
        self._rebuild_metric_widgets()

    def _update_mode_visibility(self) -> None:
        mode = self.mode_var.get()
        self.sim_frame.configure()
        self.excel_frame.configure()
        self.plots_frame.configure()

        self.sim_frame.pack_forget()
        self.excel_frame.pack_forget()
        self.plots_frame.pack_forget()

        if mode == "simulate":
            self.sim_frame.pack(fill="x", padx=10, pady=(0, 8))
            self.plots_frame.pack(fill="x", padx=10, pady=(0, 8))
        elif mode == "excel":
            self.excel_frame.pack(fill="x", padx=10, pady=(0, 8))
        elif mode == "plots":
            self.plots_frame.pack(fill="x", padx=10, pady=(0, 8))

    def _on_scenario_change(self) -> None:
        if self.scenario_var.get() == "S1_100":
            self.algos_var.set(S1_ALGOS_DEFAULT)
        else:
            self.algos_var.set(S2_ALGOS_DEFAULT)

    def _choose_workdir(self) -> None:
        d = filedialog.askdirectory(initialdir=self.workdir_var.get() or str(WORKDIR_DEFAULT))
        if d:
            self.workdir_var.set(d)
            self._refresh_dynamic_lists()

    def _choose_excel_out(self) -> None:
        workdir = Path(self.workdir_var.get())
        f = filedialog.asksaveasfilename(
            initialdir=str(workdir),
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")],
        )
        if f:
            self.excel_file_var.set(f)

    def _choose_excel_in(self) -> None:
        workdir = Path(self.workdir_var.get())
        f = filedialog.askopenfilename(
            initialdir=str(workdir),
            filetypes=[("Excel", "*.xlsx")],
        )
        if f:
            self.excel_exist_cb.set(f)
            self._refresh_metric_dropdown()

    def _set_excel_from_prefix(self) -> None:
        pref = self.excel_prefix_var.get().strip()
        if not pref:
            return
        workdir = Path(self.workdir_var.get())
        out = workdir / f"{pref}_report.xlsx"
        self.excel_file_var.set(str(out))

    def _refresh_dynamic_lists(self) -> None:
        workdir = Path(self.workdir_var.get())
        prefixes = _list_all_prefixes(workdir)
        self.excel_prefix_cb["values"] = prefixes
        if prefixes and not self.excel_prefix_var.get():
            self.excel_prefix_var.set(prefixes[0])

        excel_reports = [str(p) for p in _list_excel_reports(workdir)]
        self.excel_exist_cb["values"] = excel_reports
        if excel_reports and not self.excel_exist_cb.get():
            self.excel_exist_cb.set(excel_reports[0])

        self._refresh_metric_dropdown()

    def _append_log(self, line: str) -> None:
        self.log_text.insert("end", line)
        if not line.endswith("\n"):
            self.log_text.insert("end", "\n")
        self.log_text.see("end")

    def _tick_ui(self) -> None:
        try:
            while True:
                line = self._log_queue.get_nowait()
                self._append_log(line)
        except queue.Empty:
            pass

        self.root.after(120, self._tick_ui)

    def _set_running(self, running: bool) -> None:
        self.start_btn.configure(state=("disabled" if running else "normal"))
        self.stop_btn.configure(state=("normal" if running else "disabled"))

    def _on_start(self) -> None:
        if self._worker and self._worker.is_alive():
            return

        workdir = Path(self.workdir_var.get())
        if not workdir.exists():
            messagebox.showerror("Erreur", f"Dossier introuvable: {workdir}")
            return

        self._stop_event.clear()
        self.progress_var.set(0.0)
        self.progress_text_var.set("0%")
        self.eta_var.set("ETA: --")

        self._progress_total_units = 0
        self._progress_done_units = 0
        self._progress_t0 = 0.0

        mode = self.mode_var.get()
        self._worker = threading.Thread(target=self._run_worker, args=(mode,), daemon=True)
        self._worker.start()
        self._set_running(True)

    def _on_stop(self) -> None:
        self._stop_event.set()
        p = self._proc
        if p and p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass

    def _run_worker(self, mode: str) -> None:
        t0 = time.perf_counter()
        try:
            self._log_queue.put(f"[time] {_system_time_str()} - Début opération: mode={mode}.\n")
            if mode == "simulate":
                self._run_simulate_pipeline()
            elif mode == "excel":
                self._run_excel_only()
            elif mode == "plots":
                self._run_plots_only()
            else:
                raise ValueError(f"Unknown mode: {mode}")

            self._log_queue.put(f"[time] {_system_time_str()} - Fin opération: mode={mode}.\n")
            self._log_queue.put("[runner] Terminé.\n")
        except Exception as e:
            self._log_queue.put(f"[runner] ERREUR: {e}\n")
            messagebox.showerror("Erreur", str(e))
            self._log_queue.put(f"[time] {_system_time_str()} - Fin opération (avec erreur): mode={mode}.\n")
        finally:
            _ = time.perf_counter() - t0
            self.root.after(0, lambda: self._set_running(False))

    def _set_progress_ui(self, pct: float, eta_s: Optional[int]) -> None:
        pct = max(0.0, min(100.0, float(pct)))

        def apply() -> None:
            self.progress_var.set(pct)
            self.progress_text_var.set(f"{pct:.1f}%")
            if eta_s is None:
                self.eta_var.set("ETA: --")
            else:
                self.eta_var.set(f"ETA: {eta_s//60:02d}:{eta_s%60:02d}")

        self.root.after(0, apply)

    def _compute_and_push_progress(self) -> None:
        total = int(self._progress_total_units)
        if total <= 0:
            return

        with self._progress_lock:
            done = int(self._progress_done_units)
            t0 = float(self._progress_t0)

        pct = 100.0 * float(done) / float(total)

        eta_s: Optional[int] = None
        if done > 0 and t0 > 0:
            elapsed = max(0.001, time.time() - t0)
            rate = done / elapsed
            remain = max(0, total - done)
            eta_s = int(remain / max(1e-9, rate))

        self._set_progress_ui(pct, eta_s)

    def _run_cmd_with_progress(self, argv: List[str], count_run_lines: bool = False) -> int:
        shell_kind = self.shell_var.get()
        workdir = Path(self.workdir_var.get())

        self._log_queue.put(f"[time] {_system_time_str()} - Début cmd.\n")
        self._log_queue.put(f"[cmd] {' '.join(argv)}\n")
        proc = _popen_capture(shell_kind, workdir, argv)
        self._proc = proc

        def bump_and_update() -> None:
            with self._progress_lock:
                self._progress_done_units += 1
            self._compute_and_push_progress()

        if proc.stdout is None:
            raise RuntimeError("stdout pipe missing")

        for line in proc.stdout:
            if self._stop_event.is_set():
                break

            line = line.rstrip("\n")
            self._log_queue.put(line)

            m = RUN_LINE_RE.match(line)
            if m and count_run_lines:
                # each printed run = one completed unit across all jobs
                bump_and_update()

        if self._stop_event.is_set() and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

        rc = proc.wait()
        self._proc = None
        self._log_queue.put(f"[time] {_system_time_str()} - Fin cmd (rc={int(rc)}).\n")
        return int(rc)

    def _run_parallel_jobs(self, jobs: List[Tuple[str, int]], base_argv_builder) -> None:
        """Run multiple jobs in parallel while aggregating progress from stdout."""

        procs: List[Tuple[str, subprocess.Popen]] = []
        threads: List[threading.Thread] = []

        def reader(prefix: str, proc: subprocess.Popen) -> None:
            try:
                if proc.stdout is None:
                    return

                for raw in proc.stdout:
                    if self._stop_event.is_set():
                        break

                    line = raw.rstrip("\n")
                    self._log_queue.put(line)

                    if RUN_LINE_RE.match(line):
                        with self._progress_lock:
                            self._progress_done_units += 1

            except Exception as e:
                self._log_queue.put(f"[runner] reader error for {prefix}: {e}\n")

        # start all processes
        for prefix, base_seed in jobs:
            if self._stop_event.is_set():
                raise RuntimeError("Arrêt demandé")

            argv = base_argv_builder(prefix, base_seed)
            self._log_queue.put(f"[time] {_system_time_str()} - Début job: {prefix} (base_seed={base_seed}).\n")
            self._log_queue.put(f"[runner] Start job: {prefix} (base_seed={base_seed})\n")
            proc = _popen_capture(self.shell_var.get(), Path(self.workdir_var.get()), argv)
            procs.append((prefix, proc))
            t = threading.Thread(target=reader, args=(prefix, proc), daemon=True)
            threads.append(t)
            t.start()

        # monitor until all finished
        while True:
            if self._stop_event.is_set():
                for _pref, p in procs:
                    if p.poll() is None:
                        try:
                            p.terminate()
                        except Exception:
                            pass
                raise RuntimeError("Arrêt demandé")

            self._compute_and_push_progress()

            alive = [p for _pref, p in procs if p.poll() is None]
            if not alive:
                break

            time.sleep(0.25)

        # join readers quickly
        for t in threads:
            t.join(timeout=0.5)

        # check return codes
        bad: List[Tuple[str, int]] = []
        for pref, p in procs:
            rc = int(p.poll() if p.poll() is not None else p.wait())
            self._log_queue.put(f"[time] {_system_time_str()} - Fin job: {pref} (rc={rc}).\n")
            if rc != 0:
                bad.append((pref, rc))

        if bad:
            msg = "; ".join(f"{pref}: rc={rc}" for pref, rc in bad)
            raise RuntimeError(f"Simulation échouée pour: {msg}")

    def _run_simulate_pipeline(self) -> None:
        scenario_tag = self.scenario_var.get()
        bs = self.bs_var.get()

        runs = RUNS_FIXED
        rounds = int(self.rounds_var.get())
        if rounds < 200 or rounds > 5000:
            raise ValueError("Rounds doit être entre 200 et 5000")

        algos = [a.strip() for a in self.algos_var.get().split(",") if a.strip()]
        if not algos:
            raise ValueError("Liste d'algorithmes vide")

        scenario_flag = "--only_s1_100" if scenario_tag == "S1_100" else "--only_s2_100"
        prefix_base = _campaign_prefix_base(scenario_tag, bs, rounds)
        jobs = _make_jobs(prefix_base)

        total_units = len(jobs) * runs * len(algos)

        self._progress_total_units = int(total_units)
        with self._progress_lock:
            self._progress_done_units = 0
            self._progress_t0 = time.time()

        self._log_queue.put(f"[runner] Simulation: {scenario_tag}, bs={bs}, runs={runs}, rounds={rounds}, algos={len(algos)}\n")
        self._log_queue.put(f"[runner] Total unités (scenario×algo×run): {total_units}\n")
        self._log_queue.put(f"[time] {_system_time_str()} - Début étape: Simulation.\n")

        def build_job_argv(prefix: str, base_seed: int) -> List[str]:
            return [
                sys.executable,
                "-u",
                "-m",
                "scripts.run_experiments",
                "--prefix",
                prefix,
                "--runs",
                str(runs),
                "--max_rounds",
                str(rounds),
                "--base_seed",
                str(base_seed),
                scenario_flag,
                "--bs",
                bs,
                "--algos",
                ",".join(algos),
            ]

        # Run the 6 jobs in parallel (equivalent to Start-Process loop).
        self._run_parallel_jobs(jobs, build_job_argv)

        self._log_queue.put(f"[time] {_system_time_str()} - Fin étape: Simulation.\n")

        # Merge
        if self._stop_event.is_set():
            raise RuntimeError("Arrêt demandé")

        merge_prefix = prefix_base
        self._log_queue.put(f"[runner] Merge: {merge_prefix}\n")
        self._log_queue.put(f"[time] {_system_time_str()} - Début étape: Merge.\n")

        merge_argv = [sys.executable, "-u", "-m", "scripts.merge_csvs", "--prefix", merge_prefix, "--out_dir", "."]
        rc = self._run_cmd_with_progress(merge_argv, count_run_lines=False)
        if rc != 0:
            raise RuntimeError("Merge échoué")

        # Verify outputs exist
        workdir = Path(self.workdir_var.get())
        summary_all = workdir / f"{merge_prefix}_ALL_summary.csv"
        ts_all = workdir / f"{merge_prefix}_ALL_timeseries.csv"
        if not summary_all.exists() or not ts_all.exists():
            raise RuntimeError("Merge terminé mais fichiers *_ALL_ manquants")

        self._log_queue.put("[runner] Merge OK.\n")
        self._log_queue.put(f"[time] {_system_time_str()} - Fin étape: Merge.\n")

        # Excel report
        out_xlsx = workdir / f"{merge_prefix}_report.xlsx"
        ref_algo = self.baseline_algo_var.get().strip() or "FSS"
        correction = self.correction_var.get().strip() or "holm"

        self._log_queue.put(f"[runner] Excel: {out_xlsx.name}\n")
        self._log_queue.put(f"[time] {_system_time_str()} - Début étape: Excel.\n")
        rep_argv = [
            sys.executable,
            "-u",
            "-m",
            "scripts.report_excel",
            "--summary",
            str(summary_all),
            "--timeseries",
            str(ts_all),
            "--out",
            str(out_xlsx),
            "--wilcoxon_mode",
            "ref_vs_each",
            "--ref_algo",
            ref_algo,
            "--correction",
            correction,
        ]
        rc = self._run_cmd_with_progress(rep_argv, count_run_lines=False)
        if rc != 0:
            raise RuntimeError("Génération Excel échouée")

        if not out_xlsx.exists():
            raise RuntimeError("Excel non créé")

        self._log_queue.put(f"[time] {_system_time_str()} - Fin étape: Excel.\n")

        # Optional plots
        plot_types = self._selected_plot_types()
        if plot_types:
            self._log_queue.put(f"[runner] Graphiques: {', '.join(plot_types)}\n")
            self._log_queue.put(f"[time] {_system_time_str()} - Début étape: Graphiques.\n")
            self._run_plots_for_excel(Path(out_xlsx), workdir, plot_types)
            self._log_queue.put(f"[time] {_system_time_str()} - Fin étape: Graphiques.\n")

        self._log_queue.put("[runner] Pipeline complet OK.\n")
        self.root.after(0, self._refresh_dynamic_lists)

    def _run_excel_only(self) -> None:
        workdir = Path(self.workdir_var.get())
        pref = self.excel_prefix_var.get().strip()
        if not pref:
            raise ValueError("Choisir un préfixe")

        summary_all = workdir / f"{pref}_ALL_summary.csv"
        ts_all = workdir / f"{pref}_ALL_timeseries.csv"
        if not summary_all.exists() or not ts_all.exists():
            raise ValueError("Fichiers *_ALL_summary.csv / *_ALL_timeseries.csv introuvables pour ce préfixe")

        out = self.excel_file_var.get().strip() or str(workdir / f"{pref}_report.xlsx")
        ref_algo = self.baseline_algo_var.get().strip() or "FSS"
        correction = self.correction_var.get().strip() or "holm"

        argv = [
            sys.executable,
            "-u",
            "-m",
            "scripts.report_excel",
            "--summary",
            str(summary_all),
            "--timeseries",
            str(ts_all),
            "--out",
            out,
            "--wilcoxon_mode",
            "ref_vs_each",
            "--ref_algo",
            ref_algo,
            "--correction",
            correction,
        ]
        self._log_queue.put(f"[time] {_system_time_str()} - Début étape: Excel.\n")
        rc = self._run_cmd_with_progress(argv, count_run_lines=False)
        if rc != 0:
            raise RuntimeError("Génération Excel échouée")

        self._log_queue.put(f"[runner] Excel OK: {out}\n")
        self._log_queue.put(f"[time] {_system_time_str()} - Fin étape: Excel.\n")

    def _run_plots_only(self) -> None:
        # If user selected multiple Excel reports, run scenario comparison instead.
        if self.compare_excels:
            if len(self.compare_excels) < 2:
                raise ValueError("Comparer scénarios: sélectionnez au moins 2 fichiers Excel")

            plot_types = self._selected_plot_types()
            if not plot_types:
                self._log_queue.put("[runner] Aucun graphique sélectionné.\n")
                return

            mode = (self.metric_mode_var.get() or "one").strip()

            # Metrics to run for scenario comparison
            metrics_to_run: List[str] = []
            if mode == "one":
                metrics_to_run = [self.metric_var.get().strip() or "throughput"]
            elif mode == "select":
                # Keep UI order from _metric_values
                metrics_to_run = [m for m in self._metric_values if bool(self._metric_select_vars.get(str(m), tk.BooleanVar(value=False)).get())]
                if not metrics_to_run:
                    raise ValueError("Sélection métriques: cochez au moins une métrique")
            else:
                # all
                metrics_to_run = self._common_metrics_for_excels(self.compare_excels)
                if not metrics_to_run:
                    # Fallback to current known list (may still fail later if missing in some Excel)
                    metrics_to_run = list(self._metric_values) or [self.metric_var.get().strip() or "throughput"]

            # In 'select'/'all' modes, only keep metrics present in all selected excels.
            common_metrics = set(self._common_metrics_for_excels(self.compare_excels))
            if common_metrics:
                missing = [m for m in metrics_to_run if m not in common_metrics]
                if missing:
                    miss_txt = ", ".join(missing)
                    common_txt = self._format_short_list([m for m in self._common_metrics_for_excels(self.compare_excels)])
                    raise ValueError(
                        "Certaines métriques ne sont pas présentes dans tous les Excel sélectionnés:\n"
                        f"- Manquantes: {miss_txt}\n\n"
                        f"Métriques communes disponibles:\n{common_txt}"
                    )
                # Filter any non-common metrics (defensive)
                metrics_to_run = [m for m in metrics_to_run if m in common_metrics]

            algos_raw = self.compare_algos_var.get().strip() or "FSS"
            algos = self._validate_compare_algos_input(algos_raw)

            metrics_disp = self._format_short_list(metrics_to_run, max_items=10)
            self._log_queue.put(
                f"[runner] Compare scénarios: metrics={metrics_disp}, algos={algos}, plots={','.join(plot_types)}\n"
            )

            # Progress: one command per metric
            with self._progress_lock:
                self._progress_total_units = max(1, len(metrics_to_run))
                self._progress_done_units = 0
                self._progress_t0 = time.time()
            self._compute_and_push_progress()

            self._log_queue.put(f"[time] {_system_time_str()} - Début étape: Graphiques (comparaison scénarios).\n")

            out_dir = str(Path(self.workdir_var.get()))
            for metric in metrics_to_run:
                if self._stop_event.is_set():
                    raise RuntimeError("Arrêt demandé")

                self._log_queue.put(f"[runner] -> métrique: {metric}\n")
                argv = [
                    sys.executable,
                    "-u",
                    "-m",
                    "scripts.plot_multi_excel",
                    "--excels",
                    *self.compare_excels,
                    "--out_dir",
                    out_dir,
                    "--metric",
                    str(metric),
                    "--algos",
                    algos,
                    "--plots",
                    ",".join(plot_types),
                ]
                rc = self._run_cmd_with_progress(argv, count_run_lines=False)
                if rc != 0:
                    raise RuntimeError(f"Génération graphique scénarios échouée (metric={metric})")

                with self._progress_lock:
                    self._progress_done_units += 1
                self._compute_and_push_progress()

            self._log_queue.put("[runner] Graphiques comparaison scénarios OK.\n")
            self._log_queue.put(f"[time] {_system_time_str()} - Fin étape: Graphiques (comparaison scénarios).\n")
            self.root.after(0, self._refresh_dynamic_lists)
            return

        excel_path = self.excel_exist_cb.get().strip()
        if not excel_path:
            raise ValueError("Choisir un fichier Excel")

        plot_types = self._selected_plot_types()
        if not plot_types:
            self._log_queue.put("[runner] Aucun graphique sélectionné.\n")
            return

        workdir = Path(self.workdir_var.get())
        self._log_queue.put(f"[time] {_system_time_str()} - Début étape: Graphiques.\n")
        self._run_plots_for_excel(Path(excel_path), workdir, plot_types)

        self._log_queue.put("[runner] Graphiques OK.\n")
        self._log_queue.put(f"[time] {_system_time_str()} - Fin étape: Graphiques.\n")

    def _run_plots_for_excel(self, excel: Path, out_dir: Path, plot_types: List[str]) -> None:
        """Run plot_from_excel using the 6 generic plot types requested in the UI."""

        # Map UI plot types -> plot_from_excel plot types
        # - heatmap => generate the 3 heatmaps available in plot_from_excel
        # - scatter => volcano
        # - bar => mean_bar
        # - box/hist/line => new plot types in plot_from_excel
        global_plots: List[str] = []
        metric_plots: List[str] = []

        if "heatmap" in plot_types:
            global_plots.extend(["means_heatmap", "rank_heatmap", "pvalue_heatmap"])
        if "bar" in plot_types:
            metric_plots.append("mean_bar")
        if "scatter" in plot_types:
            metric_plots.append("volcano")
        if "box" in plot_types:
            metric_plots.append("box")
        if "hist" in plot_types:
            metric_plots.append("hist")
        if "line" in plot_types:
            metric_plots.append("line")

        # Run global plots once
        if global_plots:
            argv = [
                sys.executable,
                "-u",
                "-m",
                "scripts.plot_from_excel",
                "--excel",
                str(excel),
                "--out_dir",
                str(out_dir),
                "--metric",
                (self.metric_var.get().strip() or "throughput"),
                "--plots",
                ",".join(global_plots),
            ]
            rc = self._run_cmd_with_progress(argv, count_run_lines=False)
            if rc != 0:
                raise RuntimeError("Génération graphiques échouée")

        if not metric_plots:
            self.root.after(0, self._refresh_dynamic_lists)
            return

        # 'line' is timeseries-based and typically should be generated for a single metric,
        # not for every summary metric when mode=all/select.
        line_only = [p for p in metric_plots if p == "line"]
        multi_metric_plots = [p for p in metric_plots if p != "line"]

        mode = (self.metric_mode_var.get() or "one").strip()

        # Multi-metric plots (bar/scatter/box/hist) follow the metric selection mode.
        if multi_metric_plots:
            metrics: List[str]
            if mode == "all":
                metrics = self._extract_metrics_from_excel(excel)
                if not metrics:
                    metrics = [self.metric_var.get().strip() or "throughput"]
            elif mode == "select":
                metrics = [m for m, v in self._metric_select_vars.items() if bool(v.get())]
                if not metrics:
                    raise ValueError("Sélection métriques: cochez au moins une métrique")
            else:
                metrics = [self.metric_var.get().strip() or "throughput"]

            for m in metrics:
                argv = [
                    sys.executable,
                    "-u",
                    "-m",
                    "scripts.plot_from_excel",
                    "--excel",
                    str(excel),
                    "--out_dir",
                    str(out_dir),
                    "--metric",
                    str(m),
                    "--plots",
                    ",".join(multi_metric_plots),
                ]
                rc = self._run_cmd_with_progress(argv, count_run_lines=False)
                if rc != 0:
                    raise RuntimeError("Génération graphiques échouée")

        # Line plot: run once for the current metric.
        if line_only:
            if mode in {"all", "select"}:
                self._log_queue.put("[runner] Note: 'Line' est généré uniquement pour la métrique courante (timeseries).\n")

            line_metric = self.metric_var.get().strip() or "alive"
            argv = [
                sys.executable,
                "-u",
                "-m",
                "scripts.plot_from_excel",
                "--excel",
                str(excel),
                "--out_dir",
                str(out_dir),
                "--metric",
                str(line_metric),
                "--plots",
                "line",
            ]
            rc = self._run_cmd_with_progress(argv, count_run_lines=False)
            if rc != 0:
                raise RuntimeError("Génération graphiques échouée")

        self.root.after(0, self._refresh_dynamic_lists)

    def _selected_plot_types(self) -> List[str]:
        selected: List[str] = []
        if self.plot_scatter.get():
            selected.append("scatter")
        if self.plot_bar.get():
            selected.append("bar")
        if self.plot_box.get():
            selected.append("box")
        if self.plot_line.get():
            selected.append("line")
        if self.plot_hist.get():
            selected.append("hist")
        if self.plot_heatmap.get():
            selected.append("heatmap")
        return selected


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    app = RunnerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
    
    