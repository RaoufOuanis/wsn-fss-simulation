#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import re
import sys
import time
import queue
import shutil
import threading
import subprocess
import difflib
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd

from wsn.plot_style import ALGO_COLOR_MAP


WORKDIR_DEFAULT = Path(r"C:\wsn-project")

# Persist small UI settings (e.g., dark mode) across restarts.
CONFIG_PATH = Path.home() / ".wsn_runner_config.json"

# Default campaign size: historically 6 jobs × 5 runs = 30 seeds.
SEEDS_TOTAL_DEFAULT = 30

# Default parallelism: start with 6 jobs as before, but allow the user to raise it.
PARALLEL_JOBS_DEFAULT = 6

# Periodic mini-log interval (seconds) for long runs.
HEARTBEAT_INTERVAL_S_DEFAULT = 0.0

# Keep a constrained dropdown list as requested.
ROUND_VALUES = [str(r) for r in range(200, 5001, 100)]

GUI_EXTRA_ALGOS = ["SO", "GJO", "EMOGJO", "EMOGJO_paperCH", "ESOGJO", "FSS_legacy", "FSS_cov"]

S1_ALGOS_DEFAULT = "FSS,PSO,GWO,ABC,SO,GJO,EMOGJO,ESOGJO,LEACH,HEED,SEP,EEM_LEACH_ABC"
S2_ALGOS_DEFAULT = "FSS,PSO,GWO,ABC,SO,GJO,EMOGJO,ESOGJO,LEACH,HEED,SEP,EEM_LEACH_ABC"

# Algorithms considered negligible for ETA estimation (per user request).
ETA_FAST_ALGOS = {"LEACH", "HEED", "SEP", "EEM_LEACH_ABC"}


# Per-run persistence to enable resume on next startup.
RUNS_DIR_NAME = "_runner_runs"
RUN_SESSION_FILE = "session.json"


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    """Best-effort atomic JSON write."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _make_seed_jobs(prefix_base: str, total_seeds: int, parallel_jobs: int) -> List[Tuple[str, int, int]]:
    """Split a fixed number of seeds across parallel jobs.

    We keep seed uniqueness by assigning each job a contiguous seed block.
    Each job runs `runs_job` seeds starting from `base_seed`.

    Returns a list of (prefix, base_seed, runs_job).
    """

    total = int(total_seeds)
    if total <= 0:
        return []

    nj = int(max(1, parallel_jobs))
    nj = min(nj, total)  # do not create empty jobs

    base = total // nj
    rem = total % nj

    jobs: List[Tuple[str, int, int]] = []
    seed0 = 0
    for j in range(nj):
        runs_job = base + (1 if j < rem else 0)
        if runs_job <= 0:
            continue
        start = seed0
        end = seed0 + runs_job - 1
        suffix = f"s{start:02d}_{end:02d}"
        jobs.append((f"{prefix_base}_{suffix}", int(seed0), int(runs_job)))
        seed0 += runs_job

    return jobs

RUN_LINE_RE = re.compile(r"^\[run_experiments\]\s+(?P<scenario>[^-]+?)\s+-\s+(?P<algo>[^-]+?)\s+-\s+run\s+(?P<run>\d+)\s*$")


def _system_time_str() -> str:
    # Local system time, formatted for logs.
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _campaign_prefix_base(scenario_tag: str, bs_mode: str, rounds: int) -> str:
    # Keep exactly the naming style used in your files: paper_S1_100_center_R2200
    return f"paper_{scenario_tag}_{bs_mode}_R{rounds}"


def _make_jobs(prefix_base: str, total_seeds: int, parallel_jobs: int) -> List[Tuple[str, int, int]]:
    # Backward-compatible name kept for older call sites.
    return _make_seed_jobs(prefix_base, total_seeds=total_seeds, parallel_jobs=parallel_jobs)


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
        # The UI contains 2 stacked control panels (Simulation + Plots) plus logs.
        # A taller default prevents the Plots panel from being clipped on startup.
        self.root.geometry("900x760")

        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None
        self._log_queue: queue.Queue[str] = queue.Queue()

        # Track all currently running subprocesses (simulate/merge/excel/plots + parallel jobs)
        self._procs_lock = threading.Lock()
        self._active_procs: dict[int, subprocess.Popen] = {}
        self._paused: bool = False

        self.workdir_var = tk.StringVar(value=str(WORKDIR_DEFAULT))

        self.mode_var = tk.StringVar(value="simulate")  # simulate | excel | plots
        self.shell_var = tk.StringVar(value="powershell")  # powershell | cmd

        # Logging verbosity
        # When disabled, we still parse/count run lines but do not print each one.
        self.verbose_logs_var = tk.BooleanVar(value=False)

        # Theme
        self.dark_mode_var = tk.BooleanVar(value=False)

        # Load persisted UI settings (best-effort).
        try:
            self._load_settings()
        except Exception:
            pass

        self.scenario_var = tk.StringVar(value="S1_100")
        self.bs_var = tk.StringVar(value="center")

        # Campaign sizing / parallelism
        self.seeds_total_var = tk.StringVar(value=str(SEEDS_TOTAL_DEFAULT))
        self.parallel_jobs_var = tk.StringVar(value=str(PARALLEL_JOBS_DEFAULT))

        self.rounds_var = tk.StringVar(value="2200")
        self.algos_var = tk.StringVar(value=S1_ALGOS_DEFAULT)

        # Quick-add algorithm picker (for building the CSV list)
        self._algo_add_var = tk.StringVar(value="")

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
            "avg_nfe",
            "avg_pkt_hops_round",
            "avg_n_ch_round",
            "avg_n_ch_raw_round",
            "avg_n_ch_added_by_repair_round",
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
        self.finish_time_var = tk.StringVar(value="Fin: --")

        self._progress_total_units: int = 0
        self._progress_done_units: int = 0
        self._progress_t0: float = 0.0
        self._progress_lock = threading.Lock()

        # Last algo seen from a run progress line (used for heartbeat logs)
        self._last_run_algo: str = ""
        self._last_run_scenario: str = ""
        self._last_run_run: Optional[int] = None
        self._last_run_seed: Optional[int] = None

        # Campaign context (for more readable progress)
        self._campaign_algos_count: int = 0
        self._campaign_total_seeds: int = 0

        # ETA estimator state (smoothed units/sec)
        self._eta_last_t: float = 0.0
        self._eta_last_done: int = 0
        self._eta_rate_ema: float = 0.0

        # Periodic mini-logs (heartbeat)
        self._hb_last_t: float = 0.0
        self._hb_interval_s: float = float(HEARTBEAT_INTERVAL_S_DEFAULT)

        # Optional ETA override (e.g., job-duration-based ETA)
        self._eta_override_s: Optional[int] = None
        self._eta_override_finish_dt: Optional[datetime] = None

        # Per-job stdout tee logs (created during parallel simulation; cleaned up on success)
        self._job_logs_dir: Optional[Path] = None

        # Persisted run session (for resume on next startup)
        self._run_session_dir: Optional[Path] = None
        self._run_session_file: Optional[Path] = None
        self._resume_session_dir: Optional[Path] = None

        self._build_ui()

        # Ensure we always kill spawned processes when the window is closed.
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        except Exception:
            pass

        # Capture Tk callback exceptions (otherwise they can be "silent").
        try:
            self.root.report_callback_exception = self._on_tk_exception  # type: ignore[assignment]
        except Exception:
            pass

        self._refresh_dynamic_lists()
        self._refresh_compare_algos_help()
        self._log_queue.put(f"[time] {_system_time_str()} - Application started.\n")
        self._tick_ui()

        # Apply initial theme
        try:
            self._apply_theme(bool(self.dark_mode_var.get()))
        except Exception:
            pass

        # Offer resume for the latest unfinished run (best-effort).
        try:
            self.root.after(350, self._maybe_offer_resume_on_startup)
        except Exception:
            pass

    def _session_set(self, update: Dict[str, Any]) -> None:
        """Merge-update and persist the current run session manifest."""

        if self._run_session_file is None:
            return
        cur = _read_json(self._run_session_file) or {}
        if not isinstance(cur, dict):
            cur = {}
        cur.update(update)
        cur["updated_at"] = _system_time_str()
        try:
            _write_json_atomic(self._run_session_file, cur)
        except Exception:
            return

    def _session_set_step(self, step: str, status: str) -> None:
        if self._run_session_file is None:
            return
        cur = _read_json(self._run_session_file) or {}
        steps = cur.get("steps")
        if not isinstance(steps, dict):
            steps = {}
        steps[str(step)] = str(status)
        cur["steps"] = steps
        cur["updated_at"] = _system_time_str()
        try:
            _write_json_atomic(self._run_session_file, cur)
        except Exception:
            return

    def _maybe_offer_resume_on_startup(self) -> None:
        """Detect latest unfinished session in current workdir and offer resume."""

        try:
            workdir = Path(self.workdir_var.get())
        except Exception:
            return

        runs_root = workdir / RUNS_DIR_NAME
        if not runs_root.exists():
            return

        best_dir: Optional[Path] = None
        best_mtime: float = -1.0
        best_manifest: Optional[Dict[str, Any]] = None

        try:
            for d in runs_root.iterdir():
                if not d.is_dir():
                    continue
                mf = d / RUN_SESSION_FILE
                manifest = _read_json(mf)
                if not manifest:
                    continue
                if str(manifest.get("mode") or "") != "simulate":
                    continue
                status = str(manifest.get("status") or "")
                if status in {"complete", "dismissed", "abandoned"}:
                    continue
                try:
                    mtime = float(mf.stat().st_mtime)
                except Exception:
                    mtime = 0.0
                if mtime > best_mtime:
                    best_dir = d
                    best_mtime = mtime
                    best_manifest = manifest
        except Exception:
            return

        if best_dir is None or best_manifest is None:
            return

        started = str(best_manifest.get("started_at") or "(unknown)")
        status = str(best_manifest.get("status") or "unfinished")

        try:
            if not bool(messagebox.askyesno("Resume", f"Found an unfinished run ({status}) from {started}.\n\nResume it?")):
                # User refused resume: delete ALL logs and session metadata.
                # Primary behavior: remove the whole session directory.
                removed = False
                try:
                    shutil.rmtree(Path(best_dir), ignore_errors=False)
                    removed = True
                except Exception:
                    removed = False

                # Fallback: if deletion fails, at least delete job logs + mark dismissed
                # so we won't keep re-prompting.
                if not removed:
                    try:
                        logs_dir = Path(best_dir) / "job_logs"
                        shutil.rmtree(logs_dir, ignore_errors=True)
                    except Exception:
                        pass
                    try:
                        mf = Path(best_dir) / RUN_SESSION_FILE
                        manifest = _read_json(mf) or {}
                        if isinstance(manifest, dict):
                            manifest["status"] = "dismissed"
                            manifest["updated_at"] = _system_time_str()
                            _write_json_atomic(mf, manifest)
                    except Exception:
                        pass
                return
        except Exception:
            return

        # Load params into UI (best-effort), then start in resume mode.
        params = best_manifest.get("params")
        if isinstance(params, dict):
            try:
                if "workdir" in params:
                    self.workdir_var.set(str(params.get("workdir")))
                if "scenario" in params:
                    self.scenario_var.set(str(params.get("scenario")))
                if "bs" in params:
                    self.bs_var.set(str(params.get("bs")))
                if "seeds_total" in params:
                    self.seeds_total_var.set(str(params.get("seeds_total")))
                if "parallel_jobs" in params:
                    self.parallel_jobs_var.set(str(params.get("parallel_jobs")))
                if "rounds" in params:
                    self.rounds_var.set(str(params.get("rounds")))
                if "algos" in params:
                    self.algos_var.set(str(params.get("algos")))
                if "baseline_algo" in params:
                    self.baseline_algo_var.set(str(params.get("baseline_algo")))
                if "correction" in params:
                    self.correction_var.set(str(params.get("correction")))
            except Exception:
                pass

        try:
            self.mode_var.set("simulate")
            self._refresh_dynamic_lists()
        except Exception:
            pass

        self._resume_session_dir = Path(best_dir)
        self._on_start()

    def _load_settings(self) -> None:
        try:
            p = Path(CONFIG_PATH)
            if not p.exists():
                return
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            if "dark_mode" in data:
                self.dark_mode_var.set(bool(data.get("dark_mode")))
        except Exception:
            return

    def _save_settings(self) -> None:
        try:
            data = {
                "dark_mode": bool(self.dark_mode_var.get()),
            }
            p = Path(CONFIG_PATH)
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(p)
        except Exception:
            return

    def _on_toggle_dark_mode(self) -> None:
        dark = bool(self.dark_mode_var.get())
        self._apply_theme(dark)
        self._save_settings()

    def _apply_theme(self, dark: bool) -> None:
        """Best-effort dark mode for ttk + the log Text widget."""

        style = ttk.Style(self.root)
        if not hasattr(self, "_default_ttk_theme"):
            try:
                self._default_ttk_theme = style.theme_use()
            except Exception:
                self._default_ttk_theme = "default"

        if dark:
            try:
                style.theme_use("clam")
            except Exception:
                pass

            bg = "#1e1e1e"
            fg = "#d4d4d4"
            widget_bg = "#252526"
            accent = "#3a3a3a"

            # For readability on some Windows themes, keep entry/combobox fields light
            # and use black text (user request).
            field_bg = "white"
            field_fg = "black"

            try:
                self.root.configure(bg=bg)
            except Exception:
                pass

            for key in ["TFrame", "TLabel", "TButton", "TRadiobutton", "TCheckbutton"]:
                try:
                    style.configure(key, background=bg, foreground=fg)
                except Exception:
                    pass
            for key in ["TLabelframe", "TLabelframe.Label"]:
                try:
                    style.configure(key, background=bg, foreground=fg)
                except Exception:
                    pass
            for key in ["TEntry", "TCombobox"]:
                try:
                    style.configure(key, fieldbackground=field_bg, background=field_bg, foreground=field_fg)
                except Exception:
                    pass

            # Combobox drop-down list styling (best-effort; Tk option database).
            try:
                self.root.option_add("*TCombobox*Listbox.background", field_bg)
                self.root.option_add("*TCombobox*Listbox.foreground", field_fg)
            except Exception:
                pass
            try:
                style.configure("TProgressbar", background=accent)
            except Exception:
                pass

            try:
                self.log_text.configure(
                    bg=widget_bg,
                    fg=fg,
                    insertbackground=fg,
                    selectbackground=accent,
                )
            except Exception:
                pass
        else:
            try:
                style.theme_use(str(getattr(self, "_default_ttk_theme", "default")))
            except Exception:
                pass
            try:
                self.root.configure(bg=None)
            except Exception:
                pass
            try:
                self.log_text.configure(bg="white", fg="black", insertbackground="black")
            except Exception:
                pass

    def _on_tk_exception(self, exc, val, tb) -> None:
        # Called by Tk when a callback raises.
        try:
            lines = traceback.format_exception(exc, val, tb)
            self._log_queue.put("[tk] Exception in Tk callback:\n" + "".join(lines) + "\n")
        except Exception:
            self._log_queue.put("[tk] Exception in Tk callback (traceback unavailable).\n")

        # Best-effort: avoid leaving orphan processes.
        try:
            self._stop_event.set()
            self._kill_all_processes()
        except Exception:
            pass

        # Show error to user (must be on UI thread).
        try:
            self._ui_show_error("Tk Error", str(val) if val is not None else "Unknown error")
        except Exception:
            pass

    def _ui_show_error(self, title: str, msg: str) -> None:
        def _show() -> None:
            try:
                messagebox.showerror(title, msg)
            except Exception:
                pass

        try:
            self.root.after(0, _show)
        except Exception:
            pass

    @staticmethod
    def _split_csv_names(s: str) -> List[str]:
        parts = [p.strip() for p in str(s).split(",")]
        return [p for p in parts if p]

    def _known_algos_for_picker(self) -> List[str]:
        known = sorted((set(ALGO_COLOR_MAP.keys()) | set(GUI_EXTRA_ALGOS)), key=lambda x: (x != "FSS", x.lower()))
        return [str(a) for a in known if str(a).strip()]

    def _add_algo_to_csv_var(self, var: tk.StringVar, algo: str) -> None:
        a = str(algo).strip()
        if not a:
            return
        cur = self._split_csv_names(var.get())
        if a in cur:
            return
        cur.append(a)
        var.set(",".join(cur))

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
            self.compare_algos_help_var.set(f"Available (in selected Excel): {disp}  |  Comma-separated")
        else:
            disp = self._format_short_list(base)
            self.compare_algos_help_var.set(f"Common names: {disp}  |  Comma-separated")

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
            raise ValueError("Empty algorithms list")

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
                    suggestions.append(f"{u} → (no suggestion)")

            avail_text = self._format_short_list(sorted(allowed, key=lambda x: (x != "FSS", x.lower())))
            msg = (
                "Unknown algorithm(s) in 'Algorithms (scenarios)':\n"
                + " - "
                + "\n - ".join(unknown)
                + "\n\nSuggestions:\n - "
                + "\n - ".join(suggestions)
                + "\n\nAvailable:\n"
                + avail_text
            )
            raise ValueError(msg)

        # Preserve user order, but normalize spacing.
        return ",".join(algos)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Working directory:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.workdir_var, width=70).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(top, text="Browse...", command=self._choose_workdir).grid(row=0, column=2, sticky="e")

        ttk.Label(top, text="Shell:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        shell_frame = ttk.Frame(top)
        shell_frame.grid(row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Radiobutton(shell_frame, text="PowerShell", value="powershell", variable=self.shell_var).pack(side="left")
        ttk.Radiobutton(shell_frame, text="CMD", value="cmd", variable=self.shell_var).pack(side="left", padx=(12, 0))

        ttk.Label(top, text="Mode:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        mode_frame = ttk.Frame(top)
        mode_frame.grid(row=2, column=1, sticky="w", pady=(8, 0))
        ttk.Radiobutton(mode_frame, text="Simulate → Merge → Excel → Plots", value="simulate", variable=self.mode_var, command=self._update_mode_visibility).pack(side="left")
        ttk.Radiobutton(mode_frame, text="Excel from CSV *_ALL_*", value="excel", variable=self.mode_var, command=self._update_mode_visibility).pack(side="left", padx=(12, 0))
        ttk.Radiobutton(mode_frame, text="Plots from Excel", value="plots", variable=self.mode_var, command=self._update_mode_visibility).pack(side="left", padx=(12, 0))

        top.columnconfigure(1, weight=1)

        # Main frames
        self.sim_frame = ttk.LabelFrame(self.root, text="Simulation", padding=10)
        self.excel_frame = ttk.LabelFrame(self.root, text="Generate Excel", padding=10)
        self.plots_frame = ttk.LabelFrame(self.root, text="Plots", padding=10)

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
        ttk.Label(bottom, textvariable=self.progress_text_var, width=7).grid(row=0, column=1, padx=(8, 0), sticky="w")
        ttk.Label(bottom, textvariable=self.eta_var).grid(row=0, column=2, padx=(8, 0), sticky="w")
        ttk.Label(bottom, textvariable=self.finish_time_var).grid(row=0, column=3, padx=(8, 0), sticky="w")

        btns = ttk.Frame(bottom)
        btns.grid(row=1, column=0, columnspan=3, sticky="we", pady=(8, 0))
        self.start_btn = ttk.Button(btns, text="Start", command=self._on_start)
        self.start_btn.pack(side="left")
        self.pause_btn = ttk.Button(btns, text="Pause", command=self._on_pause, state="disabled")
        self.pause_btn.pack(side="left", padx=(8, 0))
        self.stop_btn = ttk.Button(btns, text="Stop", command=self._on_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 0))
        ttk.Checkbutton(btns, text="Detailed logs", variable=self.verbose_logs_var).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(
            btns,
            text="Dark mode",
            variable=self.dark_mode_var,
            command=self._on_toggle_dark_mode,
        ).pack(side="left", padx=(12, 0))
        ttk.Button(btns, text="Refresh lists", command=self._refresh_dynamic_lists).pack(side="right")

        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=0)
        bottom.columnconfigure(2, weight=0)
        bottom.columnconfigure(3, weight=0)

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

        ttk.Label(self.sim_frame, text="Total seeds (per campaign):").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(self.sim_frame, textvariable=self.seeds_total_var, width=10).grid(row=1, column=1, sticky="w", pady=(8, 0))

        ttk.Label(self.sim_frame, text="Parallel jobs:").grid(row=1, column=2, sticky="w", padx=(14, 0), pady=(8, 0))
        ttk.Entry(self.sim_frame, textvariable=self.parallel_jobs_var, width=10).grid(row=1, column=3, sticky="w", pady=(8, 0))

        ttk.Label(self.sim_frame, text="Algorithms (CSV):").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(self.sim_frame, textvariable=self.algos_var, width=60).grid(row=2, column=1, columnspan=4, sticky="we", pady=(8, 0))

        # Keep rounds + quick-add on their own row to avoid clipping/overlap.
        ttk.Label(self.sim_frame, text="Rounds (200..5000):").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(self.sim_frame, textvariable=self.rounds_var, values=ROUND_VALUES, state="readonly", width=10).grid(row=3, column=1, sticky="w", pady=(8, 0))

        # Quick add: pick an algo and append it to the CSV list.
        ttk.Label(self.sim_frame, text="Add algo:").grid(row=3, column=2, sticky="w", padx=(14, 0), pady=(8, 0))
        self._algo_add_cb = ttk.Combobox(
            self.sim_frame,
            textvariable=self._algo_add_var,
            values=self._known_algos_for_picker(),
            state="readonly",
            width=18,
        )
        self._algo_add_cb.grid(row=3, column=3, sticky="w", pady=(8, 0))

        def _on_add_algo() -> None:
            self._add_algo_to_csv_var(self.algos_var, self._algo_add_var.get())

        ttk.Button(self.sim_frame, text="Add", command=_on_add_algo).grid(row=3, column=4, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(self.sim_frame, text="Reference (Wilcoxon):").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(self.sim_frame, textvariable=self.baseline_algo_var, width=12).grid(row=4, column=1, sticky="w", pady=(8, 0))

        ttk.Label(self.sim_frame, text="Correction:").grid(row=4, column=2, sticky="w", padx=(14, 0), pady=(8, 0))
        ttk.Combobox(self.sim_frame, textvariable=self.correction_var, values=["holm", "none"], state="readonly", width=10).grid(row=4, column=3, sticky="w", pady=(8, 0))

        self.sim_frame.columnconfigure(1, weight=1)

    def _build_excel_frame(self) -> None:
        ttk.Label(self.excel_frame, text="Detected prefix (*_ALL_summary.csv):").grid(row=0, column=0, sticky="w")
        self.excel_prefix_cb = ttk.Combobox(self.excel_frame, textvariable=self.excel_prefix_var, values=[], state="readonly", width=45)
        self.excel_prefix_cb.grid(row=0, column=1, sticky="w")

        ttk.Button(self.excel_frame, text="Use this prefix", command=self._set_excel_from_prefix).grid(row=0, column=2, padx=(8, 0))

        ttk.Label(self.excel_frame, text="Excel file:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(self.excel_frame, textvariable=self.excel_file_var, width=60).grid(row=1, column=1, sticky="we", pady=(8, 0))
        ttk.Button(self.excel_frame, text="Choose...", command=self._choose_excel_out).grid(row=1, column=2, padx=(8, 0), pady=(8, 0))

        ttk.Label(self.excel_frame, text="Correction:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(self.excel_frame, textvariable=self.correction_var, values=["holm", "none"], state="readonly", width=10).grid(row=2, column=1, sticky="w", pady=(8, 0))

        ttk.Label(self.excel_frame, text="Reference:").grid(row=2, column=2, sticky="w", padx=(14, 0), pady=(8, 0))
        ttk.Entry(self.excel_frame, textvariable=self.baseline_algo_var, width=12).grid(row=2, column=3, sticky="w", pady=(8, 0))

        self.excel_frame.columnconfigure(1, weight=1)

    def _build_plots_frame(self) -> None:
        ttk.Label(self.plots_frame, text="Metrics:").grid(row=0, column=0, sticky="w")

        metric_mode = ttk.Frame(self.plots_frame)
        metric_mode.grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(metric_mode, text="One metric", value="one", variable=self.metric_mode_var, command=self._refresh_metric_state).pack(side="left")
        ttk.Radiobutton(metric_mode, text="All", value="all", variable=self.metric_mode_var, command=self._refresh_metric_state).pack(side="left", padx=(12, 0))
        ttk.Radiobutton(metric_mode, text="Select", value="select", variable=self.metric_mode_var, command=self._refresh_metric_state).pack(side="left", padx=(12, 0))

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
        # 2 rows × 3 columns to avoid clipping.
        for i, (label, var) in enumerate(items):
            r = i // 3
            c = i % 3
            ttk.Checkbutton(checks, text=label, variable=var).grid(row=r, column=c, sticky="w", padx=(0 if c == 0 else 14, 0), pady=(0 if r == 0 else 4, 0))

        ttk.Label(self.plots_frame, text="Existing Excel:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.excel_exist_cb = ttk.Combobox(self.plots_frame, values=[], state="readonly", width=60)
        self.excel_exist_cb.grid(row=3, column=1, columnspan=2, sticky="we", pady=(8, 0))
        ttk.Button(self.plots_frame, text="Choose...", command=self._choose_excel_in).grid(row=3, column=3, padx=(8, 0), pady=(8, 0))

        # Multi-Excel scenario comparison
        ttk.Label(self.plots_frame, text="Compare scenarios (multi-Excel):").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(self.plots_frame, textvariable=self.compare_excels_var, state="readonly", width=60).grid(
            row=4, column=1, columnspan=2, sticky="we", pady=(8, 0)
        )
        ttk.Button(self.plots_frame, text="Choose...", command=self._choose_excels_multi).grid(row=4, column=3, padx=(8, 0), pady=(8, 0))

        ttk.Label(self.plots_frame, text="Algorithms (scenarios):").grid(row=5, column=0, sticky="w", pady=(8, 0))
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
        self.compare_excels_var.set(f"{len(names)} file(s): {disp}")

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
        # Coalesce logs to reduce refresh overhead and avoid UI churn.
        max_lines = 500
        buf: List[str] = []
        for _ in range(max_lines):
            try:
                line = self._log_queue.get_nowait()
            except queue.Empty:
                break
            buf.append(line if line.endswith("\n") else (line + "\n"))

        if buf:
            try:
                self.log_text.insert("end", "".join(buf))
                self.log_text.see("end")
            except Exception:
                for line in buf:
                    self._append_log(line)

        self.root.after(300, self._tick_ui)

    def _on_close(self) -> None:
        # Called when user closes the window.

        try:
            running = False
            if self._worker is not None and self._worker.is_alive():
                running = True
            if self._snapshot_active_procs():
                running = True

            msg = "Do you want to exit WSN Runner?"
            if running:
                msg += "\n\nThere are running processes and they will be stopped."

            if not bool(messagebox.askyesno("Exit", msg)):
                return
        except Exception:
            # If UI prompt fails, proceed with best-effort close.
            pass

        try:
            self._stop_event.set()
            self._kill_all_processes()
        except Exception:
            pass

        # If a run session is active, mark it as stopped for resume-on-startup.
        try:
            if self._run_session_file is not None:
                self._session_set({"status": "stopped", "stop_reason": "window_closed"})
        except Exception:
            pass
        try:
            self._save_settings()
        except Exception:
            pass
        try:
            self.root.after(50, self.root.destroy)
        except Exception:
            try:
                self.root.destroy()
            except Exception:
                pass

    def _set_running(self, running: bool) -> None:
        self.start_btn.configure(state=("disabled" if running else "normal"))
        self.stop_btn.configure(state=("normal" if running else "disabled"))
        self.pause_btn.configure(state=("normal" if running else "disabled"))
        if not running:
            self._paused = False
            try:
                self.pause_btn.configure(text="Pause")
            except Exception:
                pass

    def _on_start(self) -> None:
        if self._worker and self._worker.is_alive():
            return

        workdir = Path(self.workdir_var.get())
        if not workdir.exists():
            messagebox.showerror("Error", f"Folder not found: {workdir}")
            return

        self._stop_event.clear()
        self.progress_var.set(0.0)
        self.progress_text_var.set("0%")
        self.eta_var.set("ETA: --")
        self.finish_time_var.set("Fin: --")

        self._progress_total_units = 0
        self._progress_done_units = 0
        self._progress_t0 = 0.0
        with self._progress_lock:
            self._last_run_algo = ""
            self._last_run_scenario = ""
            self._last_run_run = None
            self._last_run_seed = None
            self._campaign_algos_count = 0
            self._campaign_total_seeds = 0

        self._reset_eta_estimator()
        self._reset_heartbeat()

        mode = self.mode_var.get()
        self._worker = threading.Thread(target=self._run_worker, args=(mode,), daemon=True)
        self._worker.start()
        self._set_running(True)

    def _on_stop(self) -> None:
        self._stop_event.set()
        # Best-effort: kill ALL subprocesses (including parallel jobs).
        self._kill_all_processes()

    def _register_proc(self, proc: subprocess.Popen) -> None:
        try:
            pid = int(proc.pid)
        except Exception:
            return
        with self._procs_lock:
            self._active_procs[pid] = proc

    def _unregister_proc(self, proc: subprocess.Popen) -> None:
        try:
            pid = int(proc.pid)
        except Exception:
            return
        with self._procs_lock:
            self._active_procs.pop(pid, None)

    def _snapshot_active_procs(self) -> List[subprocess.Popen]:
        with self._procs_lock:
            procs = list(self._active_procs.values())
        # Include current single-proc runner handle as well.
        p = self._proc
        if p is not None and p not in procs:
            procs.append(p)
        return procs

    def _kill_one_process(self, proc: subprocess.Popen) -> None:
        try:
            if proc.poll() is not None:
                return
        except Exception:
            return

        # On Windows, taskkill reliably kills the whole tree (/T).
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(int(proc.pid)), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                return
            except Exception:
                pass

        # Fallback (POSIX or taskkill failed)
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=1.0)
            return
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass

    def _kill_all_processes(self) -> None:
        procs = self._snapshot_active_procs()
        if not procs:
            return

        self._log_queue.put(f"[runner] Stop: trying to kill {len(procs)} process(es)...\n")
        for p in procs:
            self._kill_one_process(p)

    def _set_paused(self, paused: bool) -> None:
        if paused == self._paused:
            return

        if psutil is None:
            messagebox.showerror("Pause", "Pause requires psutil (pip install psutil).")
            return

        procs = self._snapshot_active_procs()
        if not procs:
            self._paused = False
            try:
                self.pause_btn.configure(text="Pause")
            except Exception:
                pass
            return

        def iter_tree(pr):
            # Include children (recursive) + the root process.
            try:
                kids = pr.children(recursive=True)
            except Exception:
                kids = []
            return list(kids) + [pr]

        # Apply to the whole tree: on Windows the Popen PID can be a shell wrapper (cmd/powershell)
        # and the real python worker is its child.
        acted = 0
        seen_pids: set[int] = set()
        for p in procs:
            try:
                root = psutil.Process(int(p.pid))
            except Exception:
                continue

            tree = iter_tree(root)
            # Order matters a bit: suspend children first; resume parent first.
            if paused:
                ordered = tree  # children then root
            else:
                ordered = list(reversed(tree))  # root then children

            for pr in ordered:
                try:
                    pid = int(pr.pid)
                    if pid in seen_pids:
                        continue
                    if paused:
                        pr.suspend()
                    else:
                        pr.resume()
                    seen_pids.add(pid)
                    acted += 1
                except Exception:
                    continue

        self._paused = bool(paused)
        if self._paused:
            self._log_queue.put(f"[runner] Pause: {acted} process(es) suspended.\n")
            try:
                self.pause_btn.configure(text="Resume")
            except Exception:
                pass
        else:
            self._log_queue.put(f"[runner] Resume: {acted} process(es) resumed.\n")
            try:
                self.pause_btn.configure(text="Pause")
            except Exception:
                pass

    def _on_pause(self) -> None:
        # Toggle pause/resume
        self._set_paused(not bool(self._paused))

    def _run_worker(self, mode: str) -> None:
        t0 = time.perf_counter()
        try:
            if bool(self.verbose_logs_var.get()):
                self._log_queue.put(f"[time] {_system_time_str()} - Operation started: mode={mode}.\n")
            if mode == "simulate":
                self._run_simulate_pipeline()
            elif mode == "excel":
                self._run_excel_only()
            elif mode == "plots":
                self._run_plots_only()
            else:
                raise ValueError(f"Unknown mode: {mode}")

            if bool(self.verbose_logs_var.get()):
                self._log_queue.put(f"[time] {_system_time_str()} - Operation finished: mode={mode}.\n")
            self._log_queue.put("[runner] Done.\n")
        except Exception as e:
            stop_like = bool(self._stop_event.is_set()) or (str(e).strip().lower() == "stop requested")

            # Persist session status for resume-on-startup.
            try:
                if mode == "simulate" and self._run_session_file is not None:
                    if stop_like:
                        self._session_set({"status": "stopped"})
                    else:
                        self._session_set({"status": "error", "error": str(e)})
            except Exception:
                pass

            if stop_like:
                self._log_queue.put("[runner] Stopped.\n")
            else:
                self._log_queue.put(f"[runner] ERROR: {e}\n")
                # Tkinter UI calls must run on the main thread.
                self._ui_show_error("Error", str(e))
            if bool(self.verbose_logs_var.get()):
                self._log_queue.put(f"[time] {_system_time_str()} - Operation finished (error): mode={mode}.\n")
        finally:
            _ = time.perf_counter() - t0
            self.root.after(0, lambda: self._set_running(False))

    def _set_progress_ui(self, pct: float, eta_s: Optional[int], *, finish_dt: Optional[datetime] = None) -> None:
        pct = max(0.0, min(100.0, float(pct)))

        def apply() -> None:
            self.progress_var.set(pct)
            self.progress_text_var.set(f"{pct:.1f}%")
            if eta_s is None:
                self.eta_var.set("ETA: --")
                self.finish_time_var.set("Fin: --")
            else:
                self.eta_var.set(f"ETA: {self._format_eta_s(eta_s)}")
                try:
                    end_dt = finish_dt if finish_dt is not None else (datetime.now() + timedelta(seconds=int(eta_s)))
                    self.finish_time_var.set("Fin: " + end_dt.strftime("%Y-%m-%d %H:%M:%S"))
                except Exception:
                    self.finish_time_var.set("Fin: --")

        self.root.after(0, apply)

    def _reset_eta_estimator(self) -> None:
        """Reset internal ETA smoothing state."""

        now = time.perf_counter()
        with self._progress_lock:
            self._eta_last_t = float(now)
            self._eta_last_done = int(self._progress_done_units)
            self._eta_rate_ema = 0.0

    def _reset_heartbeat(self) -> None:
        now = time.perf_counter()
        with self._progress_lock:
            self._hb_last_t = float(now)

    @staticmethod
    def _format_eta_s(eta_s: Optional[int]) -> str:
        if eta_s is None or int(eta_s) < 0:
            return "--"

        s = int(eta_s)
        days = s // 86400
        s = s % 86400
        hours = s // 3600
        s = s % 3600
        mins = s // 60
        secs = s % 60

        if days > 0:
            return f"{days}d {hours:02d}:{mins:02d}:{secs:02d}"
        if hours > 0:
            return f"{hours:02d}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"

    def _estimate_eta_seconds(self, *, now: float, done: int, total: int, t0: float, rate_ema: float) -> Optional[int]:
        remain = max(0, int(total) - int(done))
        if remain <= 0:
            return 0

        # Prefer smoothed rate once we have signal.
        if done >= 2 and t0 > 0.0 and rate_ema > 1e-9:
            return int(float(remain) / float(rate_ema))

        # Fallback: global average.
        # Use done>=1 so ETA shows up after the first completed run (user expectation).
        if done >= 1 and t0 > 0.0:
            elapsed = max(0.001, float(now) - float(t0))
            rate = float(done) / elapsed
            if rate > 1e-9:
                return int(float(remain) / rate)

        return None

    def _maybe_log_heartbeat(self, *, active_jobs: Optional[int] = None, total_jobs: Optional[int] = None) -> None:
        """Emit periodic mini logs for long-running operations."""

        now = time.perf_counter()
        with self._progress_lock:
            last = float(self._hb_last_t)
            if self._hb_interval_s <= 0.0:
                return
            if last > 0.0 and (now - last) < float(self._hb_interval_s):
                return
            self._hb_last_t = float(now)

            done = int(self._progress_done_units)
            total = int(self._progress_total_units)
            t0 = float(self._progress_t0)
            rate_ema = float(self._eta_rate_ema)
            last_algo = str(self._last_run_algo or "").strip()
            last_sc = str(self._last_run_scenario or "").strip()
            last_run = self._last_run_run
            last_seed = self._last_run_seed

            algos_count = int(self._campaign_algos_count)
            total_seeds = int(self._campaign_total_seeds)

        if total <= 0:
            return

        eta_s = self._estimate_eta_seconds(now=now, done=done, total=total, t0=t0, rate_ema=rate_ema)
        pct = 100.0 * float(done) / float(total)

        rate_txt = "--" if rate_ema <= 1e-9 else f"{rate_ema:.2f} u/s"
        jobs_txt = ""
        if active_jobs is not None and total_jobs is not None:
            jobs_txt = f" | jobs actifs: {int(active_jobs)}/{int(total_jobs)}"

        seeds_txt = ""
        if algos_count > 0 and total_seeds > 0:
            seeds_done = int(done) // int(algos_count)
            seeds_txt = f" | seeds~={seeds_done}/{total_seeds}"

        last_txt = ""
        if last_sc or last_algo or (last_run is not None) or (last_seed is not None):
            parts: List[str] = []
            if last_sc:
                parts.append(last_sc)
            if last_algo:
                parts.append(last_algo)
            if last_run is not None:
                parts.append(f"run={int(last_run)}")
            if last_seed is not None:
                parts.append(f"seed={int(last_seed)}")
            last_txt = " | last: " + " ".join(parts)

        self._log_queue.put(
            f"[hb] {done}/{total} ({pct:.1f}%) | rate={rate_txt} | ETA={self._format_eta_s(eta_s)}{jobs_txt}{seeds_txt}{last_txt}\n"
        )

    def _set_eta_override(self, eta_s: Optional[int]) -> None:
        """Set or clear ETA override, keeping a stable finish timestamp."""

        with self._progress_lock:
            if eta_s is None:
                self._eta_override_s = None
                self._eta_override_finish_dt = None
                return

            v = max(0, int(eta_s))
            # If the value didn't change, keep the existing finish timestamp.
            if self._eta_override_s is not None and int(self._eta_override_s) == int(v) and self._eta_override_finish_dt is not None:
                return

            self._eta_override_s = v
            try:
                self._eta_override_finish_dt = datetime.now() + timedelta(seconds=v)
            except Exception:
                self._eta_override_finish_dt = None

    def _compute_and_push_progress(self) -> None:
        total = int(self._progress_total_units)
        if total <= 0:
            return

        now = time.perf_counter()
        with self._progress_lock:
            done = int(self._progress_done_units)
            t0 = float(self._progress_t0)

            # Update smoothed rate (units/sec)
            dt = float(now - float(self._eta_last_t))
            dd = int(done - int(self._eta_last_done))
            # Important: only update the rate when progress is made.
            # Otherwise, long gaps between run lines (CPU-heavy runs) would
            # push the instantaneous rate toward 0 and make ETA *increase*.
            if dt > 0.0 and dd > 0:
                inst_rate = float(dd) / dt
                alpha = 0.25  # higher => more reactive, lower => smoother
                if self._eta_rate_ema <= 0.0:
                    self._eta_rate_ema = inst_rate
                else:
                    self._eta_rate_ema = (1.0 - alpha) * self._eta_rate_ema + alpha * inst_rate

                self._eta_last_t = float(now)
                self._eta_last_done = int(done)
            rate_ema = float(self._eta_rate_ema)

        pct = 100.0 * float(done) / float(total)

        # Prefer explicit override if provided (e.g., job-duration-based ETA).
        with self._progress_lock:
            override = self._eta_override_s
            override_finish = self._eta_override_finish_dt
        if override is not None and int(override) >= 0:
            # Keep a stable end-time display and let the remaining seconds tick down.
            if override_finish is not None:
                try:
                    rem = int((override_finish - datetime.now()).total_seconds())
                except Exception:
                    rem = int(override)
                self._set_progress_ui(pct, max(0, rem), finish_dt=override_finish)
            else:
                self._set_progress_ui(pct, int(override))
            return

        eta_s = self._estimate_eta_seconds(now=now, done=done, total=total, t0=t0, rate_ema=rate_ema)
        self._set_progress_ui(pct, eta_s)

    def _run_cmd_with_progress(self, argv: List[str], count_run_lines: bool = False) -> int:
        shell_kind = self.shell_var.get()
        workdir = Path(self.workdir_var.get())

        if bool(self.verbose_logs_var.get()):
            self._log_queue.put(f"[time] {_system_time_str()} - Command start.\n")
            self._log_queue.put(f"[cmd] {' '.join(argv)}\n")
        proc = _popen_capture(shell_kind, workdir, argv)
        self._proc = proc
        self._register_proc(proc)

        def bump_and_update() -> None:
            with self._progress_lock:
                self._progress_done_units += 1
            self._compute_and_push_progress()

        if proc.stdout is None:
            raise RuntimeError("stdout pipe missing")

        for line in proc.stdout:
            if self._stop_event.is_set():
                break

            # On Windows, lines are typically CRLF; strip both so regex matches.
            line = line.rstrip("\r\n")

            m = RUN_LINE_RE.match(line)
            if m:
                try:
                    run_idx = int(m.group("run"))
                    with self._progress_lock:
                        self._last_run_scenario = str(m.group("scenario")).strip()
                        self._last_run_algo = str(m.group("algo")).strip()
                        self._last_run_run = int(run_idx)
                        # In this mode, base_seed is unknown; leave seed unset.
                        self._last_run_seed = None
                except Exception:
                    pass

            # Log policy:
            # - Always show per-run lines (most useful)
            # - Other output only when Detailed logs is enabled
            if m:
                try:
                    sc = str(m.group("scenario")).strip()
                    algo = str(m.group("algo")).strip()
                    run_idx = int(m.group("run"))
                    self._log_queue.put(f"[run] {sc} | {algo} | run={run_idx}\n")
                except Exception:
                    self._log_queue.put(line)
            else:
                if bool(self.verbose_logs_var.get()):
                    self._log_queue.put(line)

            if m and count_run_lines:
                # each printed run = one completed unit across all jobs
                bump_and_update()

        if self._stop_event.is_set() and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

        try:
            rc = proc.wait()
        finally:
            self._unregister_proc(proc)
            if self._proc is proc:
                self._proc = None
        if bool(self.verbose_logs_var.get()):
            self._log_queue.put(f"[time] {_system_time_str()} - Command end (rc={int(rc)}).\n")
        return int(rc)

    def _run_parallel_jobs(self, jobs: List[Tuple[str, int, int]], base_argv_builder, *, algos: List[str], total_seeds: int) -> None:
        """Run multiple jobs in parallel while aggregating progress from stdout."""

        procs: List[Tuple[str, subprocess.Popen]] = []
        threads: List[threading.Thread] = []
        job_t0: dict[str, float] = {}
        job_completed: set[str] = set()
        job_durations: List[float] = []

        # ETA calibration logic (user request):
        # - Baseline algorithm (default FSS): watch run index progression across parallel jobs
        # - Each time ALL eligible jobs reach the next baseline run (r -> r+1), recompute dt
        # - Use the latest (smoothed) dt to estimate remaining wall time, ignoring fast algos
        baseline_algo = (self.baseline_algo_var.get() or "FSS").strip().upper()
        fast_algos = set(ETA_FAST_ALGOS)
        heavy_algos = [a for a in [str(x).strip() for x in algos] if a and a.upper() not in fast_algos]
        heavy_count = int(len(heavy_algos))
        heavy_total_units = int(max(0, int(total_seeds)) * max(0, heavy_count))

        eligible_prefixes = {pref for pref, _base_seed, runs_job in jobs if int(runs_job) >= 2}
        eligible_n = int(len(eligible_prefixes))
        eta_state_lock = threading.Lock()
        baseline_seen: dict[int, set[str]] = {}
        baseline_all_t: dict[int, float] = {}
        eta_dt_ema: Optional[float] = None
        eta_done_heavy_units: int = 0

        # Tee each job's stdout to a log file (so we can inspect later if something goes wrong).
        # Logs are deleted on a fully successful pipeline (user request).
        logs_root: Optional[Path] = self._job_logs_dir
        if logs_root is not None:
            try:
                logs_root.mkdir(parents=True, exist_ok=True)
                if bool(self.verbose_logs_var.get()):
                    self._log_queue.put(f"[runner] Job logs dir: {logs_root}\n")
            except Exception:
                logs_root = None

        def reader(prefix: str, base_seed: int, proc: subprocess.Popen) -> None:
            try:
                if proc.stdout is None:
                    return

                log_fh = None
                if logs_root is not None:
                    try:
                        safe_prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", str(prefix))
                        log_path = Path(logs_root) / f"{safe_prefix}.log"
                        log_fh = open(log_path, "a", encoding="utf-8", errors="replace")
                    except Exception:
                        log_fh = None

                try:
                    for raw in proc.stdout:
                        if log_fh is not None:
                            try:
                                log_fh.write(raw)
                            except Exception:
                                pass

                        if self._stop_event.is_set():
                            break

                        # On Windows, lines are typically CRLF; strip both so regex matches.
                        line = raw.rstrip("\r\n")
                        m = RUN_LINE_RE.match(line)

                        if m:
                            try:
                                run_idx = int(m.group("run"))
                                seed = int(base_seed) + int(run_idx)
                                algo_u = str(m.group("algo")).strip().upper()

                                # Track done units for ETA excluding fast algos.
                                if heavy_total_units > 0:
                                    with eta_state_lock:
                                        if algo_u not in fast_algos:
                                            eta_done_heavy_units += 1

                                # Calibration markers based on baseline algo.
                                # Re-evaluate every time all parallel jobs complete the next baseline run.
                                if eligible_n > 0 and prefix in eligible_prefixes and algo_u == baseline_algo:
                                    now = time.perf_counter()
                                    with eta_state_lock:
                                        s = baseline_seen.get(int(run_idx))
                                        if s is None:
                                            s = set()
                                            baseline_seen[int(run_idx)] = s
                                        s.add(prefix)

                                        # When all jobs have reached this run_idx, record the timestamp.
                                        if int(run_idx) not in baseline_all_t and len(s) >= eligible_n:
                                            baseline_all_t[int(run_idx)] = float(now)

                                            prev = int(run_idx) - 1
                                            if prev in baseline_all_t:
                                                dt = max(0.001, float(now) - float(baseline_all_t[prev]))
                                                alpha = 0.35  # smoothing factor for dt updates
                                                if eta_dt_ema is None:
                                                    eta_dt_ema = float(dt)
                                                else:
                                                    eta_dt_ema = (1.0 - alpha) * float(eta_dt_ema) + alpha * float(dt)

                                                self._log_queue.put(
                                                    f"[eta] Recalibrated: baseline={baseline_algo}, step={prev}->{int(run_idx)}, dt={dt:.2f}s, ema={float(eta_dt_ema):.2f}s over {eligible_n} job(s).\n"
                                                )

                                with self._progress_lock:
                                    self._last_run_scenario = str(m.group("scenario")).strip()
                                    self._last_run_algo = str(m.group("algo")).strip()
                                    self._last_run_run = int(run_idx)
                                    self._last_run_seed = int(seed)

                                # Progress: each printed run = one completed unit across all jobs
                                with self._progress_lock:
                                    self._progress_done_units += 1
                            except Exception:
                                pass

                        # Log policy:
                        # - Always show per-run lines with seed
                        # - Other output only when Detailed logs is enabled
                        if m:
                            try:
                                sc = str(m.group("scenario")).strip()
                                algo = str(m.group("algo")).strip()
                                run_idx = int(m.group("run"))
                                seed = int(base_seed) + int(run_idx)
                                self._log_queue.put(f"[run] {sc} | {algo} | run={run_idx} | seed={seed}\n")
                            except Exception:
                                self._log_queue.put(line)
                        else:
                            if bool(self.verbose_logs_var.get()):
                                self._log_queue.put(line)

                finally:
                    if log_fh is not None:
                        try:
                            log_fh.flush()
                        except Exception:
                            pass
                        try:
                            log_fh.close()
                        except Exception:
                            pass

            except Exception as e:
                self._log_queue.put(f"[runner] reader error for {prefix}: {e}\n")

        try:
            # start all processes
            for prefix, base_seed, runs_job in jobs:
                if self._stop_event.is_set():
                    raise RuntimeError("Stop requested")

                argv = base_argv_builder(prefix, base_seed, runs_job)
                if bool(self.verbose_logs_var.get()):
                    self._log_queue.put(
                        f"[time] {_system_time_str()} - Job start: {prefix} (base_seed={base_seed}, runs={runs_job}).\n"
                    )
                proc = _popen_capture(self.shell_var.get(), Path(self.workdir_var.get()), argv)
                self._register_proc(proc)
                procs.append((prefix, proc))
                job_t0[prefix] = time.perf_counter()
                t = threading.Thread(target=reader, args=(prefix, int(base_seed), proc), daemon=True)
                threads.append(t)
                t.start()

            # monitor until all finished
            last_dt_step_used: Optional[float] = None
            last_active_jobs_used: int = -1
            last_completed_jobs_count: int = 0
            while True:
                if self._stop_event.is_set():
                    for _pref, p in procs:
                        self._kill_one_process(p)
                    raise RuntimeError("Stop requested")

                # Job-duration ETA override (wait for at least one finished job)
                now = time.perf_counter()
                for pref, p in procs:
                    if p.poll() is None:
                        continue
                    if pref in job_completed:
                        continue
                    st = job_t0.get(pref)
                    if st is None:
                        continue
                    job_completed.add(pref)
                    job_durations.append(max(0.0, float(now - float(st))))

                completed_jobs_count = int(len(job_completed))

                # ETA override: prefer calibration-based estimate once available.
                active = [p for _pref, p in procs if p.poll() is None]
                active_jobs = int(max(1, len(active)))
                with eta_state_lock:
                    dt_step = eta_dt_ema
                    done_heavy = int(eta_done_heavy_units)

                if dt_step is not None and heavy_total_units > 0 and heavy_count > 0:
                    # Update override only when the calibration step time changes materially
                    # (i.e., after a new baseline step is completed), or when active
                    # parallelism changes.
                    need_update = False
                    if last_dt_step_used is None:
                        need_update = True
                    elif abs(float(dt_step) - float(last_dt_step_used)) / max(1e-9, float(last_dt_step_used)) >= 0.05:
                        need_update = True
                    if int(active_jobs) != int(last_active_jobs_used):
                        need_update = True

                    if need_update:
                        remain_heavy = max(0, int(heavy_total_units) - int(done_heavy))
                        # dt_step approximates wall time for a baseline seed-step across jobs;
                        # distribute across heavy algos.
                        per_unit_s = float(dt_step) / float(max(1, heavy_count))
                        eta_s = int((float(remain_heavy) / float(active_jobs)) * per_unit_s)
                        self._set_eta_override(max(0, int(eta_s)))
                        last_dt_step_used = float(dt_step)
                        last_active_jobs_used = int(active_jobs)
                else:
                    # Fallback: job-duration-based ETA once we have at least one finished job.
                    if job_durations:
                        # Only update when another job completes (signal changes).
                        if completed_jobs_count != int(last_completed_jobs_count):
                            avg_dur = float(sum(job_durations) / max(1, len(job_durations)))
                            rem_est = 0.0
                            for pref, p in procs:
                                if p.poll() is not None:
                                    continue
                                st = job_t0.get(pref, now)
                                rem_est = max(rem_est, max(0.0, avg_dur - float(now - float(st))))
                            self._set_eta_override(int(rem_est))
                            last_completed_jobs_count = int(completed_jobs_count)
                    else:
                        self._set_eta_override(None)

                self._compute_and_push_progress()

                self._maybe_log_heartbeat(active_jobs=len(active), total_jobs=len(procs))

                if not active:
                    break

                time.sleep(0.25)

            # join readers quickly
            for t in threads:
                t.join(timeout=0.5)

            # check return codes
            bad: List[Tuple[str, int]] = []
            for pref, p in procs:
                rc = int(p.poll() if p.poll() is not None else p.wait())
                if bool(self.verbose_logs_var.get()):
                    self._log_queue.put(f"[time] {_system_time_str()} - Job end: {pref} (rc={rc}).\n")
                if rc != 0:
                    bad.append((pref, rc))

            if bad:
                msg = "; ".join(f"{pref}: rc={rc}" for pref, rc in bad)
                raise RuntimeError(f"Simulation failed for: {msg}")
        finally:
            for _pref, p in procs:
                try:
                    self._unregister_proc(p)
                except Exception:
                    pass
            self._set_eta_override(None)

    def _run_simulate_pipeline(self) -> None:
        # If resuming, reuse the previous session dir.
        resume_dir = self._resume_session_dir
        self._resume_session_dir = None

        scenario_tag = self.scenario_var.get()
        bs = self.bs_var.get()

        try:
            total_seeds = int(str(self.seeds_total_var.get()).strip())
        except Exception:
            raise ValueError("Total seeds must be an integer")
        if total_seeds <= 0:
            raise ValueError("Total seeds must be > 0")

        try:
            parallel_jobs = int(str(self.parallel_jobs_var.get()).strip())
        except Exception:
            raise ValueError("Parallel jobs must be an integer")
        if parallel_jobs <= 0:
            raise ValueError("Parallel jobs must be > 0")
        rounds = int(self.rounds_var.get())
        if rounds < 200 or rounds > 5000:
            raise ValueError("Rounds must be between 200 and 5000")

        algos = [a.strip() for a in self.algos_var.get().split(",") if a.strip()]
        if not algos:
            raise ValueError("Empty algorithms list")

        # Create (or load) a persisted run session for resume-on-startup.
        workdir = Path(self.workdir_var.get())
        if resume_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = workdir / RUNS_DIR_NAME / stamp
            session_file = session_dir / RUN_SESSION_FILE
            self._run_session_dir = session_dir
            self._run_session_file = session_file
            manifest: Dict[str, Any] = {
                "version": 1,
                "session_id": stamp,
                "mode": "simulate",
                "status": "running",
                "started_at": _system_time_str(),
                "updated_at": _system_time_str(),
                "params": {
                    "workdir": str(workdir),
                    "scenario": str(scenario_tag),
                    "bs": str(bs),
                    "seeds_total": int(total_seeds),
                    "parallel_jobs": int(parallel_jobs),
                    "rounds": int(rounds),
                    "algos": ",".join(algos),
                    "baseline_algo": str(self.baseline_algo_var.get().strip() or "FSS"),
                    "correction": str(self.correction_var.get().strip() or "holm"),
                },
                "steps": {"simulate": "pending", "merge": "pending", "excel": "pending", "plots": "pending"},
                "artifacts": {},
            }
            try:
                _write_json_atomic(session_file, manifest)
            except Exception:
                pass
        else:
            session_dir = Path(resume_dir)
            session_file = session_dir / RUN_SESSION_FILE
            self._run_session_dir = session_dir
            self._run_session_file = session_file
            self._session_set({"status": "running"})

        scenario_flag = "--only_s1_100" if scenario_tag == "S1_100" else "--only_s2_100"
        prefix_base = _campaign_prefix_base(scenario_tag, bs, rounds)
        jobs = _make_jobs(prefix_base, total_seeds=total_seeds, parallel_jobs=parallel_jobs)
        if not jobs:
            raise ValueError("No jobs to run (check seeds/jobs)")

        self._session_set({"artifacts": {"prefix_base": str(prefix_base)}})

        # Each printed run line corresponds to one (seed, algo) unit.
        total_units = int(total_seeds) * len(algos)

        self._progress_total_units = int(total_units)
        with self._progress_lock:
            self._progress_done_units = 0
            self._progress_t0 = time.perf_counter()
            self._campaign_algos_count = int(len(algos))
            self._campaign_total_seeds = int(total_seeds)

        self._reset_eta_estimator()

        self._log_queue.put(
            f"[runner] Simulation: {scenario_tag}, bs={bs}, total_seeds={total_seeds}, jobs={len(jobs)}, rounds={rounds}, algos={len(algos)}\n"
        )
        if bool(self.verbose_logs_var.get()):
            self._log_queue.put(f"[runner] Total units (seed×algo): {total_units}\n")
            self._log_queue.put(f"[time] {_system_time_str()} - Step start: Simulation.\n")

        def build_job_argv(prefix: str, base_seed: int, runs_job: int) -> List[str]:
            return [
                sys.executable,
                "-u",
                "-m",
                "scripts.run_experiments",
                "--prefix",
                prefix,
                "--runs",
                str(int(runs_job)),
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

        # Decide which steps to run (resume-aware).
        manifest = _read_json(self._run_session_file) if self._run_session_file is not None else None
        steps = manifest.get("steps") if isinstance(manifest, dict) else None
        if not isinstance(steps, dict):
            steps = {}

        # Simulation
        simulate_done = str(steps.get("simulate") or "") == "done"
        if not simulate_done:
            self._session_set_step("simulate", "running")
            # Organize per-job logs under the session dir.
            try:
                if self._run_session_dir is not None:
                    self._job_logs_dir = Path(self._run_session_dir) / "job_logs"
            except Exception:
                self._job_logs_dir = None

            # Run jobs in parallel.
            self._run_parallel_jobs(jobs, build_job_argv, algos=algos, total_seeds=total_seeds)
            self._session_set_step("simulate", "done")
        else:
            self._log_queue.put("[runner] Resume: Simulation step already done; skipping.\n")

        if bool(self.verbose_logs_var.get()):
            self._log_queue.put(f"[time] {_system_time_str()} - Step end: Simulation.\n")

        # Merge
        if self._stop_event.is_set():
            self._session_set({"status": "stopped"})
            raise RuntimeError("Stop requested")

        manifest = _read_json(self._run_session_file) if self._run_session_file is not None else None
        steps = manifest.get("steps") if isinstance(manifest, dict) else None
        if not isinstance(steps, dict):
            steps = {}
        merge_done = str(steps.get("merge") or "") == "done"

        merge_prefix = prefix_base
        summary_all = workdir / f"{merge_prefix}_ALL_summary.csv"
        ts_all = workdir / f"{merge_prefix}_ALL_timeseries.csv"
        if not merge_done:
            self._session_set_step("merge", "running")
            if bool(self.verbose_logs_var.get()):
                self._log_queue.put(f"[runner] Merge: {merge_prefix}\n")
                self._log_queue.put(f"[time] {_system_time_str()} - Step start: Merge.\n")

            merge_argv = [sys.executable, "-u", "-m", "scripts.merge_csvs", "--prefix", merge_prefix, "--out_dir", "."]
            rc = self._run_cmd_with_progress(merge_argv, count_run_lines=False)
            if rc != 0:
                raise RuntimeError("Merge failed")

            # Verify outputs exist
            if not summary_all.exists() or not ts_all.exists():
                raise RuntimeError("Merge finished but *_ALL_* files are missing")

            self._session_set_step("merge", "done")
            self._session_set(
                {
                    "artifacts": {
                        "prefix_base": str(prefix_base),
                        "merge_prefix": str(merge_prefix),
                        "summary_all": str(summary_all),
                        "timeseries_all": str(ts_all),
                    }
                }
            )

            if bool(self.verbose_logs_var.get()):
                self._log_queue.put("[runner] Merge OK.\n")
                self._log_queue.put(f"[time] {_system_time_str()} - Step end: Merge.\n")
        else:
            if summary_all.exists() and ts_all.exists():
                self._log_queue.put("[runner] Resume: Merge step already done; skipping.\n")
            else:
                # If outputs are missing, force merge again.
                self._session_set_step("merge", "pending")
                raise RuntimeError("Resume: merge was marked done but *_ALL_* files are missing")

        # Excel report
        out_xlsx = workdir / f"{merge_prefix}_report.xlsx"
        ref_algo = self.baseline_algo_var.get().strip() or "FSS"
        correction = self.correction_var.get().strip() or "holm"

        manifest = _read_json(self._run_session_file) if self._run_session_file is not None else None
        steps = manifest.get("steps") if isinstance(manifest, dict) else None
        if not isinstance(steps, dict):
            steps = {}
        excel_done = str(steps.get("excel") or "") == "done"

        if not excel_done:
            self._session_set_step("excel", "running")
            if bool(self.verbose_logs_var.get()):
                self._log_queue.put(f"[runner] Excel: {out_xlsx.name}\n")
                self._log_queue.put(f"[time] {_system_time_str()} - Step start: Excel.\n")
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
                raise RuntimeError("Excel generation failed")

            if not out_xlsx.exists():
                raise RuntimeError("Excel file not created")

            self._session_set_step("excel", "done")
            self._session_set({"artifacts": {"excel": str(out_xlsx)}})

            if bool(self.verbose_logs_var.get()):
                self._log_queue.put(f"[time] {_system_time_str()} - Step end: Excel.\n")
        else:
            if out_xlsx.exists():
                self._log_queue.put("[runner] Resume: Excel step already done; skipping.\n")
            else:
                self._session_set_step("excel", "pending")
                raise RuntimeError("Resume: excel was marked done but the .xlsx is missing")

        # Optional plots
        plot_types = self._selected_plot_types()
        manifest = _read_json(self._run_session_file) if self._run_session_file is not None else None
        steps = manifest.get("steps") if isinstance(manifest, dict) else None
        if not isinstance(steps, dict):
            steps = {}
        plots_done = str(steps.get("plots") or "") == "done"

        if plot_types and not plots_done:
            self._session_set_step("plots", "running")
            if bool(self.verbose_logs_var.get()):
                self._log_queue.put(f"[runner] Plots: {', '.join(plot_types)}\n")
                self._log_queue.put(f"[time] {_system_time_str()} - Step start: Plots.\n")
            self._run_plots_for_excel(Path(out_xlsx), workdir, plot_types)
            self._session_set_step("plots", "done")
            if bool(self.verbose_logs_var.get()):
                self._log_queue.put(f"[time] {_system_time_str()} - Step end: Plots.\n")
        elif plot_types and plots_done:
            self._log_queue.put("[runner] Resume: Plots step already done; skipping.\n")

        self._log_queue.put("[runner] Pipeline complete.\n")

        self._session_set({"status": "complete"})

        # If the pipeline completed successfully, delete job logs (user request).
        try:
            if self._job_logs_dir is not None and not self._stop_event.is_set():
                shutil.rmtree(self._job_logs_dir, ignore_errors=True)
        except Exception:
            pass
        finally:
            self._job_logs_dir = None
            self._run_session_dir = None
            self._run_session_file = None

        self.root.after(0, self._refresh_dynamic_lists)

    def _run_excel_only(self) -> None:
        workdir = Path(self.workdir_var.get())
        pref = self.excel_prefix_var.get().strip()
        if not pref:
            raise ValueError("Select a prefix")

        summary_all = workdir / f"{pref}_ALL_summary.csv"
        ts_all = workdir / f"{pref}_ALL_timeseries.csv"
        if not summary_all.exists() or not ts_all.exists():
            raise ValueError("Missing *_ALL_summary.csv / *_ALL_timeseries.csv for this prefix")

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
        if bool(self.verbose_logs_var.get()):
            self._log_queue.put(f"[time] {_system_time_str()} - Step start: Excel.\n")
        rc = self._run_cmd_with_progress(argv, count_run_lines=False)
        if rc != 0:
            raise RuntimeError("Excel generation failed")

        self._log_queue.put(f"[runner] Excel OK: {out}\n")
        if bool(self.verbose_logs_var.get()):
            self._log_queue.put(f"[time] {_system_time_str()} - Step end: Excel.\n")

    def _run_plots_only(self) -> None:
        # If user selected multiple Excel reports, run scenario comparison instead.
        if self.compare_excels:
            if len(self.compare_excels) < 2:
                raise ValueError("Compare scenarios: select at least 2 Excel files")

            plot_types = self._selected_plot_types()
            if not plot_types:
                self._log_queue.put("[runner] No plot selected.\n")
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
                    raise ValueError("Metric selection: pick at least one metric")
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
                        "Some metrics are not present in all selected Excel files:\n"
                        f"- Manquantes: {miss_txt}\n\n"
                        f"Common metrics available:\n{common_txt}"
                    )
                # Filter any non-common metrics (defensive)
                metrics_to_run = [m for m in metrics_to_run if m in common_metrics]

            algos_raw = self.compare_algos_var.get().strip() or "FSS"
            algos = self._validate_compare_algos_input(algos_raw)

            metrics_disp = self._format_short_list(metrics_to_run, max_items=10)
            self._log_queue.put(
                f"[runner] Compare scenarios: metrics={metrics_disp}, algos={algos}, plots={','.join(plot_types)}\n"
            )

            # Progress: one command per metric
            with self._progress_lock:
                self._progress_total_units = max(1, len(metrics_to_run))
                self._progress_done_units = 0
                self._progress_t0 = time.perf_counter()

            self._reset_eta_estimator()
            self._compute_and_push_progress()

            if bool(self.verbose_logs_var.get()):
                self._log_queue.put(f"[time] {_system_time_str()} - Step start: Plots (scenario comparison).\n")

            out_dir = str(Path(self.workdir_var.get()))
            for metric in metrics_to_run:
                if self._stop_event.is_set():
                    raise RuntimeError("Stop requested")

                if bool(self.verbose_logs_var.get()):
                    self._log_queue.put(f"[runner] -> metric: {metric}\n")
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
                    raise RuntimeError(f"Scenario plot generation failed (metric={metric})")

                with self._progress_lock:
                    self._progress_done_units += 1
                self._compute_and_push_progress()

            self._log_queue.put("[runner] Scenario comparison plots OK.\n")
            if bool(self.verbose_logs_var.get()):
                self._log_queue.put(f"[time] {_system_time_str()} - Step end: Plots (scenario comparison).\n")
            self.root.after(0, self._refresh_dynamic_lists)
            return

        excel_path = self.excel_exist_cb.get().strip()
        if not excel_path:
            raise ValueError("Select an Excel file")

        plot_types = self._selected_plot_types()
        if not plot_types:
            self._log_queue.put("[runner] No plot selected.\n")
            return

        workdir = Path(self.workdir_var.get())
        if bool(self.verbose_logs_var.get()):
            self._log_queue.put(f"[time] {_system_time_str()} - Step start: Plots.\n")
        self._run_plots_for_excel(Path(excel_path), workdir, plot_types)

        self._log_queue.put("[runner] Plots OK.\n")
        if bool(self.verbose_logs_var.get()):
            self._log_queue.put(f"[time] {_system_time_str()} - Step end: Plots.\n")

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
                raise RuntimeError("Plot generation failed")

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
                    raise ValueError("Metric selection: pick at least one metric")
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
                    raise RuntimeError("Plot generation failed")

        # Line plot: run once for the current metric.
        if line_only:
            if mode in {"all", "select"}:
                if bool(self.verbose_logs_var.get()):
                    self._log_queue.put("[runner] Note: 'Line' is generated only for the current metric (timeseries).\n")

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
                raise RuntimeError("Plot generation failed")

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
    
    