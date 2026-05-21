"""
analysis.py — results analysis and visualisation for the LoRA rank sweep.

Phase 1 outputs (per model × task)
------------------------------------
  plots/{model_slug}/{task}_rank_sweep.png
      Test metric vs training step for all rank conditions, with SOTA baseline.

Phase 2 outputs (cross-architecture)
--------------------------------------
  plots/{task}_cross_model_rstar.png
      Bar chart of requisite rank R* for each model on a given task.
      Consistent R* across models supports the task-dimensionality hypothesis.

  plots/heatmap_requisite_rank.png
      Heatmap — rows = tasks, columns = models, cells = R*.

Usage
-----
  python analysis.py                           # all models, all tasks
  python analysis.py --task sst2              # all models, single task
  python analysis.py --model roberta-base     # single model, all tasks
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

from src.config import LORA_RANKS, TASK_REGISTRY, TaskConfig, MODELS, MODEL_REGISTRY

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

# Half a metric point — small enough to be meaningful, large enough to absorb
# the evaluation variance typical of single-run GLUE / SQuAD experiments.
REQUISITE_THRESHOLD: float = 0.5

# Visual style — rank sweep plots
RANK_CMAP = cm.plasma
FULL_COLOR = "black"
FULL_LINESTYLE = "--"
FULL_LINEWIDTH = 2.0
SOTA_COLOR = "crimson"
SOTA_LINESTYLE = ":"
RANK_LINEWIDTH = 1.5
ALPHA = 0.85

ALL_RANK_LABELS: list[str] = [str(r) for r in LORA_RANKS] + ["full"]

# Model slug → short display name for plot axes and table columns
MODEL_DISPLAY: dict[str, str] = {
    m.hf_name.replace("/", "--"): m.hf_name.split("/")[-1]
    for m in MODELS
}
ALL_MODEL_SLUGS: list[str] = [m.hf_name.replace("/", "--") for m in MODELS]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(
    task_name: str | None = None,
    model_slug: str | None = None,
) -> dict[tuple[str, str, str], pd.DataFrame]:
    """
    Load all training_log.csv files from results/{model_slug}/{task}/{rank}/.

    Returns a dict keyed by (model_slug, task_name, rank_label) whose values
    are DataFrames with columns [step, train_loss, test_metric], filtered to
    checkpoint rows where test_metric is non-empty.
    """
    task_glob = task_name or "*"
    model_glob = model_slug or "*"
    pattern = str(RESULTS_DIR / model_glob / task_glob / "*" / "training_log.csv")
    paths = glob.glob(pattern)

    if not paths:
        print(f"[WARNING] No training_log.csv files found matching: {pattern}")

    data: dict[tuple[str, str, str], pd.DataFrame] = {}
    for path in sorted(paths):
        parts = Path(path).parts  # (..., model_slug, task, rank, "training_log.csv")
        m_slug, task, rank = parts[-4], parts[-3], parts[-2]

        df = pd.read_csv(path)
        df = df[df["test_metric"].notna() & (df["test_metric"] != "")]
        df["test_metric"] = pd.to_numeric(df["test_metric"], errors="coerce")
        df = df.dropna(subset=["test_metric"])
        if "exact_match" in df.columns:
            df["exact_match"] = pd.to_numeric(df["exact_match"], errors="coerce")

        if not df.empty:
            data[(m_slug, task, rank)] = df

    return data


# ---------------------------------------------------------------------------
# Rank sweep plot (per model × task)
# ---------------------------------------------------------------------------

def _rank_color(rank_label: str, n_lora_ranks: int) -> tuple:
    """Map a LoRA rank label to a colour from the sequential colourmap."""
    lora_labels = [str(r) for r in LORA_RANKS]
    idx = lora_labels.index(rank_label)
    # Sample from 0.15–0.90 of plasma to avoid the near-white and near-black
    # extremes, which are hard to distinguish on screen or when printed.
    t = 0.15 + 0.75 * (idx / max(n_lora_ranks - 1, 1))
    return RANK_CMAP(t)


def plot_rank_sweep(
    model_slug: str,
    task_name: str,
    task_data: dict[str, pd.DataFrame],
    task_cfg: TaskConfig,
) -> Path:
    """
    Plot test metric vs step for all rank conditions of one (model, task) pair.
    Saves to plots/{model_slug}/{task}_rank_sweep.png.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    lora_ranks_present = [r for r in ALL_RANK_LABELS if r != "full" and r in task_data]
    n_lora = len(lora_ranks_present)

    for rank_label in lora_ranks_present:
        df = task_data[rank_label]
        color = _rank_color(rank_label, n_lora)
        ax.plot(df["step"], df["test_metric"], color=color, linewidth=RANK_LINEWIDTH,
                alpha=ALPHA, label=f"r={rank_label}")

    if "full" in task_data:
        df = task_data["full"]
        ax.plot(df["step"], df["test_metric"], color=FULL_COLOR, linestyle=FULL_LINESTYLE,
                linewidth=FULL_LINEWIDTH, alpha=ALPHA, label="Full fine-tune")

    ax.axhline(y=task_cfg.sota_baseline, color=SOTA_COLOR, linestyle=SOTA_LINESTYLE,
               linewidth=1.5, label=f"Published baseline ({task_cfg.sota_baseline})")
    ax.axhspan(task_cfg.sota_baseline - REQUISITE_THRESHOLD, task_cfg.sota_baseline,
               alpha=0.06, color=SOTA_COLOR, label=f"±{REQUISITE_THRESHOLD}-pt threshold")

    sm = cm.ScalarMappable(cmap=RANK_CMAP,
                           norm=plt.Normalize(vmin=LORA_RANKS[0], vmax=LORA_RANKS[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("LoRA rank (r)", fontsize=10)

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel(f"{task_cfg.metric}", fontsize=12)
    ax.set_title(
        f"{task_cfg.display_name} ({task_name}) — rank sweep\n"
        f"model: {MODEL_DISPLAY.get(model_slug, model_slug)}   "
        f"threshold: within {REQUISITE_THRESHOLD} pts of baseline",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3)

    out_dir = PLOTS_DIR / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task_name}_rank_sweep.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Requisite rank computation
# ---------------------------------------------------------------------------

def compute_requisite_rank(
    task_data: dict[str, pd.DataFrame],
    sota_baseline: float,
) -> tuple[str | None, float | None]:
    """
    Return (requisite_rank, peak_metric): the lowest integer LoRA rank whose
    peak test metric falls within REQUISITE_THRESHOLD of sota_baseline.
    Returns (None, None) if no rank meets the criterion.
    """
    for rank_label in [str(r) for r in LORA_RANKS]:
        if rank_label not in task_data:
            continue
        peak = float(task_data[rank_label]["test_metric"].max())
        if peak >= sota_baseline - REQUISITE_THRESHOLD:
            return rank_label, peak
    return None, None


# ---------------------------------------------------------------------------
# Cross-model R* bar chart (per task)
# ---------------------------------------------------------------------------

def plot_cross_model_rstar(
    task_name: str,
    task_cfg: TaskConfig,
    rows_for_task: list[dict],
) -> Path:
    """
    Bar chart of R* across all models for a single task.

    Log₂-scale y-axis aligns with the exponential rank sweep.
    Models with no qualifying rank are shown as grey hatched bars at the
    top of the scale labelled "N/A".
    Saves to plots/{task}_cross_model_rstar.png.
    """
    n = len(rows_for_task)
    fig, ax = plt.subplots(figsize=(max(6, n * 2.0), 5))

    for i, row in enumerate(rows_for_task):
        rstar = row["requisite_rank"]
        label = MODEL_DISPLAY.get(row["model_slug"], row["model_slug"])
        if rstar is not None:
            rank_val = int(rstar)
            color = _rank_color(rstar, len(LORA_RANKS))
            ax.bar(i, rank_val, color=color, alpha=ALPHA, width=0.6)
            ax.text(i, rank_val * 1.18, str(rank_val), ha="center", va="bottom",
                    fontsize=12, fontweight="bold")
        else:
            ax.bar(i, LORA_RANKS[-1], color="lightgray", alpha=0.65, width=0.6,
                   hatch="//", edgecolor="gray")
            ax.text(i, LORA_RANKS[-1] * 1.18, "N/A", ha="center", va="bottom",
                    fontsize=12, color="dimgray")

    ax.set_yscale("log", base=2)
    ax.set_ylim(0.6, LORA_RANKS[-1] * 2.5)
    ax.set_yticks(LORA_RANKS)
    ax.set_yticklabels([str(r) for r in LORA_RANKS], fontsize=10)
    ax.set_xticks(range(n))
    ax.set_xticklabels(
        [MODEL_DISPLAY.get(r["model_slug"], r["model_slug"]) for r in rows_for_task],
        fontsize=11,
    )
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Requisite rank R*", fontsize=12)
    ax.set_title(
        f"{task_cfg.display_name} ({task_name}) — requisite rank R* across models\n"
        f"SOTA baseline: {task_cfg.sota_baseline}   threshold: ±{REQUISITE_THRESHOLD} pts",
        fontsize=12,
    )
    ax.grid(True, axis="y", alpha=0.3)

    PLOTS_DIR.mkdir(exist_ok=True)
    out_path = PLOTS_DIR / f"{task_name}_cross_model_rstar.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Global heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(summary_rows: list[dict]) -> Path:
    """
    Heatmap of R* — rows = tasks, columns = models.

    Colour encodes log₂(R*): green = low rank needed, red = high rank needed.
    Grey cells = no qualifying rank found or no results yet.
    Saves to plots/heatmap_requisite_rank.png.
    """
    # Use canonical ordering; only include axes for which data exists
    present_tasks = [t for t in TASK_REGISTRY if any(r["task"] == t for r in summary_rows)]
    present_models = [m for m in ALL_MODEL_SLUGS if any(r["model_slug"] == m for r in summary_rows)]

    n_tasks, n_models = len(present_tasks), len(present_models)

    data = np.full((n_tasks, n_models), np.nan)
    annot: list[list[str]] = [["" for _ in range(n_models)] for _ in range(n_tasks)]

    for row in summary_rows:
        ti = present_tasks.index(row["task"]) if row["task"] in present_tasks else -1
        mi = present_models.index(row["model_slug"]) if row["model_slug"] in present_models else -1
        if ti < 0 or mi < 0:
            continue
        rstar = row["requisite_rank"]
        if rstar is not None:
            data[ti, mi] = np.log2(int(rstar))
            annot[ti][mi] = str(rstar)
        else:
            annot[ti][mi] = "N/A"

    task_labels = [TASK_REGISTRY[t].display_name for t in present_tasks]
    model_labels = [MODEL_DISPLAY.get(m, m) for m in present_models]

    fig, ax = plt.subplots(figsize=(max(6, n_models * 2.4), max(3.5, n_tasks * 1.4)))

    cmap = plt.get_cmap("RdYlGn_r").copy()
    # Grey for NaN cells (no data / no qualifying rank); white would be
    # indistinguishable from the light end of the colormap.
    cmap.set_bad(color="#d0d0d0")

    # Log₂ scale so the colormap is evenly spaced across [1,2,4,8,16,32,64].
    # A linear scale would compress the low-rank differences into near-zero.
    vmin, vmax = 0.0, float(np.log2(LORA_RANKS[-1]))   # 0 → log2(64) = 6
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Cell annotations
    for ti in range(n_tasks):
        for mi in range(n_models):
            text = annot[ti][mi] or "—"
            val = data[ti, mi]
            if np.isnan(val):
                fg = "dimgray"
            elif val > vmax * 0.6:
                fg = "white"
            else:
                fg = "black"
            ax.text(mi, ti, text, ha="center", va="center",
                    fontsize=13, fontweight="bold", color=fg)

    # Colorbar with rank labels instead of log values
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_ticks([np.log2(r) for r in LORA_RANKS])
    cbar.set_ticklabels([str(r) for r in LORA_RANKS])
    cbar.set_label("Requisite rank R*", fontsize=10)

    ax.set_xticks(range(n_models))
    ax.set_yticks(range(n_tasks))
    ax.set_xticklabels(model_labels, fontsize=11)
    ax.set_yticklabels(task_labels, fontsize=11)

    # Minor tick grid lines between cells
    ax.set_xticks(np.arange(n_models + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_tasks + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title(
        "Requisite rank R* — tasks × models\n"
        "green = lower rank sufficient   red = higher rank required   grey = not achieved / no data",
        fontsize=12,
    )

    PLOTS_DIR.mkdir(exist_ok=True)
    out_path = PLOTS_DIR / "heatmap_requisite_rank.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(summary_rows: list[dict]) -> None:
    col_widths = {
        "Model": 20,
        "Task": 26,
        "Requisite R*": 14,
        "Peak F1/metric": 15,
        "Exact Match": 13,
        "SOTA baseline": 15,
        "Gap (F1)": 10,
    }
    header = "".join(k.ljust(v) for k, v in col_widths.items())
    divider = "-" * sum(col_widths.values())

    print("\n" + divider)
    print("  Requisite Rank Summary — cross-architecture")
    print(divider)
    print(header)
    print(divider)

    for row in summary_rows:
        peak = row["peak_metric"]
        em = row.get("exact_match_peak")
        sota = row["sota_baseline"]
        gap = f"{peak - sota:+.2f}" if peak is not None else "—"
        model_label = MODEL_DISPLAY.get(row["model_slug"], row["model_slug"])
        print(
            model_label.ljust(col_widths["Model"])
            + row["task_display"].ljust(col_widths["Task"])
            + (row["requisite_rank"] or "none").ljust(col_widths["Requisite R*"])
            + (f"{peak:.2f}" if peak is not None else "—").ljust(col_widths["Peak F1/metric"])
            + (f"{em:.2f}" if em is not None else "—").ljust(col_widths["Exact Match"])
            + str(sota).ljust(col_widths["SOTA baseline"])
            + gap
        )

    print(divider + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse LoRA rank sweep results")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=list(TASK_REGISTRY.keys()),
        help="Analyse only this task (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help="Analyse only this model (default: all). Use HuggingFace model ID.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    model_slug_filter = args.model.replace("/", "--") if args.model else None
    model_slugs = [model_slug_filter] if model_slug_filter else ALL_MODEL_SLUGS
    task_names = [args.task] if args.task else list(TASK_REGISTRY.keys())

    all_data = load_results(task_name=args.task, model_slug=model_slug_filter)

    if not all_data:
        print("No results found. Run run_experiment.py first.")
        return

    summary_rows: list[dict] = []

    # --- Per-(model, task) rank sweep plots ---
    for model_slug in model_slugs:
        for task_name in task_names:
            task_cfg = TASK_REGISTRY[task_name]

            task_data: dict[str, pd.DataFrame] = {
                rank: df
                for (ms, tn, rank), df in all_data.items()
                if ms == model_slug and tn == task_name
            }

            if not task_data:
                print(f"[SKIP] No results for {MODEL_DISPLAY.get(model_slug, model_slug)} / {task_name}")
                continue

            out_path = plot_rank_sweep(model_slug, task_name, task_data, task_cfg)
            print(f"[PLOT] {MODEL_DISPLAY.get(model_slug, model_slug)} / {task_cfg.display_name} → {out_path}")

            req_rank, peak = compute_requisite_rank(task_data, task_cfg.sota_baseline)

            em_peak: float | None = None
            if task_cfg.task_type == "span_extraction":
                # EM peak is independent of R* — we want the best EM the model
                # achieved at any rank, not just at the requisite rank.
                em_vals = [
                    float(df["exact_match"].max())
                    for df in task_data.values()
                    if "exact_match" in df.columns and not df["exact_match"].isna().all()
                ]
                if em_vals:
                    em_peak = max(em_vals)

            summary_rows.append({
                "model_slug": model_slug,
                "task": task_name,
                "task_display": task_cfg.display_name,
                "requisite_rank": req_rank,
                "peak_metric": peak,
                "exact_match_peak": em_peak,
                "sota_baseline": task_cfg.sota_baseline,
            })

    if not summary_rows:
        return

    # --- Cross-model R* plots (one per task, requires ≥2 models) ---
    if len(model_slugs) >= 2:
        for task_name in task_names:
            rows_for_task = [r for r in summary_rows if r["task"] == task_name]
            if len(rows_for_task) < 2:
                continue
            task_cfg = TASK_REGISTRY[task_name]
            out_path = plot_cross_model_rstar(task_name, task_cfg, rows_for_task)
            print(f"[PLOT] Cross-model R* / {task_cfg.display_name} → {out_path}")

        # --- Global heatmap ---
        out_path = plot_heatmap(summary_rows)
        print(f"[PLOT] Heatmap → {out_path}")

    print_summary_table(summary_rows)


if __name__ == "__main__":
    main()
