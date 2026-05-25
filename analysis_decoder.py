"""
analysis_decoder.py — results analysis for the decoder LoRA rank sweep.

Plots per (model, variant):
  1. plots/decoder/{model_slug}/{variant}/{task}_ppl_curves.png
       Perplexity vs training step — one line per LoRA rank.

  2. Final quality metric vs LoRA rank (connected scatter, one point per rank):
       mbpp_indist_rank_sweep.png  — MBPP held-out pass@1 (in-distribution)
       mbpp_ood_rank_sweep.png     — HumanEval pass@1 (out-of-distribution)
       gsm8k_rank_sweep.png        — GSM8K exact match
       trivia_qa_rank_sweep.png    — TriviaQA F1

Usage
-----
  python analysis_decoder.py
  python analysis_decoder.py --task gsm8k
  python analysis_decoder.py --variant attn_mlp
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

from src_decoder.config import LORA_RANKS, TASK_REGISTRY, MODELS, MODEL_REGISTRY

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots") / "decoder"

RANK_CMAP = cm.plasma
FULL_COLOR = "black"
FULL_LINESTYLE = "--"
FULL_LINEWIDTH = 2.0
RANK_LINEWIDTH = 1.8
ALPHA = 0.9
ZEROSHOT_COLOR = "steelblue"
ZEROSHOT_LINESTYLE = "-."
ZEROSHOT_LINEWIDTH = 1.5

LORA_RANK_LABELS: list[str] = [str(r) for r in LORA_RANKS]
ALL_RANK_LABELS: list[str] = LORA_RANK_LABELS + ["full"]

MODEL_DISPLAY: dict[str, str] = {
    m.hf_name.replace("/", "--"): m.hf_name.split("/")[-1]
    for m in MODELS
}
ALL_MODEL_SLUGS: list[str] = [m.hf_name.replace("/", "--") for m in MODELS]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(val) -> float | None:
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _rank_color(rank_label: str) -> tuple:
    idx = LORA_RANK_LABELS.index(rank_label)
    t = 0.15 + 0.75 * (idx / max(len(LORA_RANK_LABELS) - 1, 1))
    return RANK_CMAP(t)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_logs(
    model_slug: str,
    variant: str,
    task_name: str | None = None,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Load all training_log.csv files for a model/variant combo.

    Returns {(task_name, rank_label): DataFrame (full, unfiltered)}.
    """
    task_glob = task_name or "*"
    pattern = str(RESULTS_DIR / model_slug / variant / task_glob / "*" / "training_log.csv")
    paths = glob.glob(pattern)
    if not paths:
        print(f"[WARNING] No training logs found: {pattern}")
    data: dict[tuple[str, str], pd.DataFrame] = {}
    for path in sorted(paths):
        parts = Path(path).parts  # (..., model_slug, variant, task, rank, training_log.csv)
        task, rank = parts[-3], parts[-2]
        data[(task, rank)] = pd.read_csv(path)
    return data


def load_zero_shot(model_slug: str, variant: str) -> dict[str, dict]:
    """Return {task: {"indist": float|None, "ood": float|None}} from run_summary.json."""
    path = RESULTS_DIR / model_slug / variant / "run_summary.json"
    if not path.exists():
        return {}
    with open(path) as f:
        summary = json.load(f)
    scores: dict[str, dict] = {}
    for key, val in summary.items():
        if key.endswith("/baseline") and val.get("status") == "done":
            task = key.split("/")[0]
            scores[task] = {
                "indist": _to_float(val.get("final_metric")),
                "ood": _to_float(val.get("final_metric_ood")),
            }
    return scores


def _quality_mask(df: pd.DataFrame) -> pd.Series:
    quality_cols = ["final_pass_at_1", "final_pass_at_1_ood", "final_em_math", "final_f1", "final_rouge"]
    mask = pd.Series(False, index=df.index)
    for col in quality_cols:
        if col in df.columns:
            mask |= pd.to_numeric(df[col], errors="coerce").notna()
    return mask


def _ppl_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Eval perplexity rows (test_metric), excluding the final quality-metric row."""
    ppl_df = df[~_quality_mask(df)].copy()
    ppl_df["test_metric"] = pd.to_numeric(ppl_df["test_metric"], errors="coerce")
    return ppl_df.dropna(subset=["test_metric"])


def _train_ppl_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Training perplexity (exp(train_loss)) for every step except the final quality row."""
    train_df = df[~_quality_mask(df)].copy()
    train_df["train_loss"] = pd.to_numeric(train_df["train_loss"], errors="coerce")
    train_df = train_df.dropna(subset=["train_loss"])
    train_df["train_ppl"] = np.exp(train_df["train_loss"])
    return train_df


def _final_metrics(df: pd.DataFrame, task_type: str) -> dict[str, float | None]:
    """Extract final quality metric(s) from the last row of a training log."""
    if df.empty:
        return {"indist": None, "ood": None}
    last = df.iloc[-1]
    if task_type == "code_generation":
        return {
            "indist": _to_float(last.get("final_pass_at_1")),
            "ood": _to_float(last.get("final_pass_at_1_ood")),
        }
    elif task_type == "math_reasoning":
        return {"indist": _to_float(last.get("final_em_math")), "ood": None}
    elif task_type == "generative_qa":
        return {"indist": _to_float(last.get("final_f1")), "ood": None}
    else:
        return {"indist": _to_float(last.get("test_metric")), "ood": None}


# ---------------------------------------------------------------------------
# Plot: perplexity training curves
# ---------------------------------------------------------------------------

def plot_ppl_curves(
    model_slug: str,
    task_name: str,
    task_cfg,
    rank_dfs: dict[str, pd.DataFrame],
    out_dir: Path,
    variant: str = "attn",
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))

    lora_present = [r for r in LORA_RANK_LABELS if r in rank_dfs]
    for rank_label in lora_present:
        df = rank_dfs[rank_label]
        color = _rank_color(rank_label)

        train_df = _train_ppl_rows(df)
        if not train_df.empty:
            ax.plot(
                train_df["step"], train_df["train_ppl"],
                color=color, linewidth=RANK_LINEWIDTH * 0.7,
                alpha=0.35, linestyle="--", label="_nolegend_",
            )

        eval_df = _ppl_rows(df)
        if not eval_df.empty:
            ax.plot(
                eval_df["step"], eval_df["test_metric"],
                color=color, linewidth=RANK_LINEWIDTH,
                alpha=ALPHA, label=f"r={rank_label}",
            )

    if "full" in rank_dfs:
        df = rank_dfs["full"]
        train_df = _train_ppl_rows(df)
        if not train_df.empty:
            ax.plot(
                train_df["step"], train_df["train_ppl"],
                color=FULL_COLOR, linestyle=":", linewidth=FULL_LINEWIDTH * 0.7,
                alpha=0.35, label="_nolegend_",
            )
        eval_df = _ppl_rows(df)
        if not eval_df.empty:
            ax.plot(
                eval_df["step"], eval_df["test_metric"],
                color=FULL_COLOR, linestyle=FULL_LINESTYLE,
                linewidth=FULL_LINEWIDTH, alpha=ALPHA, label="Full fine-tune",
            )

    sm = cm.ScalarMappable(cmap=RANK_CMAP, norm=plt.Normalize(vmin=LORA_RANKS[0], vmax=LORA_RANKS[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("LoRA rank (r)", fontsize=10)

    style_handles = [
        Line2D([0], [0], color="gray", linewidth=RANK_LINEWIDTH, alpha=ALPHA, label="Eval ppl (solid)"),
        Line2D([0], [0], color="gray", linewidth=RANK_LINEWIDTH * 0.7, linestyle="--", alpha=0.5, label="Train ppl (dashed)"),
    ]
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=style_handles + existing_handles,
        labels=["Eval ppl (solid)", "Train ppl (dashed)"] + existing_labels,
        loc="upper right", fontsize=9, framealpha=0.85,
    )

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title(
        f"{task_cfg.display_name} — train (dashed) vs eval (solid) perplexity\n"
        f"model: {MODEL_DISPLAY.get(model_slug, model_slug)}",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task_name}_{variant}_ppl_curves.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot: final quality metric vs rank
# ---------------------------------------------------------------------------

def plot_final_metric_vs_rank(
    model_slug: str,
    metric_name: str,
    rank_scores: dict[str, float],
    zero_shot_score: float | None,
    out_path: Path,
    title: str,
) -> Path:
    """Connected scatter: one point per rank condition, x = rank label, y = metric (%)."""
    x_labels = [r for r in ALL_RANK_LABELS if r in rank_scores]
    y_vals = [rank_scores[r] for r in x_labels]
    x_pos = list(range(len(x_labels)))

    if not x_labels:
        return out_path

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(x_pos, y_vals, color="gray", linewidth=1.2, alpha=0.5, zorder=1)

    spread = max(max(y_vals) - min(y_vals), 1.0)
    for xi, yi, rl in zip(x_pos, y_vals, x_labels):
        color = FULL_COLOR if rl == "full" else _rank_color(rl)
        marker = "D" if rl == "full" else "o"
        ax.scatter(xi, yi, color=color, s=80, zorder=3, marker=marker, alpha=ALPHA)
        ax.text(xi, yi + spread * 0.04 + 0.2, f"{yi:.1f}",
                ha="center", va="bottom", fontsize=9)

    if zero_shot_score is not None:
        ax.axhline(
            y=zero_shot_score, color=ZEROSHOT_COLOR, linestyle=ZEROSHOT_LINESTYLE,
            linewidth=ZEROSHOT_LINEWIDTH, alpha=ALPHA,
            label=f"Zero-shot ({zero_shot_score:.2f})",
        )
        ax.legend(loc="lower right", fontsize=9, framealpha=0.85)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"r={r}" if r != "full" else "Full FT" for r in x_labels],
        fontsize=10,
    )
    ax.set_xlabel("LoRA rank", fontsize=12)
    ax.set_ylabel(f"{metric_name} (%)", fontsize=12)
    ax.set_title(f"{title}\nmodel: {MODEL_DISPLAY.get(model_slug, model_slug)}", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    y_lo = max(0.0, min(y_vals) - spread * 0.25)
    y_hi = min(100.0, max(y_vals) + spread * 0.35)
    ax.set_ylim(y_lo, y_hi)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse decoder LoRA rank sweep results")
    parser.add_argument("--task", type=str, default=None, choices=list(TASK_REGISTRY.keys()))
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help="HuggingFace model ID (default: all)",
    )
    parser.add_argument("--variant", type=str, default="attn", choices=["attn", "attn_mlp"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    variant = args.variant
    model_slugs = [args.model.replace("/", "--")] if args.model else ALL_MODEL_SLUGS
    task_names = [args.task] if args.task else list(TASK_REGISTRY.keys())

    for model_slug in model_slugs:
        out_dir = PLOTS_DIR / model_slug / variant
        zero_shot = load_zero_shot(model_slug, variant)
        all_logs = load_all_logs(model_slug, variant)

        if not all_logs:
            print(f"[SKIP] No results for {MODEL_DISPLAY.get(model_slug, model_slug)}")
            continue

        for task_name in task_names:
            task_cfg = TASK_REGISTRY[task_name]

            rank_dfs: dict[str, pd.DataFrame] = {
                rank: df
                for (tn, rank), df in all_logs.items()
                if tn == task_name
            }
            if not rank_dfs:
                print(f"[SKIP] {MODEL_DISPLAY.get(model_slug, model_slug)} / {task_name} — no results")
                continue

            # Perplexity training curves
            out = plot_ppl_curves(model_slug, task_name, task_cfg, rank_dfs, out_dir, variant)
            print(f"[PLOT] {task_cfg.display_name} | ppl curves  → {out}")

            # Collect final quality metrics across rank conditions
            indist_scores: dict[str, float] = {}
            ood_scores: dict[str, float] = {}
            for rank_label, df in rank_dfs.items():
                m = _final_metrics(df, task_cfg.task_type)
                if m["indist"] is not None:
                    indist_scores[rank_label] = m["indist"]
                if m["ood"] is not None:
                    ood_scores[rank_label] = m["ood"]

            zs = zero_shot.get(task_name, {})

            if task_cfg.task_type == "code_generation":
                if indist_scores:
                    out = plot_final_metric_vs_rank(
                        model_slug, "Pass@1", indist_scores,
                        zero_shot_score=zs.get("indist"),
                        out_path=out_dir / f"mbpp_indist_{variant}_rank_sweep.png",
                        title="MBPP held-out — pass@1 vs LoRA rank",
                    )
                    print(f"[PLOT] {task_cfg.display_name} | MBPP in-dist  → {out}")
                if ood_scores:
                    out = plot_final_metric_vs_rank(
                        model_slug, "Pass@1", ood_scores,
                        zero_shot_score=zs.get("ood"),
                        out_path=out_dir / f"mbpp_ood_{variant}_rank_sweep.png",
                        title="HumanEval (OOD) — pass@1 vs LoRA rank",
                    )
                    print(f"[PLOT] {task_cfg.display_name} | HumanEval OOD → {out}")

            elif task_cfg.task_type == "math_reasoning":
                if indist_scores:
                    out = plot_final_metric_vs_rank(
                        model_slug, "Exact Match", indist_scores,
                        zero_shot_score=zs.get("indist"),
                        out_path=out_dir / f"{task_name}_{variant}_rank_sweep.png",
                        title=f"{task_cfg.display_name} — exact match vs LoRA rank",
                    )
                    print(f"[PLOT] {task_cfg.display_name} | EM vs rank    → {out}")

            elif task_cfg.task_type == "generative_qa":
                if indist_scores:
                    out = plot_final_metric_vs_rank(
                        model_slug, "F1", indist_scores,
                        zero_shot_score=zs.get("indist"),
                        out_path=out_dir / f"{task_name}_{variant}_rank_sweep.png",
                        title=f"{task_cfg.display_name} — F1 vs LoRA rank",
                    )
                    print(f"[PLOT] {task_cfg.display_name} | F1 vs rank    → {out}")


if __name__ == "__main__":
    main()
