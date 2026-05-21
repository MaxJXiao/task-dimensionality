#!/usr/bin/env python3
"""
run_experiment.py — master runner for the LoRA rank sweep.

Executes 4 tasks × 8 conditions (7 LoRA ranks + full fine-tuning) = 32 runs.
Results are saved to results/{task}/{rank}/training_log.csv after each run.

Usage
-----
  python run_experiment.py                           # all 32 runs
  python run_experiment.py --task sst2              # all 8 conditions for SST-2
  python run_experiment.py --rank 8                 # rank-8 LoRA across all 4 tasks
  python run_experiment.py --task sst2 --rank full  # single run (debugging)
  python run_experiment.py --device cpu             # force CPU (slow but useful)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import timedelta

import torch
from transformers import AutoTokenizer

from src.config import LORA_RANKS, INCLUDE_FULL_PARAM_BASELINE, TASK_REGISTRY, MODEL_REGISTRY, DEFAULT_MODEL
from src.model import get_lora_model, get_full_model, trainable_param_summary
from src.train import train_one_run

# ---------------------------------------------------------------------------
# Condition list — order: cheapest first so partial runs are still useful
# ---------------------------------------------------------------------------

ALL_RANKS: list[str] = [str(r) for r in LORA_RANKS]
if INCLUDE_FULL_PARAM_BASELINE:
    ALL_RANKS.append("full")

ALL_TASKS: list[str] = list(TASK_REGISTRY.keys())   # [sst2, cola, snli, squad2]

RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA requisite rank experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=ALL_TASKS,
        metavar="TASK",
        help=f"Run only this task. Choices: {ALL_TASKS}",
    )
    parser.add_argument(
        "--rank",
        type=str,
        default=None,
        metavar="RANK",
        help=f"Run only this rank condition (integer or 'full'). Choices: {ALL_RANKS}",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(MODEL_REGISTRY.keys()),
        metavar="MODEL",
        help=f"Base model to fine-tune. Choices: {list(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="DEVICE",
        help="PyTorch device string, e.g. 'cuda', 'cuda:1', 'cpu'. Default: auto.",
    )
    return parser.parse_args()


def _validate_rank(rank_str: str) -> None:
    if rank_str not in ALL_RANKS:
        print(f"[ERROR] --rank '{rank_str}' is not valid. Choose from: {ALL_RANKS}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Formatting helpers for Colab readability
# ---------------------------------------------------------------------------

def _banner(text: str, width: int = 70) -> str:
    pad = (width - len(text) - 2) // 2
    return "\n" + "=" * width + f"\n{'=' * pad} {text} {'=' * pad}\n" + "=" * width


def _fmt_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _print_run_header(
    run_idx: int,
    total_runs: int,
    task_name: str,
    rank_label: str,
    model_name: str,
    param_summary: dict,
) -> None:
    print(_banner(f"Run {run_idx}/{total_runs}  |  task={task_name}  rank={rank_label}"))
    print(f"  Model       : {model_name}")
    print(f"  Total params: {param_summary['total']:,}")
    print(f"  Trainable   : {param_summary['trainable']:,}  ({param_summary['trainable_pct']}%)")
    print()


def _print_run_result(
    task_name: str,
    rank_label: str,
    elapsed: float,
    final_metric: float | str,
    sota: float,
    eta_seconds: float | None,
) -> None:
    gap = f"{final_metric - sota:+.2f}" if isinstance(final_metric, float) else "—"
    eta_str = f"ETA remaining: {_fmt_duration(eta_seconds)}" if eta_seconds else ""
    print(
        f"\n  [DONE] {task_name} | r={rank_label} | "
        f"final metric={final_metric}  (SOTA {sota}, gap {gap}) | "
        f"elapsed {_fmt_duration(elapsed)}  {eta_str}\n"
    )


# ---------------------------------------------------------------------------
# Summary JSON — updated after every completed run
# ---------------------------------------------------------------------------

def _load_summary(summary_path: str) -> dict:
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)
    return {}


def _save_summary(summary: dict, summary_path: str) -> None:
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    model_name = args.model
    model_slug = model_name.replace("/", "--")
    summary_path = os.path.join(RESULTS_DIR, model_slug, "run_summary.json")

    # Validate and resolve --rank
    if args.rank is not None:
        _validate_rank(args.rank)
        rank_conditions = [args.rank]
    else:
        rank_conditions = ALL_RANKS

    task_names = [args.task] if args.task else ALL_TASKS

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_runs = len(task_names) * len(rank_conditions)

    print(_banner("LoRA Requisite Rank Experiment"))
    print(f"  Base model : {model_name}")
    print(f"  Device     : {device}")
    print(f"  Tasks      : {task_names}")
    print(f"  Ranks      : {rank_conditions}")
    print(f"  Total runs : {total_runs}")
    print()

    # Tokenizer is stateless after setup, so one load serves all 32 runs.
    print(f"Loading tokenizer ({model_name}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_cfg = MODEL_REGISTRY[model_name]
    if model_cfg.architecture == "decoder":
        # Decoder models must pad on the left so the causal attention mask
        # never attends to padding tokens that precede the real sequence.
        tokenizer.padding_side = "left"
        # Llama has no dedicated pad token; eos is the standard substitute.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Tokenizer ready.\n")

    summary = _load_summary(summary_path)
    run_idx = 0
    experiment_start = time.time()
    completed_times: list[float] = []

    for task_name in task_names:
        task = TASK_REGISTRY[task_name]

        for rank_label in rank_conditions:
            run_idx += 1
            run_key = f"{task_name}/{rank_label}"

            # Colab sessions disconnect mid-sweep; re-running the script picks up
            # here and skips anything already marked "done" in run_summary.json.
            if run_key in summary and summary[run_key].get("status") == "done":
                print(f"  [SKIP] {run_key} already complete — delete from run_summary.json to re-run.")
                continue

            # Fresh model every run — reusing would carry over learned weights
            # from the previous rank condition and corrupt the weight initialisation.
            if rank_label == "full":
                model = get_full_model(model_name, task.task_type, task.num_labels)
            else:
                model = get_lora_model(int(rank_label), model_name, task.task_type, task.num_labels)

            param_summary = trainable_param_summary(model)
            _print_run_header(run_idx, total_runs, task_name, rank_label, model_name, param_summary)

            run_start = time.time()
            final_metric: float | str = "—"
            status = "error"

            try:
                log_rows = train_one_run(task, model, tokenizer, rank_label, device, model_name)

                # Pull the final evaluated metric from the log
                evaluated = [r for r in log_rows if r["test_metric"] != ""]
                if evaluated:
                    final_metric = evaluated[-1]["test_metric"]
                status = "done"

            except Exception:
                print(f"\n[ERROR] Run {run_key} failed:")
                traceback.print_exc()
                status = "error"

            elapsed = time.time() - run_start
            completed_times.append(elapsed)

            # ETA: average completed-run time × remaining runs
            remaining = total_runs - run_idx
            eta = (sum(completed_times) / len(completed_times)) * remaining if remaining else None

            _print_run_result(task_name, rank_label, elapsed, final_metric, task.sota_baseline, eta)

            summary[run_key] = {
                "status": status,
                "final_metric": final_metric,
                "sota_baseline": task.sota_baseline,
                "elapsed_s": round(elapsed, 1),
                "trainable_params": param_summary["trainable"],
                "trainable_pct": param_summary["trainable_pct"],
            }
            _save_summary(summary, summary_path)

    total_elapsed = time.time() - experiment_start
    completed = sum(1 for v in summary.values() if v.get("status") == "done")
    errors = sum(1 for v in summary.values() if v.get("status") == "error")

    print(_banner("Experiment Complete"))
    print(f"  Total time : {_fmt_duration(total_elapsed)}")
    print(f"  Completed  : {completed}/{total_runs}")
    if errors:
        print(f"  Errors     : {errors}  (see traceback output above)")
    print(f"  Summary    : {summary_path}")
    print(f"  Logs       : {RESULTS_DIR}/{model_slug}/{{task}}/{{rank}}/training_log.csv")
    print()


if __name__ == "__main__":
    main()
