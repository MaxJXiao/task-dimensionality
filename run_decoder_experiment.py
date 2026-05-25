#!/usr/bin/env python3
"""
run_decoder_experiment.py — LoRA rank sweep for decoder models (plain fp16, no QLoRA).

Mirrors run_experiment.py but targets decoder-only models (Llama) and uses
smaller batch sizes (8 / 4 / 4) to compensate for the higher VRAM footprint of
unquantized fp16 weights.

Usage
-----
  python run_decoder_experiment.py                                        # all tasks, attn variant, Llama-3.2-1B
  python run_decoder_experiment.py --variant attn_mlp                    # QKV + FFN variant
  python run_decoder_experiment.py --task billsum                        # single task
  python run_decoder_experiment.py --rank 8                              # rank-8 across all tasks
  python run_decoder_experiment.py --task billsum --rank full            # single run (debugging)
  python run_decoder_experiment.py --device cpu                          # force CPU
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import replace as dc_replace
from datetime import timedelta

import torch
from transformers import AutoTokenizer

from datasets import load_dataset

from torch.utils.data import DataLoader
from src_decoder.data_loader import _collate_with_strings, build_ood_eval_dataset, get_dataloaders
from src.train import evaluate_classification, evaluate_squad, evaluate_perplexity
from src_decoder.config import (
    INCLUDE_FULL_PARAM_BASELINE,
    LORA_RANKS,
    MODEL_REGISTRY,
    DEFAULT_MODEL,
    TASK_REGISTRY,
    TEST_TRAIN_SAMPLES,
    TEST_EVAL_SAMPLES,
    TEST_EPOCHS,
    TEST_EVAL_STEPS,
)
from src_decoder.model import get_lora_model, get_full_model, trainable_param_summary
from src_decoder.train import train_one_run, _training_cfg_for_task, evaluate_pass_at_1, evaluate_generative_qa, evaluate_math_em

# ---------------------------------------------------------------------------
# Condition list
# ---------------------------------------------------------------------------

ALL_RANKS: list[str] = ["baseline"] + [str(r) for r in LORA_RANKS]
if INCLUDE_FULL_PARAM_BASELINE:
    ALL_RANKS.append("full")

ALL_TASKS: list[str] = list(TASK_REGISTRY.keys())

RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA rank sweep — decoder models (fp16, no QLoRA)",
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
        help=f"Decoder model to fine-tune. Choices: {list(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="DEVICE",
        help="PyTorch device string, e.g. 'cuda', 'cuda:1', 'cpu'. Default: auto.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="attn",
        choices=["attn", "attn_mlp"],
        help="LoRA target scope: 'attn' (q/v_proj) or 'attn_mlp' (q/v + FFN). Default: attn.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Smoke-test mode: uses TEST_* constants from src_decoder/config.py. "
             "Validates the full pipeline quickly without real training.",
    )
    return parser.parse_args()


def _validate_rank(rank_str: str) -> None:
    if rank_str not in ALL_RANKS:
        print(f"[ERROR] --rank '{rank_str}' is not valid. Choose from: {ALL_RANKS}")
        sys.exit(1)


def _apply_test_mode(task):
    """Patch a TaskConfig for smoke-test mode using config TEST_* constants."""
    ood = task.ood_eval
    if ood is not None:
        ood = dc_replace(ood, max_eval_samples=TEST_EVAL_SAMPLES)
    return dc_replace(
        task,
        max_train_samples=TEST_TRAIN_SAMPLES,
        max_eval_samples=TEST_EVAL_SAMPLES,
        num_epochs=TEST_EPOCHS,
        eval_steps=TEST_EVAL_STEPS,
        ood_eval=ood,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _banner(text: str, width: int = 70) -> str:
    pad = (width - len(text) - 2) // 2
    return "\n" + "=" * width + f"\n{'=' * pad} {text} {'=' * pad}\n" + "=" * width


def _fmt_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _print_run_header(run_idx, total_runs, task_name, rank_label, model_name, param_summary):
    print(_banner(f"Run {run_idx}/{total_runs}  |  task={task_name}  rank={rank_label}"))
    print(f"  Model       : {model_name}")
    print(f"  Total params: {param_summary['total']:,}")
    print(f"  Trainable   : {param_summary['trainable']:,}  ({param_summary['trainable_pct']}%)")
    print()


def _print_run_result(task_name, rank_label, elapsed, final_metric, sota, eta_seconds):
    if sota is not None and isinstance(final_metric, float):
        gap = f"{final_metric - sota:+.2f}"
        sota_str = f"SOTA {sota}, gap {gap}"
    else:
        sota_str = "no SOTA baseline"
    eta_str = f"ETA remaining: {_fmt_duration(eta_seconds)}" if eta_seconds else ""
    print(
        f"\n  [DONE] {task_name} | r={rank_label} | "
        f"final metric={final_metric}  ({sota_str}) | "
        f"elapsed {_fmt_duration(elapsed)}  {eta_str}\n"
    )


# ---------------------------------------------------------------------------
# Summary JSON
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
# Zero-shot baseline
# ---------------------------------------------------------------------------

def _eval_baseline(task, model, tokenizer, device) -> float:
    if not getattr(model, "hf_device_map", None):
        model.to(device)
    model.eval()
    loaders = get_dataloaders(task, tokenizer, _training_cfg_for_task(task), eval_only=True)
    if task.task_type == "classification":
        return round(evaluate_classification(model, loaders["eval"], task, device), 4)
    elif task.task_type == "causal_lm":
        return round(evaluate_perplexity(model, loaders["eval_ppl"], device), 4)
    elif task.task_type == "code_generation":
        return round(evaluate_pass_at_1(model, loaders["eval"], tokenizer, device), 4)
    elif task.task_type == "generative_qa":
        _, f1 = evaluate_generative_qa(model, loaders["eval"], tokenizer, device)
        return round(f1, 4)
    elif task.task_type == "math_reasoning":
        return round(evaluate_math_em(model, loaders["eval"], tokenizer, device), 4)
    else:
        raw_val = load_dataset(task.dataset_name, split="validation")
        raw_lookup = {ex["id"]: ex for ex in raw_val}
        f1, _ = evaluate_squad(model, loaders["eval"], loaders["eval_dataset"], raw_lookup, device)
        return round(f1, 4)


def _eval_baseline_ood(task, model, tokenizer, device) -> float | None:
    if task.ood_eval is None:
        return None
    ood_ds = build_ood_eval_dataset(task.ood_eval, tokenizer)
    ood_loader = DataLoader(
        ood_ds, batch_size=4,
        shuffle=False, num_workers=0, pin_memory=True,
        collate_fn=_collate_with_strings,
    )
    if task.task_type == "code_generation":
        return round(evaluate_pass_at_1(model, ood_loader, tokenizer, device), 4)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    model_name = args.model
    variant = args.variant
    model_slug = model_name.replace("/", "--")
    summary_path = os.path.join(RESULTS_DIR, model_slug, variant, "run_summary.json")

    model_cfg = MODEL_REGISTRY[model_name]

    if args.rank is not None:
        _validate_rank(args.rank)
        rank_conditions = [args.rank]
    else:
        rank_conditions = [
            r for r in ALL_RANKS
            if r in ("baseline", "full") or int(r) <= model_cfg.max_lora_rank
        ]

    task_names = [args.task] if args.task else ALL_TASKS

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_runs = len(task_names) * len(rank_conditions)

    print(_banner("LoRA Decoder Rank Experiment  (fp16, no QLoRA)"))
    print(f"  Base model : {model_name}")
    print(f"  Variant    : {variant}")
    print(f"  Device     : {device}")
    print(f"  Tasks      : {task_names}")
    print(f"  Ranks      : {rank_conditions}")
    print(f"  Total runs : {total_runs}")
    print()

    if args.test:
        print(f"  [TEST MODE] {TEST_TRAIN_SAMPLES} train / {TEST_EVAL_SAMPLES} eval samples, {TEST_EPOCHS} epoch, eval_steps={TEST_EVAL_STEPS}\n")

    print(f"Loading tokenizer ({model_name}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Decoder models pad on the left so causal attention never attends to padding.
    tokenizer.padding_side = "left"
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
        if args.test:
            task = _apply_test_mode(task)

        for rank_label in rank_conditions:
            run_idx += 1
            run_key = f"{task_name}/{rank_label}"

            if run_key in summary and summary[run_key].get("status") == "done":
                print(f"  [SKIP] {run_key} already complete — delete from run_summary.json to re-run.")
                continue

            if rank_label == "baseline":
                model = get_full_model(model_name, task.task_type, task.num_labels)
                for p in model.parameters():
                    p.requires_grad = False
            elif rank_label == "full":
                model = get_full_model(model_name, task.task_type, task.num_labels)
            else:
                model = get_lora_model(int(rank_label), model_name, task.task_type, task.num_labels, variant)

            param_summary = trainable_param_summary(model)
            _print_run_header(run_idx, total_runs, task_name, rank_label, model_name, param_summary)

            run_start = time.time()
            final_metric: float | str = "—"
            final_metric_ood: float | str | None = None
            status = "error"

            try:
                if rank_label == "baseline":
                    final_metric = _eval_baseline(task, model, tokenizer, device)
                    final_metric_ood = _eval_baseline_ood(task, model, tokenizer, device)
                else:
                    log_rows = train_one_run(task, model, tokenizer, rank_label, device, model_name, variant)
                    evaluated = [r for r in log_rows if r["test_metric"] != ""]
                    if evaluated:
                        final_metric = evaluated[-1]["test_metric"]
                    ood_rows = [r for r in log_rows if r.get("final_pass_at_1_ood") != ""]
                    if ood_rows:
                        final_metric_ood = ood_rows[-1]["final_pass_at_1_ood"]
                status = "done"

            except Exception:
                print(f"\n[ERROR] Run {run_key} failed:")
                traceback.print_exc()
                status = "error"

            elapsed = time.time() - run_start
            completed_times.append(elapsed)

            remaining = total_runs - run_idx
            eta = (sum(completed_times) / len(completed_times)) * remaining if remaining else None

            _print_run_result(task_name, rank_label, elapsed, final_metric, task.sota_baseline, eta)

            if final_metric_ood is not None:
                print(f"  [OOD]  {task_name} | r={rank_label} | HumanEval pass@1={final_metric_ood}")

            summary[run_key] = {
                "status": status,
                "final_metric": final_metric,
                "final_metric_ood": final_metric_ood,
                "sota_baseline": task.sota_baseline,
                "elapsed_s": round(elapsed, 1),
                "trainable_params": param_summary["trainable"],
                "trainable_pct": param_summary["trainable_pct"],
            }
            _save_summary(summary, summary_path)

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_elapsed = time.time() - experiment_start
    completed = sum(1 for v in summary.values() if v.get("status") == "done")
    errors = sum(1 for v in summary.values() if v.get("status") == "error")

    print(_banner("Experiment Complete"))
    print(f"  Total time : {_fmt_duration(total_elapsed)}")
    print(f"  Completed  : {completed}/{total_runs}")
    if errors:
        print(f"  Errors     : {errors}  (see traceback output above)")
    print(f"  Summary    : {summary_path}")
    print(f"  Logs       : {RESULTS_DIR}/{model_slug}/{variant}/{{task}}/{{rank}}/training_log.csv")
    print()


if __name__ == "__main__":
    main()
