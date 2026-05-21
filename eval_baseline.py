"""
eval_baseline.py
----------------
Evaluate untrained (zero-shot) roberta-base on all four experiment tasks.

Loads the model with no LoRA adapters and no training, runs the same
evaluation functions used by train.py, and saves results to:
    results/roberta-base/baseline/baseline_results.json

Usage:
    python eval_baseline.py                        # all 4 tasks
    python eval_baseline.py --task cola            # single task
    python eval_baseline.py --device cpu           # force CPU
"""

import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering

from src.config import TASK_REGISTRY, DEFAULT_MODEL, RESULTS_DIR
from src.data_loader import get_dataloaders
from src.train import evaluate_classification, evaluate_squad


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME  = DEFAULT_MODEL
MODEL_SLUG  = MODEL_NAME.replace("/", "--")
OUTPUT_DIR  = os.path.join(RESULTS_DIR, MODEL_SLUG, "baseline")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "baseline_results.json")
TASK_ORDER  = list(TASK_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Model loader (no LoRA, no training)
# ---------------------------------------------------------------------------
def load_baseline_model(task, device):
    """
    Load roberta-base with the correct head for the task, all weights frozen.
    We don't actually freeze them (eval mode is enough), but we never call
    an optimiser, so parameters are never updated.
    """
    task_cfg = TASK_REGISTRY[task]

    if task_cfg.task_type == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=task_cfg.num_labels,
        )
    else:  # span_extraction
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Single-task evaluation
# ---------------------------------------------------------------------------
def eval_one_task(task_name, device):
    print(f"\n{'='*60}")
    print(f"  Task: {task_name.upper()}")
    print(f"{'='*60}")

    task_cfg  = TASK_REGISTRY[task_name]

    # Tokenizer — same setup as run_experiment.py
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # (roberta-base is an encoder; no left-padding or pad_token fixups needed)

    print("  Loading data...")
    dataloaders = get_dataloaders(task_cfg, tokenizer)

    print("  Loading untrained model...")
    model = load_baseline_model(task_name, device)

    print("  Running evaluation...")
    t0 = time.time()

    if task_cfg.task_type == "classification":
        score = evaluate_classification(model, dataloaders["eval"], task_cfg, device)
        elapsed = time.time() - t0
        result = {
            "task":         task_name,
            "metric":       task_cfg.metric,
            "score":        round(score, 4),
            "sota":         task_cfg.sota_baseline,
            "gap_to_sota":  round(score - task_cfg.sota_baseline, 4),
            "elapsed_s":    round(elapsed, 1),
        }
        print(f"  {task_cfg.metric}: {score:.4f}  (SOTA {task_cfg.sota_baseline}, gap {score - task_cfg.sota_baseline:+.2f})")

    else:  # SQuAD 2.0
        raw_val = load_dataset(task_cfg.dataset_name, split="validation")
        raw_lookup = {ex["id"]: ex for ex in raw_val}
        f1, em = evaluate_squad(model, dataloaders["eval"], dataloaders["eval_dataset"], raw_lookup, device)
        elapsed = time.time() - t0
        result = {
            "task":         task_name,
            "metric":       "f1",
            "score":        round(f1, 4),
            "exact_match":  round(em, 4),
            "sota_f1":      task_cfg.sota_baseline,
            "gap_to_sota":  round(f1 - task_cfg.sota_baseline, 4),
            "elapsed_s":    round(elapsed, 1),
        }
        print(f"  F1: {f1:.4f}  EM: {em:.4f}  (SOTA F1 {task_cfg.sota_baseline}, gap {f1 - task_cfg.sota_baseline:+.2f})")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate untrained roberta-base baseline.")
    parser.add_argument("--task",   type=str, default=None,
                        help="Single task to evaluate (cola | sst2 | snli | squad2). "
                             "Omit to run all four.")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cuda | cpu). Auto-detected if omitted.")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Tasks to run
    tasks = [args.task] if args.task else TASK_ORDER

    # Load existing results so we can append without re-running completed tasks
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            all_results = json.load(f)
        print(f"\nLoaded {len(all_results)} existing baseline result(s) from {OUTPUT_FILE}")
    else:
        all_results = {}

    # Run each task
    for task_name in tasks:
        if task_name not in TASK_REGISTRY:
            print(f"[WARN] Unknown task '{task_name}' — skipping.")
            continue

        if task_name in all_results:
            print(f"\n[SKIP] {task_name} already in {OUTPUT_FILE}")
            continue

        try:
            result = eval_one_task(task_name, device)
            all_results[task_name] = result

            # Save after every task (safe against disconnections)
            with open(OUTPUT_FILE, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  Saved → {OUTPUT_FILE}")

        except Exception as e:
            print(f"\n[ERROR] {task_name} failed: {e}")
            import traceback; traceback.print_exc()
            all_results[task_name] = {"task": task_name, "status": "error", "error": str(e)}
            with open(OUTPUT_FILE, "w") as f:
                json.dump(all_results, f, indent=2)

    # Final summary
    print(f"\n{'='*60}")
    print("  BASELINE SUMMARY (untrained roberta-base)")
    print(f"{'='*60}")
    for task_name, r in all_results.items():
        if r.get("status") == "error":
            print(f"  {task_name:8s}  ERROR: {r.get('error', '?')}")
        elif task_name == "squad2":
            print(f"  {task_name:8s}  F1={r['score']:.4f}  EM={r.get('exact_match', '?')}  "
                  f"(SOTA {r['sota_f1']}, gap {r['gap_to_sota']:+.2f})")
        else:
            print(f"  {task_name:8s}  {r['metric']}={r['score']:.4f}  "
                  f"(SOTA {r['sota']}, gap {r['gap_to_sota']:+.2f})")
    print(f"\nFull results: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()