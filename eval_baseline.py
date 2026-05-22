"""
eval_baseline.py
----------------
Evaluate an untrained (zero-shot) model on all four experiment tasks.

Loads the model with no LoRA adapters and no training, runs the same
evaluation functions used by train.py, and saves results to:
    results/<model-slug>/baseline/baseline_results.json

Usage:
    python eval_baseline.py                                   # all 4 tasks, default model
    python eval_baseline.py --task cola                       # single task
    python eval_baseline.py --model bert-base-uncased         # different model
    python eval_baseline.py --device cpu                      # force CPU
"""

import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering

from src.config import TASK_REGISTRY, DEFAULT_MODEL, MODEL_REGISTRY, RESULTS_DIR
from src.data_loader import get_dataloaders
from src.train import evaluate_classification, evaluate_squad

TASK_ORDER = list(TASK_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Model loader (no LoRA, no training)
# ---------------------------------------------------------------------------
def load_baseline_model(task, device, model_name):
    """Load the model with the correct head for the task. Weights are not updated."""
    task_cfg = TASK_REGISTRY[task]

    if task_cfg.task_type == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=task_cfg.num_labels,
        )
    else:  # span_extraction
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Single-task evaluation
# ---------------------------------------------------------------------------
def eval_one_task(task_name, device, model_name, tokenizer):
    print(f"\n{'='*60}")
    print(f"  Task: {task_name.upper()}")
    print(f"{'='*60}")

    task_cfg = TASK_REGISTRY[task_name]

    print("  Loading data...")
    dataloaders = get_dataloaders(task_cfg, tokenizer)

    print("  Loading untrained model...")
    model = load_baseline_model(task_name, device, model_name)

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
    parser = argparse.ArgumentParser(description="Evaluate an untrained model baseline.")
    parser.add_argument("--task",   type=str, default=None,
                        help="Single task to evaluate (cola | sst2 | snli | squad2). "
                             "Omit to run all four.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=list(MODEL_REGISTRY.keys()),
                        help=f"Model to evaluate. Default: {DEFAULT_MODEL}")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cuda | cpu). Auto-detected if omitted.")
    args = parser.parse_args()

    model_name = args.model
    model_slug = model_name.replace("/", "--")
    output_dir  = os.path.join(RESULTS_DIR, model_slug, "baseline")
    output_file = os.path.join(output_dir, "baseline_results.json")

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nModel : {model_name}")
    print(f"Device: {device}")

    # Tokenizer — shared across all tasks for this model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_cfg = MODEL_REGISTRY[model_name]
    if model_cfg.architecture == "decoder":
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    # Tasks to run
    tasks = [args.task] if args.task else TASK_ORDER

    # Load existing results so we can append without re-running completed tasks
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_file):
        with open(output_file) as f:
            all_results = json.load(f)
        print(f"\nLoaded {len(all_results)} existing baseline result(s) from {output_file}")
    else:
        all_results = {}

    # Run each task
    for task_name in tasks:
        if task_name not in TASK_REGISTRY:
            print(f"[WARN] Unknown task '{task_name}' — skipping.")
            continue

        if task_name in all_results:
            print(f"\n[SKIP] {task_name} already in {output_file}")
            continue

        try:
            result = eval_one_task(task_name, device, model_name, tokenizer)
            all_results[task_name] = result

            # Save after every task (safe against disconnections)
            with open(output_file, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  Saved → {output_file}")

        except Exception as e:
            print(f"\n[ERROR] {task_name} failed: {e}")
            import traceback; traceback.print_exc()
            all_results[task_name] = {"task": task_name, "status": "error", "error": str(e)}
            with open(output_file, "w") as f:
                json.dump(all_results, f, indent=2)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  BASELINE SUMMARY (untrained {model_name})")
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
    print(f"\nFull results: {output_file}")


if __name__ == "__main__":
    main()