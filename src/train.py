"""
Training loop for the LoRA rank sweep experiment.

Design constraints
------------------
- Fixed-length training: no early stopping, no validation gating.
  NUM_EPOCHS (~3× typical RoBERTa GLUE convergence) ensures we run well past
  convergence so per-rank overfitting / plateau dynamics are visible.
- Test-set metric is logged every EVAL_EVERY optimizer steps.
- Training loss is logged at every step.
- One CSV per (task, rank) at results/{task}/{rank}/training_log.csv
  with columns: step, train_loss, test_metric.
  Rows where no evaluation occurred have test_metric = "".
"""

from __future__ import annotations

import collections
import csv
import os
from dataclasses import replace

import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import evaluate as hf_evaluate
from datasets import load_dataset
from tqdm import tqdm

from src.config import TaskConfig, TrainingConfig, TRAINING
from src.data_loader import get_dataloaders

# ---------------------------------------------------------------------------
# Experiment hyperparameters
# ---------------------------------------------------------------------------

LR: float = 2e-5
WEIGHT_DECAY: float = 0.01
NUM_EPOCHS: int = 10       # ~3× the typical 3-epoch GLUE convergence for RoBERTa

EVAL_EVERY: int = 200      # steps between test-set evaluations

BATCH_SIZE_CLS: int = 32
BATCH_SIZE_SQUAD: int = 16     # SQuAD sequences are 384 tokens; half the batch to stay within memory

# SQuAD 2.0 post-processing
_N_BEST: int = 20              # top-20 start/end pairs is standard; more gives diminishing returns
_MAX_ANSWER_LEN: int = 30      # spans longer than 30 tokens are almost certainly extraction errors
_NULL_SCORE_DIFF: float = 0.0  # neutral threshold: predict null only if it strictly beats best span
                                # positive values bias toward unanswerable; negative toward answering


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _output_dir(task_name: str, rank_label: str | int, model_name: str = "roberta-base") -> str:
    model_slug = model_name.replace("/", "--")
    path = os.path.join("results", model_slug, task_name, str(rank_label))
    os.makedirs(path, exist_ok=True)
    return path


def _save_log(rows: list[dict], out_dir: str) -> None:
    if not rows:
        return
    path = os.path.join(out_dir, "training_log.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "test_metric", "exact_match"])
        writer.writeheader()
        writer.writerows(rows)


def _training_cfg_for_task(task: TaskConfig) -> TrainingConfig:
    """Return a TrainingConfig with the correct batch size for *task*."""
    batch_size = BATCH_SIZE_SQUAD if task.task_type == "span_extraction" else BATCH_SIZE_CLS
    return replace(TRAINING, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Evaluation — classification (SST-2, CoLA, SNLI)
# ---------------------------------------------------------------------------

def evaluate_classification(
    model,
    eval_loader,
    task: TaskConfig,
    device: torch.device,
) -> float:
    """Return the primary metric score for a classification task."""
    # Loaded fresh each call — HF metric objects accumulate state internally,
    # so reusing one across eval calls would corrupt the running totals.
    metric = hf_evaluate.load(task.metric)
    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            metric.add_batch(predictions=preds.cpu(), references=labels.cpu())
    model.train()
    result = metric.compute()
    # metric.compute() returns e.g. {"accuracy": 0.93} or {"matthews_correlation": 0.61}
    return float(list(result.values())[0])


# ---------------------------------------------------------------------------
# Evaluation — SQuAD 2.0 (span extraction)
# ---------------------------------------------------------------------------

def _postprocess_squad(
    eval_dataset,
    start_logits_all: np.ndarray,
    end_logits_all: np.ndarray,
    raw_lookup: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """
    Convert per-feature start/end logits into per-example text predictions.

    For each example we:
      1. Collect all candidate spans across all its windowed features.
      2. Compare the best non-null span score against the null score.
      3. Predict the empty string (unanswerable) when null wins by more than
         _NULL_SCORE_DIFF.

    Returns (predictions, references) in the format expected by squad_v2 metric.
    """
    example_to_feats: dict[str, list[int]] = collections.defaultdict(list)
    for i, ex_id in enumerate(eval_dataset["example_id"]):
        example_to_feats[ex_id].append(i)

    predictions: list[dict] = []
    references: list[dict] = []

    for ex_id, feat_idxs in example_to_feats.items():
        context: str = raw_lookup[ex_id]["context"]
        min_null = float("inf")
        candidates: list[dict] = []

        for fi in feat_idxs:
            start_logits = start_logits_all[fi]
            end_logits = end_logits_all[fi]
            offsets: list[tuple[int, int]] = eval_dataset[fi]["offset_mapping"]

            # Null score: both pointers at CLS (position 0)
            null_score = float(start_logits[0] + end_logits[0])
            # Take the minimum null score across all windows — the window most
            # confident the question is unanswerable sets the bar for null prediction.
            min_null = min(min_null, null_score)

            top_starts = np.argsort(start_logits)[-1 : -_N_BEST - 1 : -1]
            top_ends = np.argsort(end_logits)[-1 : -_N_BEST - 1 : -1]

            for s in top_starts:
                for e in top_ends:
                    # Skip padding / question tokens (offset was zeroed in data_loader)
                    if offsets[s] == (0, 0) or offsets[e] == (0, 0):
                        continue
                    if e < s or (e - s + 1) > _MAX_ANSWER_LEN:
                        continue
                    candidates.append({
                        "score": float(start_logits[s] + end_logits[e]),
                        "text": context[offsets[s][0] : offsets[e][1]],
                    })

        if not candidates:
            pred_text = ""
        else:
            best_score = max(c["score"] for c in candidates)
            if min_null - best_score > _NULL_SCORE_DIFF:
                pred_text = ""
            else:
                pred_text = max(candidates, key=lambda c: c["score"])["text"]

        predictions.append({
            "id": ex_id,
            "prediction_text": pred_text,
            "no_answer_probability": 0.0,
        })
        references.append({
            "id": ex_id,
            "answers": raw_lookup[ex_id]["answers"],
        })

    return predictions, references


def evaluate_squad(
    model,
    eval_loader,
    eval_dataset,
    raw_lookup: dict[str, dict],
    device: torch.device,
) -> tuple[float, float]:
    """Return (F1, Exact Match) scores for SQuAD 2.0."""
    model.eval()
    all_start: list[np.ndarray] = []
    all_end: list[np.ndarray] = []

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_start.append(outputs.start_logits.cpu().numpy())
            all_end.append(outputs.end_logits.cpu().numpy())

    start_logits = np.concatenate(all_start)
    end_logits = np.concatenate(all_end)

    predictions, references = _postprocess_squad(
        eval_dataset, start_logits, end_logits, raw_lookup
    )
    metric = hf_evaluate.load("squad_v2")
    result = metric.compute(predictions=predictions, references=references)
    model.train()
    return float(result["f1"]), float(result["exact_match"])


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_one_run(
    task: TaskConfig,
    model,
    tokenizer,
    rank_label: str | int,
    device: torch.device,
    model_name: str = "roberta-base",
) -> list[dict]:
    """
    Train *model* on *task* for NUM_EPOCHS and log results.

    Parameters
    ----------
    rank_label:
        The LoRA rank r, or the string "full" for the full fine-tuning baseline.
        Used only for output directory naming.
    model_name:
        HuggingFace model ID. Used to construct the results path so runs for
        different models don't clobber each other.

    Returns
    -------
    List of log row dicts (also written to training_log.csv).
    """
    out_dir = _output_dir(task.name, rank_label, model_name)
    training_cfg = _training_cfg_for_task(task)

    loaders = get_dataloaders(task, tokenizer, training_cfg)
    train_loader = loaders["train"]
    eval_loader = loaders["eval"]
    eval_dataset = loaders["eval_dataset"]  # Dataset | None

    # For SQuAD, build a lookup from example_id → raw example (needed to
    # reconstruct answer text from char offsets during post-processing).
    raw_lookup: dict[str, dict] | None = None
    if task.task_type == "span_extraction":
        raw_val = load_dataset(task.dataset_name, split="validation")
        raw_lookup = {ex["id"]: ex for ex in raw_val}

    # Quantized models (Llama) are already placed on devices via device_map="auto";
    # calling .to() on them raises an error.
    if not getattr(model, "hf_device_map", None):
        model = model.to(device)
    model.train()

    total_steps = NUM_EPOCHS * len(train_loader)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps,
    )

    log_rows: list[dict] = []
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(
            train_loader,
            desc=f"[{task.name} | r={rank_label}] epoch {epoch + 1}/{NUM_EPOCHS}",
        )
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            train_loss = round(loss.item(), 6)
            pbar.set_postfix(loss=f"{train_loss:.4f}", step=global_step)

            test_metric: float | str = ""
            exact_match: float | str = ""
            if global_step % EVAL_EVERY == 0:
                if task.task_type == "classification":
                    test_metric = round(
                        evaluate_classification(model, eval_loader, task, device), 4
                    )
                else:
                    f1, em = evaluate_squad(model, eval_loader, eval_dataset, raw_lookup, device)
                    test_metric = round(f1, 4)
                    exact_match = round(em, 4)

            log_rows.append({
                "step": global_step,
                "train_loss": train_loss,
                "test_metric": test_metric,
                "exact_match": exact_match,
            })

    # If training ended on a non-eval step the last CSV row has no metric;
    # evaluate now so analysis always has a clean final data point.
    if global_step % EVAL_EVERY != 0:
        if task.task_type == "classification":
            final_score = evaluate_classification(model, eval_loader, task, device)
            log_rows[-1]["test_metric"] = round(final_score, 4)
        else:
            final_f1, final_em = evaluate_squad(model, eval_loader, eval_dataset, raw_lookup, device)
            log_rows[-1]["test_metric"] = round(final_f1, 4)
            log_rows[-1]["exact_match"] = round(final_em, 4)

    _save_log(log_rows, out_dir)

    ckpt_dir = os.path.join(out_dir, "checkpoint")
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"  [CKPT] Saved → {ckpt_dir}")

    return log_rows
