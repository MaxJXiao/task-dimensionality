"""
Training loop for the LoRA rank sweep experiment.

Design constraints
------------------
- Fixed-length training: no early stopping, no validation gating.
  TrainingConfig.num_epochs ensures we run well past convergence so per-rank
  overfitting / plateau dynamics are visible.
- Test-set metric is logged every TrainingConfig.eval_steps optimizer steps.
- Training loss is logged at every step.
- One CSV per (task, rank) at results/{model}/{variant}/{task}/{rank}/training_log.csv
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

from src.config import TaskConfig, TrainingConfig, TRAINING, MODEL_REGISTRY
from src.data_loader import get_dataloaders

# ---------------------------------------------------------------------------
# Experiment hyperparameters
# ---------------------------------------------------------------------------

BATCH_SIZE_CLS: int = 32
BATCH_SIZE_SQUAD: int = 16     # SQuAD sequences are 384 tokens; half the batch to stay within memory
BATCH_SIZE_CAUSAL_LM: int = 4  # 1024-token sequences + 1B QLoRA model

# SQuAD 2.0 post-processing
_N_BEST: int = 20              # top-20 start/end pairs is standard; more gives diminishing returns
_MAX_ANSWER_LEN: int = 30      # spans longer than 30 tokens are almost certainly extraction errors
_NULL_SCORE_DIFF: float = 0.0  # neutral threshold: predict null only if it strictly beats best span
                                # positive values bias toward unanswerable; negative toward answering


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _output_dir(task_name: str, rank_label: str | int, model_name: str = "roberta-base", variant: str = "attn") -> str:
    model_slug = model_name.replace("/", "--")
    path = os.path.join("results", model_slug, variant, task_name, str(rank_label))
    os.makedirs(path, exist_ok=True)
    return path


def _save_log(rows: list[dict], out_dir: str) -> None:
    if not rows:
        return
    path = os.path.join(out_dir, "training_log.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "test_metric", "exact_match", "final_rouge"])
        writer.writeheader()
        writer.writerows(rows)


def _training_cfg_for_task(task: TaskConfig) -> TrainingConfig:
    """Return a TrainingConfig with the correct batch size and epoch count for *task*."""
    if task.task_type == "span_extraction":
        batch_size = BATCH_SIZE_SQUAD
    elif task.task_type == "causal_lm":
        batch_size = BATCH_SIZE_CAUSAL_LM
    else:
        batch_size = BATCH_SIZE_CLS
    cfg = replace(TRAINING, batch_size=batch_size)
    if task.num_epochs is not None:
        cfg = replace(cfg, num_epochs=task.num_epochs)
    if task.eval_steps is not None:
        cfg = replace(cfg, eval_steps=task.eval_steps)
    return cfg


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
    score = float(list(result.values())[0])
    # MCC is in [-1, 1]; scale to percentage points so all metrics share the
    # same 0-100 scale as the SOTA baselines in TaskConfig.
    if task.metric in ("matthews_correlation", "accuracy"):
        score *= 100
    return score


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
            fwd = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            if isinstance(batch.get("token_type_ids"), torch.Tensor):
                fwd["token_type_ids"] = batch["token_type_ids"].to(device)
            outputs = model(**fwd)
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
    return float(result["f1"]), float(result["exact"])


# ---------------------------------------------------------------------------
# Evaluation — causal LM / summarization (BillSum)
# ---------------------------------------------------------------------------

def evaluate_causal_lm(
    model,
    eval_loader,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 512,
) -> float:
    """Generate summaries on the eval set and return ROUGE-L."""
    rouge = hf_evaluate.load("rouge")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model.eval()
    all_preds: list[str] = []
    all_refs: list[str] = []

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            references: list[str] = batch["reference"]

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            new_tokens = generated[:, input_ids.shape[1]:]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            all_preds.extend(decoded)
            all_refs.extend(references)

    model.train()
    # rouge metric returns values in [0, 1]; scale to percentage points for
    # consistency with all other metrics in this codebase (accuracy, MCC, F1).
    result = rouge.compute(predictions=all_preds, references=all_refs)
    return float(result["rougeL"]) * 100


def evaluate_perplexity(
    model,
    ppl_loader,
    device: torch.device,
) -> float:
    """Forward-pass perplexity on the eval set (no generation required)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in ppl_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # outputs.loss is mean NLL over non-masked tokens; recover the sum
            # so we can compute perplexity over the full eval set.
            n_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    model.train()
    return float(torch.exp(torch.tensor(total_loss / total_tokens)).item())


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
    variant: str = "attn",
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
    variant:
        LoRA target scope: "attn" (QKV only) or "attn_mlp" (QKV + dense).
        Used only for output directory naming.

    Returns
    -------
    List of log row dicts (also written to training_log.csv).
    """
    out_dir = _output_dir(task.name, rank_label, model_name, variant)
    training_cfg = _training_cfg_for_task(task)
    model_cfg = MODEL_REGISTRY[model_name]
    if model_cfg.learning_rate is not None:
        training_cfg = replace(training_cfg, learning_rate=model_cfg.learning_rate)

    loaders = get_dataloaders(task, tokenizer, training_cfg)
    train_loader = loaders["train"]
    eval_loader = loaders["eval"]
    eval_ppl_loader = loaders["eval_ppl"]   # training-fmt eval set for perplexity; None for non-causal tasks
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

    total_steps = training_cfg.num_epochs * len(train_loader)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(training_cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    log_rows: list[dict] = []
    global_step = 0

    for epoch in range(training_cfg.num_epochs):
        pbar = tqdm(
            train_loader,
            desc=f"[{task.name} | r={rank_label}] epoch {epoch + 1}/{training_cfg.num_epochs}",
        )
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            train_loss = round(loss.item(), 6)
            pbar.set_postfix(loss=f"{train_loss:.4f}", step=global_step)

            test_metric: float | str = ""
            exact_match: float | str = ""
            if global_step % training_cfg.eval_steps == 0:
                if task.task_type == "classification":
                    test_metric = round(
                        evaluate_classification(model, eval_loader, task, device), 4
                    )
                elif task.task_type == "causal_lm":
                    test_metric = round(
                        evaluate_perplexity(model, eval_ppl_loader, device), 4
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
                "final_rouge": "",
            })

    # If training ended on a non-eval step the last CSV row has no metric;
    # evaluate now so analysis always has a clean final data point.
    if global_step % training_cfg.eval_steps != 0:
        if task.task_type == "classification":
            final_score = evaluate_classification(model, eval_loader, task, device)
            log_rows[-1]["test_metric"] = round(final_score, 4)
        elif task.task_type == "causal_lm":
            final_ppl = evaluate_perplexity(model, eval_ppl_loader, device)
            log_rows[-1]["test_metric"] = round(final_ppl, 4)
        else:
            final_f1, final_em = evaluate_squad(model, eval_loader, eval_dataset, raw_lookup, device)
            log_rows[-1]["test_metric"] = round(final_f1, 4)
            log_rows[-1]["exact_match"] = round(final_em, 4)

    # One-shot ROUGE eval at the end of each BillSum run (generation is slow;
    # we only need it once for final comparison, not every eval step).
    if task.task_type == "causal_lm":
        final_rouge = evaluate_causal_lm(model, eval_loader, tokenizer, device)
        log_rows[-1]["final_rouge"] = round(final_rouge, 4)

    _save_log(log_rows, out_dir)

    ckpt_dir = os.path.join(out_dir, "checkpoint")
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"  [CKPT] Saved → {ckpt_dir}")

    return log_rows
