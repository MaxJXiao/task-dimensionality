"""
Training loop for the decoder LoRA rank sweep experiment.

Identical in structure to src/train.py but uses decoder-appropriate batch sizes
(8 / 4 / 4 for cls / squad / causal_lm vs. 32 / 16 / 8 for encoders).
All evaluation functions are imported from src.train to avoid duplication.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import replace

import evaluate as hf_evaluate
import torch
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.train import (
    evaluate_causal_lm,
    evaluate_classification,
    evaluate_perplexity,
    evaluate_squad,
)
from src_decoder.config import (
    BATCH_SIZE_CAUSAL_LM,
    BATCH_SIZE_CLS,
    BATCH_SIZE_SQUAD,
    MODEL_REGISTRY,
    TRAINING,
    TaskConfig,
    TrainingConfig,
)
from src_decoder.data_loader import get_dataloaders


def _output_dir(task_name: str, rank_label: str | int, model_name: str = "meta-llama/Llama-3.2-1B", variant: str = "attn") -> str:
    model_slug = model_name.replace("/", "--")
    path = os.path.join("results", model_slug, variant, task_name, str(rank_label))
    os.makedirs(path, exist_ok=True)
    return path


def _save_log(rows: list[dict], out_dir: str) -> None:
    if not rows:
        return
    path = os.path.join(out_dir, "training_log.csv")
    fieldnames = ["step", "train_loss", "test_metric", "exact_match", "final_rouge", "final_pass_at_1", "final_pass_at_1_ood", "final_em_math", "final_f1"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_pass_at_1(
    model,
    eval_loader,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 256,
) -> float:
    """Generate one solution per MBPP problem and return pass@1 (%) via evaluate code_eval."""
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    metric = hf_evaluate.load("code_eval")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model.eval()

    all_predictions: list[list[str]] = []
    all_tests: list[str] = []

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tests_batch: list[str] = batch["tests"]

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            new_tokens = generated[:, input_ids.shape[1]:]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for code, tests in zip(decoded, tests_batch):
                all_predictions.append([code])
                all_tests.append(tests)

    model.train()
    if not all_predictions:
        return 0.0
    pass_at_k, _ = metric.compute(predictions=all_predictions, references=all_tests, k=[1])
    return pass_at_k["pass@1"] * 100.0


def _extract_final_number(text: str) -> str:
    """Return the number after '####' in a GSM8K-style answer; fallback to last number."""
    m = re.search(r"####\s*([\d,\.\-]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"[\d,\.\-]+", text)
    return nums[-1].replace(",", "").strip() if nums else ""


def evaluate_math_em(
    model,
    eval_loader,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 256,
) -> float:
    """Generate solutions and return exact match on extracted final number (%) via evaluate exact_match."""
    metric = hf_evaluate.load("exact_match")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model.eval()

    all_predictions: list[str] = []
    all_references: list[str] = []

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gold_batch: list[str] = batch["gold_answers"]

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            new_tokens = generated[:, input_ids.shape[1]:]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for pred, gold in zip(decoded, gold_batch):
                all_predictions.append(_extract_final_number(pred))
                all_references.append(gold)

    model.train()
    if not all_predictions:
        return 0.0
    result = metric.compute(predictions=all_predictions, references=all_references)
    return result["exact_match"] * 100.0


def evaluate_generative_qa(
    model,
    eval_loader,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 32,
) -> tuple[float, float]:
    """Generate answers and return (EM, F1) as percentages via evaluate squad.

    Returns (exact_match %, f1 %) — squad metric output is already 0-100 scale.
    TriviaQA has multiple valid aliases per question; all are passed as the
    reference answer list so the metric picks the best-matching one.
    """
    metric = hf_evaluate.load("squad")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model.eval()

    all_predictions: list[dict] = []
    all_references: list[dict] = []

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gold_batch: list[str] = batch["gold_answers"]

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            new_tokens = generated[:, input_ids.shape[1]:]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for pred, gold_str in zip(decoded, gold_batch):
                qid = str(len(all_predictions))
                aliases = gold_str.split("\t") if gold_str else [""]
                all_predictions.append({"id": qid, "prediction_text": pred})
                all_references.append({
                    "id": qid,
                    "answers": {"answer_start": [0] * len(aliases), "text": aliases},
                })

    model.train()
    if not all_predictions:
        return 0.0, 0.0
    result = metric.compute(predictions=all_predictions, references=all_references)
    return result["exact_match"], result["f1"]


def _training_cfg_for_task(task: TaskConfig) -> TrainingConfig:
    if task.task_type == "span_extraction":
        batch_size = BATCH_SIZE_SQUAD
    elif task.task_type in ("causal_lm", "code_generation", "generative_qa", "math_reasoning"):
        batch_size = BATCH_SIZE_CAUSAL_LM
    else:
        batch_size = BATCH_SIZE_CLS
    cfg = replace(TRAINING, batch_size=batch_size)
    if task.num_epochs is not None:
        cfg = replace(cfg, num_epochs=task.num_epochs)
    if task.eval_steps is not None:
        cfg = replace(cfg, eval_steps=task.eval_steps)
    return cfg


def train_one_run(
    task: TaskConfig,
    model,
    tokenizer,
    rank_label: str | int,
    device: torch.device,
    model_name: str = "meta-llama/Llama-3.2-1B",
    variant: str = "attn",
) -> list[dict]:
    out_dir = _output_dir(task.name, rank_label, model_name, variant)
    training_cfg = _training_cfg_for_task(task)
    model_cfg = MODEL_REGISTRY[model_name]
    if model_cfg.learning_rate is not None:
        training_cfg = replace(training_cfg, learning_rate=model_cfg.learning_rate)

    loaders = get_dataloaders(task, tokenizer, training_cfg)
    train_loader = loaders["train"]
    eval_loader = loaders["eval"]
    eval_ppl_loader = loaders["eval_ppl"]
    eval_dataset = loaders["eval_dataset"]
    eval_ood_loader = loaders.get("eval_ood")

    raw_lookup: dict[str, dict] | None = None
    if task.task_type == "span_extraction":
        raw_val = load_dataset(task.dataset_name, split="validation")
        raw_lookup = {ex["id"]: ex for ex in raw_val}

    # fp16 models have no hf_device_map; move to device normally.
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
                elif task.task_type in ("causal_lm", "code_generation", "generative_qa", "math_reasoning"):
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
                "final_pass_at_1": "",
                "final_pass_at_1_ood": "",
                "final_em_math": "",
                "final_f1": "",
            })

    if global_step % training_cfg.eval_steps != 0:
        if task.task_type == "classification":
            final_score = evaluate_classification(model, eval_loader, task, device)
            log_rows[-1]["test_metric"] = round(final_score, 4)
        elif task.task_type in ("causal_lm", "code_generation", "generative_qa", "math_reasoning"):
            final_ppl = evaluate_perplexity(model, eval_ppl_loader, device)
            log_rows[-1]["test_metric"] = round(final_ppl, 4)
        else:
            final_f1, final_em = evaluate_squad(model, eval_loader, eval_dataset, raw_lookup, device)
            log_rows[-1]["test_metric"] = round(final_f1, 4)
            log_rows[-1]["exact_match"] = round(final_em, 4)

    if task.task_type == "causal_lm":
        final_rouge = evaluate_causal_lm(model, eval_loader, tokenizer, device)
        log_rows[-1]["final_rouge"] = round(final_rouge, 4)
    elif task.task_type == "code_generation":
        final_p1 = evaluate_pass_at_1(model, eval_loader, tokenizer, device)
        log_rows[-1]["test_metric"] = round(final_p1, 4)
        log_rows[-1]["final_pass_at_1"] = round(final_p1, 4)
        if eval_ood_loader is not None:
            final_p1_ood = evaluate_pass_at_1(model, eval_ood_loader, tokenizer, device)
            log_rows[-1]["final_pass_at_1_ood"] = round(final_p1_ood, 4)
    elif task.task_type == "generative_qa":
        final_em, final_f1 = evaluate_generative_qa(model, eval_loader, tokenizer, device)
        log_rows[-1]["test_metric"] = round(final_f1, 4)
        log_rows[-1]["exact_match"] = round(final_em, 4)
        log_rows[-1]["final_f1"] = round(final_f1, 4)
    elif task.task_type == "math_reasoning":
        final_em = evaluate_math_em(model, eval_loader, tokenizer, device)
        log_rows[-1]["test_metric"] = round(final_em, 4)
        log_rows[-1]["final_em_math"] = round(final_em, 4)

    _save_log(log_rows, out_dir)

    ckpt_dir = os.path.join(out_dir, "checkpoint")
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"  [CKPT] Saved → {ckpt_dir}")

    return log_rows
