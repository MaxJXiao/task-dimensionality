"""
Data loading and tokenisation for the LoRA rank sweep experiment.

Handles all four task types:
  - Single-sentence classification   (SST-2, CoLA)
  - Sentence-pair classification      (SNLI)
  - Extractive QA / span extraction   (SQuAD 2.0)

Public API
----------
get_dataloaders(task, tokenizer, training_cfg)
    -> {"train": DataLoader, "eval": DataLoader, "eval_dataset": Dataset | None}

The eval_dataset entry is only populated for SQuAD 2.0, where offset_mapping
and example_id are needed by the trainer to post-process logits into text spans.
"""

from __future__ import annotations

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from src.config import TRAINING, TaskConfig, TrainingConfig

# 128-token overlap between consecutive windows so answers that straddle a
# boundary appear fully inside at least one window and are not missed.
_SQUAD_STRIDE = 128


# ---------------------------------------------------------------------------
# Raw dataset loading
# ---------------------------------------------------------------------------

def load_raw_dataset(task: TaskConfig, split: str) -> Dataset:
    """Fetch the raw HuggingFace dataset for *task* and *split*."""
    if task.name == "snli":
        ds = load_dataset("snli", split=split)
        # label == -1 means annotators couldn't agree; these rows have no ground truth. Thus, filter out these samples.
        ds = ds.filter(lambda ex: ex["label"] != -1)
    elif task.name == "squad2":
        ds = load_dataset("rajpurkar/squad_v2", split=split)
    else:
        ds = load_dataset(task.dataset_name, task.dataset_config, split=split)
    return ds


# ---------------------------------------------------------------------------
# Tokenisation — classification
# ---------------------------------------------------------------------------

def _tokenize_classification(
    examples: dict,
    tokenizer: PreTrainedTokenizerBase,
    task: TaskConfig,
) -> dict:
    texts = examples[task.text_column]
    if task.second_text_column:
        tokenized = tokenizer(
            texts,
            examples[task.second_text_column],
            max_length=task.max_input_length,
            truncation=True,
            padding="max_length",
        )
    else:
        tokenized = tokenizer(
            texts,
            max_length=task.max_input_length,
            truncation=True,
            padding="max_length",
        )
    tokenized["labels"] = examples[task.label_column]
    return tokenized


# ---------------------------------------------------------------------------
# Tokenisation — SQuAD 2.0 (span extraction)
# ---------------------------------------------------------------------------

def _tokenize_squad_train(
    examples: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict:
    """
    Tokenise SQuAD 2.0 examples for training.

    Long contexts are split into overlapping windows via *stride*.  For each
    window we compute token-level start/end positions for the answer span.
    Windows where the answer falls outside the current window, and all
    unanswerable questions, are labelled with the CLS position (index 0).
    """
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=_SQUAD_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = tokenized.pop("overflow_to_sample_mapping")
    offset_map = tokenized.pop("offset_mapping")

    start_positions: list[int] = []
    end_positions: list[int] = []

    for feat_idx, offsets in enumerate(offset_map):
        sample_idx = sample_map[feat_idx]
        answers = examples["answers"][sample_idx]
        # CLS (position 0) is the SQuAD 2.0 convention for "no answer".
        cls_idx: int = tokenized["input_ids"][feat_idx].index(tokenizer.cls_token_id)

        if not answers["answer_start"]:
            # Unanswerable: supervise model to point at CLS.
            start_positions.append(cls_idx)
            end_positions.append(cls_idx)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        seq_ids = tokenized.sequence_ids(feat_idx)

        # Locate the token range that covers the context (sequence id == 1).
        ctx_start = next(i for i, s in enumerate(seq_ids) if s == 1)
        ctx_end = len(seq_ids) - 1
        while seq_ids[ctx_end] != 1:  # walk back past padding and EOS
            ctx_end -= 1

        # Answer outside this window → treat as unanswerable.
        if offsets[ctx_start][0] > start_char or offsets[ctx_end][1] < end_char:
            start_positions.append(cls_idx)
            end_positions.append(cls_idx)
            continue

        # Walk forward to find the first token whose start >= answer start char.
        # The loop overshoots by one, so we step back.
        tok_start = ctx_start
        while tok_start <= ctx_end and offsets[tok_start][0] <= start_char:
            tok_start += 1
        start_positions.append(tok_start - 1)

        # Walk backward to find the last token whose end <= answer end char.
        # Same overshoot-by-one pattern; step forward to correct.
        tok_end = ctx_end
        while tok_end >= ctx_start and offsets[tok_end][1] >= end_char:
            tok_end -= 1
        end_positions.append(tok_end + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def _tokenize_squad_eval(
    examples: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict:
    """
    Tokenise SQuAD 2.0 examples for evaluation.

    Returns offset_mapping (zeroed for non-context tokens) and example_id
    alongside the standard model inputs.  The trainer uses these to map
    predicted token spans back to character spans for squad_v2 metric scoring.
    """
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=_SQUAD_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = tokenized["overflow_to_sample_mapping"]

    # Map each feature back to its source example id for post-processing.
    tokenized["example_id"] = [
        examples["id"][sample_map[i]] for i in range(len(sample_map))
    ]

    # Zero offsets for question tokens and padding so only context offsets
    # remain valid during span extraction.
    clean_offsets: list[list[tuple[int, int]]] = []
    for i, offsets in enumerate(tokenized["offset_mapping"]):
        seq_ids = tokenized.sequence_ids(i)
        clean_offsets.append(
            [(s, e) if seq_ids[k] == 1 else (0, 0) for k, (s, e) in enumerate(offsets)]
        )
    tokenized["offset_mapping"] = clean_offsets
    tokenized.pop("overflow_to_sample_mapping")
    return tokenized


# ---------------------------------------------------------------------------
# Custom collate for SQuAD eval
# ---------------------------------------------------------------------------

def _squad_eval_collate(batch: list[dict]) -> dict:
    """
    Custom collate for SQuAD eval batches.

    PyTorch's default collate crashes on offset_mapping (list of (int, int) tuples)
    and example_id (strings). Those fields are only needed by post-processing, not
    the model, so we keep them as Python lists and stack only the tensor fields.
    """
    result: dict = {}
    for key in batch[0]:
        values = [ex[key] for ex in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataset(
    task: TaskConfig,
    tokenizer: PreTrainedTokenizerBase,
    split: str,
) -> Dataset:
    """Return a tokenised HuggingFace Dataset ready for wrapping in a DataLoader."""
    raw = load_raw_dataset(task, split)

    if task.task_type == "classification":
        ds = raw.map(
            _tokenize_classification,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer, "task": task},
            remove_columns=raw.column_names,
        )
        fmt_cols = ["input_ids", "attention_mask", "labels"]
        if "token_type_ids" in ds.column_names:
            fmt_cols = ["input_ids", "attention_mask", "token_type_ids", "labels"]
        ds.set_format("torch", columns=fmt_cols)

    elif task.task_type == "span_extraction":
        is_train = split == "train"
        fn = _tokenize_squad_train if is_train else _tokenize_squad_eval
        ds = raw.map(
            fn,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer, "max_length": task.max_input_length},
            remove_columns=raw.column_names,
        )
        has_tti = "token_type_ids" in ds.column_names
        if is_train:
            fmt_cols = ["input_ids", "attention_mask", "start_positions", "end_positions"]
            if has_tti:
                fmt_cols = ["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"]
            ds.set_format("torch", columns=fmt_cols)
        else:
            # output_all_columns=True keeps offset_mapping and example_id accessible
            # alongside the tensor columns; without it set_format silently drops them.
            fmt_cols = ["input_ids", "attention_mask"]
            if has_tti:
                fmt_cols.append("token_type_ids")
            ds.set_format("torch", columns=fmt_cols, output_all_columns=True)

    else:
        raise ValueError(f"Unknown task_type: {task.task_type!r}")

    return ds


def get_dataloaders(
    task: TaskConfig,
    tokenizer: PreTrainedTokenizerBase,
    training_cfg: TrainingConfig = TRAINING,
) -> dict:
    """
    Build and return DataLoaders for train and eval splits.

    Returns
    -------
    dict with keys:
        "train"        : DataLoader — shuffled, batch_size from training_cfg
        "eval"         : DataLoader — unshuffled, batch_size * 2
        "eval_dataset" : Dataset | None — full eval Dataset for SQuAD 2.0
                         post-processing (contains offset_mapping, example_id);
                         None for classification tasks.
    """
    train_ds = build_dataset(task, tokenizer, split="train")
    eval_ds = build_dataset(task, tokenizer, split="validation")

    train_loader = DataLoader(
        train_ds,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,   # async CPU→GPU transfer; free speedup on CUDA
    )

    is_squad_eval = task.task_type == "span_extraction"
    eval_loader = DataLoader(
        eval_ds,
        batch_size=training_cfg.batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=_squad_eval_collate if is_squad_eval else None,
    )

    return {
        "train": train_loader,
        "eval": eval_loader,
        "eval_dataset": eval_ds if is_squad_eval else None,
    }
