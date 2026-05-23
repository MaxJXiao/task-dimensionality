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
    elif task.name == "billsum":
        # BillSum has no validation split; use test as the held-out eval set.
        actual_split = "test" if split == "validation" else split
        ds = load_dataset("billsum", split=actual_split)
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
# Tokenisation — causal LM (BillSum)
# ---------------------------------------------------------------------------

_BILLSUM_PROMPT_PREFIX = "Summarize the following US congressional bill:\n\n"
_BILLSUM_PROMPT_SUFFIX = "\n\nSummary:"

# How many tokens to reserve for the summary in training sequences.
# The bill text gets the remaining budget after template overhead and summary.
# Without this split, long bills fill the entire 1024-token budget and leave
# no supervised summary tokens, making those examples no-ops in the loss.
_SUMMARY_BUDGET: int = 256


def _tokenize_causal_lm_train(
    examples: dict,
    tokenizer: PreTrainedTokenizerBase,
    task: TaskConfig,
    max_length: int,
) -> dict:
    """
    Tokenise BillSum examples for causal LM training.

    The prompt is built at the token level (prefix + truncated bill text + suffix)
    so that "\n\nSummary:" is always present even when the bill is long enough to
    trigger truncation. Naively formatting the full prompt string and truncating
    from the right silently drops the "Summary:" cue for the majority of BillSum
    examples, causing the model to predict bill continuations instead of summaries.

    Sequences are right-padded to max_length regardless of tokenizer.padding_side.
    Labels are -100 for prompt tokens and padding; only summary tokens are supervised.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    # Tokenize the fixed template pieces once per batch.
    # prefix includes BOS; suffix and bill text carry no special tokens.
    prefix_ids: list[int] = tokenizer(
        _BILLSUM_PROMPT_PREFIX, add_special_tokens=True, return_attention_mask=False,
    )["input_ids"]
    suffix_ids: list[int] = tokenizer(
        _BILLSUM_PROMPT_SUFFIX, add_special_tokens=False, return_attention_mask=False,
    )["input_ids"]
    text_budget = max_length - _SUMMARY_BUDGET - len(prefix_ids) - len(suffix_ids)

    text_enc = tokenizer(
        examples[task.text_column],
        max_length=text_budget, truncation=True,
        padding=False, add_special_tokens=False, return_attention_mask=False,
    )
    # Reserve 1 slot for EOS so the model learns when to stop generating.
    summary_enc = tokenizer(
        examples[task.label_column],
        max_length=_SUMMARY_BUDGET - 1, truncation=True,
        padding=False, add_special_tokens=False, return_attention_mask=False,
    )

    all_input_ids, all_attn, all_labels = [], [], []
    for t_ids, s_ids in zip(text_enc["input_ids"], summary_enc["input_ids"]):
        p_ids = prefix_ids + t_ids + suffix_ids
        if eos_id is not None:
            s_ids = s_ids + [eos_id]
        full_ids = p_ids + s_ids
        plen = len(p_ids)
        content_len = len(full_ids)
        pad_len = max_length - content_len

        input_ids = full_ids + [pad_id] * pad_len
        attn      = [1] * content_len + [0] * pad_len
        labels    = [-100] * plen + s_ids + [-100] * pad_len

        all_input_ids.append(input_ids)
        all_attn.append(attn)
        all_labels.append(labels)

    return {"input_ids": all_input_ids, "attention_mask": all_attn, "labels": all_labels}


def _tokenize_causal_lm_eval(
    examples: dict,
    tokenizer: PreTrainedTokenizerBase,
    task: TaskConfig,
    max_length: int,
) -> dict:
    """
    Tokenise BillSum examples for causal LM evaluation.

    Same prefix/suffix token-level split as train so "Summary:" is always present.
    Sequences are manually left-padded so generation appends cleanly after the prompt.
    The reference summary is kept as a string for ROUGE-L computation.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    prefix_ids: list[int] = tokenizer(
        _BILLSUM_PROMPT_PREFIX, add_special_tokens=True, return_attention_mask=False,
    )["input_ids"]
    suffix_ids: list[int] = tokenizer(
        _BILLSUM_PROMPT_SUFFIX, add_special_tokens=False, return_attention_mask=False,
    )["input_ids"]
    text_budget = max_length - len(prefix_ids) - len(suffix_ids)

    text_enc = tokenizer(
        examples[task.text_column],
        max_length=text_budget, truncation=True,
        padding=False, add_special_tokens=False, return_attention_mask=False,
    )

    all_input_ids, all_attn = [], []
    for t_ids in text_enc["input_ids"]:
        p_ids = prefix_ids + t_ids + suffix_ids
        pad_len = max_length - len(p_ids)
        input_ids = [pad_id] * pad_len + p_ids
        attn      = [0] * pad_len + [1] * len(p_ids)
        all_input_ids.append(input_ids)
        all_attn.append(attn)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attn,
        "reference": examples[task.label_column],
    }


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

    if split == "train" and task.max_train_samples is not None:
        raw = raw.select(range(min(task.max_train_samples, len(raw))))
    elif split == "validation" and task.max_eval_samples is not None:
        raw = raw.select(range(min(task.max_eval_samples, len(raw))))

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

    elif task.task_type == "causal_lm":
        is_train = split == "train"
        fn = _tokenize_causal_lm_train if is_train else _tokenize_causal_lm_eval
        ds = raw.map(
            fn,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer, "task": task, "max_length": task.max_input_length},
            remove_columns=raw.column_names,
        )
        if is_train:
            ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        else:
            ds.set_format("torch", columns=["input_ids", "attention_mask"], output_all_columns=True)

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

    needs_nonstandard_collate = task.task_type in ("span_extraction", "causal_lm")
    eval_loader = DataLoader(
        eval_ds,
        batch_size=training_cfg.batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=_squad_eval_collate if needs_nonstandard_collate else None,
    )

    return {
        "train": train_loader,
        "eval": eval_loader,
        "eval_dataset": eval_ds if task.task_type == "span_extraction" else None,
    }
