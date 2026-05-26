"""
Data loading for the decoder LoRA rank sweep — adds code_generation support
(MBPP) on top of src.data_loader for the other task types.
"""

from __future__ import annotations

import re
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from src_decoder.config import OodEvalConfig, TaskConfig, TrainingConfig, TRAINING


_CODE_GEN_PROMPT_PREFIX = "### Problem:\n"
_CODE_GEN_PROMPT_SUFFIX = "\n\n### Solution:\n"
# Tokens reserved for the generated solution during training; the rest goes to the problem.
_CODE_BUDGET: int = 256

_QA_PROMPT_PREFIX = "Question: "
_QA_PROMPT_SUFFIX = "\n\nAnswer:\n"
_QA_ANSWER_BUDGET: int = 64  # factual answers are short; leave the rest for the question

_MATH_PROMPT_PREFIX = "Question: "
_MATH_PROMPT_SUFFIX = "\n\nAnswer:\n"
_MATH_ANSWER_BUDGET: int = 256  # GSM8K solutions include chain-of-thought

_CAUSAL_LM_ANSWER_BUDGET: int = 256  # tokens reserved for label / generated output


def _load_decoder_dataset(task: TaskConfig, split: str) -> Dataset:
    if task.dataset_config is not None:
        return load_dataset(task.dataset_name, task.dataset_config, split=split)
    return load_dataset(task.dataset_name, split=split)


def _tokenize_generative_qa_train(examples, tokenizer, task, max_length):
    """Format Question/Answer pairs for supervised finetuning; mask the question from loss."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    prefix_ids = tokenizer(
        _QA_PROMPT_PREFIX, add_special_tokens=True, return_attention_mask=False,
    )["input_ids"]
    suffix_ids = tokenizer(
        _QA_PROMPT_SUFFIX, add_special_tokens=False, return_attention_mask=False,
    )["input_ids"]
    question_budget = max_length - _QA_ANSWER_BUDGET - len(prefix_ids) - len(suffix_ids) - 1

    # TriviaQA: label_column is "answer", a struct column; "value" holds the canonical answer
    # answers = examples[task.label_column]["value"]
    # To this:
    answers = [ans["value"] for ans in examples[task.label_column]]

    question_enc = tokenizer(
        examples[task.text_column],
        max_length=question_budget, truncation=True, padding=False,
        add_special_tokens=False, return_attention_mask=False,
    )
    answer_enc = tokenizer(
        answers,
        max_length=_QA_ANSWER_BUDGET - 1, truncation=True, padding=False,
        add_special_tokens=False, return_attention_mask=False,
    )

    all_input_ids, all_attn, all_labels = [], [], []
    for q_ids, a_ids in zip(question_enc["input_ids"], answer_enc["input_ids"]):
        prompt_ids = prefix_ids + q_ids + suffix_ids
        if eos_id is not None:
            a_ids = a_ids + [eos_id]
        full_ids = prompt_ids + a_ids
        plen = len(prompt_ids)
        content_len = len(full_ids)
        pad_len = max_length - content_len

        all_input_ids.append(full_ids + [pad_id] * pad_len)
        all_attn.append([1] * content_len + [0] * pad_len)
        all_labels.append([-100] * plen + a_ids + [-100] * pad_len)

    return {"input_ids": all_input_ids, "attention_mask": all_attn, "labels": all_labels}


def _tokenize_generative_qa_eval(examples, tokenizer, task, max_length):
    """Prompt-only tokenisation (left-padded); stores normalized aliases for EM/F1 eval."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    prefix_ids = tokenizer(
        _QA_PROMPT_PREFIX, add_special_tokens=True, return_attention_mask=False,
    )["input_ids"]
    suffix_ids = tokenizer(
        _QA_PROMPT_SUFFIX, add_special_tokens=False, return_attention_mask=False,
    )["input_ids"]
    question_budget = max_length - len(prefix_ids) - len(suffix_ids)

    question_enc = tokenizer(
        examples[task.text_column],
        max_length=question_budget, truncation=True, padding=False,
        add_special_tokens=False, return_attention_mask=False,
    )

    # normalized_aliases covers spelling variants; use them for EM/F1 matching
    # examples[label_column] is a list of dicts in batched map — must iterate
    all_aliases = [ans["normalized_aliases"] for ans in examples[task.label_column]]

    all_input_ids, all_attn, all_gold = [], [], []
    for q_ids, aliases in zip(question_enc["input_ids"], all_aliases):
        p_ids_full = prefix_ids + q_ids + suffix_ids
        pad_len = max_length - len(p_ids_full)
        all_input_ids.append([pad_id] * pad_len + p_ids_full)
        all_attn.append([0] * pad_len + [1] * len(p_ids_full))
        all_gold.append("\t".join(aliases) if aliases else "")

    return {"input_ids": all_input_ids, "attention_mask": all_attn, "gold_answers": all_gold}


def _tokenize_code_gen_train(examples, tokenizer, task, max_length):
    """Prompt + solution as a single sequence; prompt tokens are masked from loss."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    prefix_ids = tokenizer(
        _CODE_GEN_PROMPT_PREFIX, add_special_tokens=True, return_attention_mask=False,
    )["input_ids"]
    suffix_ids = tokenizer(
        _CODE_GEN_PROMPT_SUFFIX, add_special_tokens=False, return_attention_mask=False,
    )["input_ids"]
    problem_budget = max_length - _CODE_BUDGET - len(prefix_ids) - len(suffix_ids) - 1  # -1 for EOS

    problem_enc = tokenizer(
        examples[task.text_column],
        max_length=problem_budget, truncation=True, padding=False,
        add_special_tokens=False, return_attention_mask=False,
    )
    code_enc = tokenizer(
        examples[task.label_column],
        max_length=_CODE_BUDGET - 1, truncation=True, padding=False,
        add_special_tokens=False, return_attention_mask=False,
    )

    all_input_ids, all_attn, all_labels = [], [], []
    for p_ids, c_ids in zip(problem_enc["input_ids"], code_enc["input_ids"]):
        prompt_ids = prefix_ids + p_ids + suffix_ids
        if eos_id is not None:
            c_ids = c_ids + [eos_id]
        full_ids = prompt_ids + c_ids
        plen = len(prompt_ids)
        content_len = len(full_ids)
        pad_len = max_length - content_len

        all_input_ids.append(full_ids + [pad_id] * pad_len)
        all_attn.append([1] * content_len + [0] * pad_len)
        all_labels.append([-100] * plen + c_ids + [-100] * pad_len)

    return {"input_ids": all_input_ids, "attention_mask": all_attn, "labels": all_labels}


def _tokenize_code_gen_eval(examples, tokenizer, task, max_length):
    """Prompt-only tokenisation (left-padded for generation); stores test strings for execution."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    prefix_ids = tokenizer(
        _CODE_GEN_PROMPT_PREFIX, add_special_tokens=True, return_attention_mask=False,
    )["input_ids"]
    suffix_ids = tokenizer(
        _CODE_GEN_PROMPT_SUFFIX, add_special_tokens=False, return_attention_mask=False,
    )["input_ids"]
    problem_budget = max_length - len(prefix_ids) - len(suffix_ids)

    problem_enc = tokenizer(
        examples[task.text_column],
        max_length=problem_budget, truncation=True, padding=False,
        add_special_tokens=False, return_attention_mask=False,
    )

    # test_setup_code runs before the tests (usually empty, but present in MBPP schema)
    setup_codes = examples.get("test_setup_code") or [""] * len(examples[task.text_column])

    all_input_ids, all_attn, all_tests = [], [], []
    for p_ids, tests, setup in zip(problem_enc["input_ids"], examples["test_list"], setup_codes):
        p_ids_full = prefix_ids + p_ids + suffix_ids
        pad_len = max_length - len(p_ids_full)
        all_input_ids.append([pad_id] * pad_len + p_ids_full)
        all_attn.append([0] * pad_len + [1] * len(p_ids_full))
        test_str = (setup + "\n" if setup else "") + "\n".join(tests)
        all_tests.append(test_str)

    return {"input_ids": all_input_ids, "attention_mask": all_attn, "tests": all_tests}


def _tokenize_math_train(examples, tokenizer, task, max_length):
    """Format Question/full-solution pairs; mask the question from loss."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    prefix_ids = tokenizer(
        _MATH_PROMPT_PREFIX, add_special_tokens=True, return_attention_mask=False,
    )["input_ids"]
    suffix_ids = tokenizer(
        _MATH_PROMPT_SUFFIX, add_special_tokens=False, return_attention_mask=False,
    )["input_ids"]
    question_budget = max_length - _MATH_ANSWER_BUDGET - len(prefix_ids) - len(suffix_ids) - 1

    question_enc = tokenizer(
        examples[task.text_column],
        max_length=question_budget, truncation=True, padding=False,
        add_special_tokens=False, return_attention_mask=False,
    )
    answer_enc = tokenizer(
        examples[task.label_column],
        max_length=_MATH_ANSWER_BUDGET - 1, truncation=True, padding=False,
        add_special_tokens=False, return_attention_mask=False,
    )

    all_input_ids, all_attn, all_labels = [], [], []
    for q_ids, a_ids in zip(question_enc["input_ids"], answer_enc["input_ids"]):
        prompt_ids = prefix_ids + q_ids + suffix_ids
        if eos_id is not None:
            a_ids = a_ids + [eos_id]
        full_ids = prompt_ids + a_ids
        plen = len(prompt_ids)
        content_len = len(full_ids)
        pad_len = max_length - content_len

        all_input_ids.append(full_ids + [pad_id] * pad_len)
        all_attn.append([1] * content_len + [0] * pad_len)
        all_labels.append([-100] * plen + a_ids + [-100] * pad_len)

    return {"input_ids": all_input_ids, "attention_mask": all_attn, "labels": all_labels}


def _tokenize_math_eval(examples, tokenizer, task, max_length):
    """Prompt-only tokenisation (left-padded); extracts gold final number for EM scoring."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    prefix_ids = tokenizer(
        _MATH_PROMPT_PREFIX, add_special_tokens=True, return_attention_mask=False,
    )["input_ids"]
    suffix_ids = tokenizer(
        _MATH_PROMPT_SUFFIX, add_special_tokens=False, return_attention_mask=False,
    )["input_ids"]
    question_budget = max_length - len(prefix_ids) - len(suffix_ids)

    question_enc = tokenizer(
        examples[task.text_column],
        max_length=question_budget, truncation=True, padding=False,
        add_special_tokens=False, return_attention_mask=False,
    )

    all_input_ids, all_attn, all_gold = [], [], []
    for q_ids, answer in zip(question_enc["input_ids"], examples[task.label_column]):
        p_ids_full = prefix_ids + q_ids + suffix_ids
        pad_len = max_length - len(p_ids_full)
        all_input_ids.append([pad_id] * pad_len + p_ids_full)
        all_attn.append([0] * pad_len + [1] * len(p_ids_full))
        m = re.search(r"####\s*([\d,\.\-]+)", answer)
        gold_num = m.group(1).replace(",", "").strip() if m else answer.strip()
        all_gold.append(gold_num)

    return {"input_ids": all_input_ids, "attention_mask": all_attn, "gold_answers": all_gold}


def _tokenize_causal_lm_train(examples, tokenizer, task, max_length):
    """Generic causal-LM training tokeniser: prompt_prefix + text + prompt_suffix + label.

    Prompt tokens are masked from the loss; only label tokens are supervised.
    Uses task.prompt_prefix / prompt_suffix (both default to "" in TaskConfig).
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    prefix_ids = tokenizer(task.prompt_prefix, add_special_tokens=True, return_attention_mask=False)["input_ids"]
    suffix_ids = tokenizer(task.prompt_suffix, add_special_tokens=False, return_attention_mask=False)["input_ids"] if task.prompt_suffix else []
    text_budget = max_length - _CAUSAL_LM_ANSWER_BUDGET - len(prefix_ids) - len(suffix_ids) - 1

    text_enc = tokenizer(
        examples[task.text_column],
        max_length=text_budget, truncation=True, padding=False,
        add_special_tokens=False, return_attention_mask=False,
    )
    label_enc = tokenizer(
        examples[task.label_column],
        max_length=_CAUSAL_LM_ANSWER_BUDGET - 1, truncation=True, padding=False,
        add_special_tokens=False, return_attention_mask=False,
    )

    all_input_ids, all_attn, all_labels = [], [], []
    for t_ids, l_ids in zip(text_enc["input_ids"], label_enc["input_ids"]):
        prompt_ids = prefix_ids + t_ids + suffix_ids
        if eos_id is not None:
            l_ids = l_ids + [eos_id]
        full_ids = prompt_ids + l_ids
        plen = len(prompt_ids)
        content_len = len(full_ids)
        pad_len = max_length - content_len

        all_input_ids.append(full_ids + [pad_id] * pad_len)
        all_attn.append([1] * content_len + [0] * pad_len)
        all_labels.append([-100] * plen + l_ids + [-100] * pad_len)

    return {"input_ids": all_input_ids, "attention_mask": all_attn, "labels": all_labels}


def _tokenize_causal_lm_eval(examples, tokenizer, task, max_length):
    """Generic causal-LM eval tokeniser: left-padded prompt; stores label as reference string."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    prefix_ids = tokenizer(task.prompt_prefix, add_special_tokens=True, return_attention_mask=False)["input_ids"]
    suffix_ids = tokenizer(task.prompt_suffix, add_special_tokens=False, return_attention_mask=False)["input_ids"] if task.prompt_suffix else []
    text_budget = max_length - len(prefix_ids) - len(suffix_ids)

    text_enc = tokenizer(
        examples[task.text_column],
        max_length=text_budget, truncation=True, padding=False,
        add_special_tokens=False, return_attention_mask=False,
    )

    all_input_ids, all_attn = [], []
    for t_ids in text_enc["input_ids"]:
        p_ids_full = prefix_ids + t_ids + suffix_ids
        pad_len = max_length - len(p_ids_full)
        all_input_ids.append([pad_id] * pad_len + p_ids_full)
        all_attn.append([0] * pad_len + [1] * len(p_ids_full))

    return {"input_ids": all_input_ids, "attention_mask": all_attn, "reference": examples[task.label_column]}


def _tokenize_humaneval_eval(examples, tokenizer, ood_cfg, max_length):
    """Prompt-only tokenisation for HumanEval OOD eval; stores combined test strings."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    prompt_enc = tokenizer(
        examples[ood_cfg.text_column],
        max_length=max_length, truncation=True, padding=False,
        add_special_tokens=True, return_attention_mask=False,
    )

    all_input_ids, all_attn, all_tests = [], [], []
    for p_ids, test_code, entry_point in zip(
        prompt_enc["input_ids"], examples["test"], examples["entry_point"]
    ):
        pad_len = max_length - len(p_ids)
        all_input_ids.append([pad_id] * pad_len + p_ids)
        all_attn.append([0] * pad_len + [1] * len(p_ids))
        all_tests.append(test_code + f"\ncheck({entry_point})")

    # code_eval runs `prediction + "\n" + reference`; for HumanEval the model only
    # generates the function body (the prompt already contains the def/docstring), so
    # we store the original prompt text here and prepend it in evaluate_pass_at_1.
    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attn,
        "tests": all_tests,
        "prompts": list(examples[ood_cfg.text_column]),
    }


def build_ood_eval_dataset(ood_cfg: OodEvalConfig, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    """Build an OOD evaluation dataset (currently only HumanEval format)."""
    if ood_cfg.dataset_config is not None:
        raw = load_dataset(ood_cfg.dataset_name, ood_cfg.dataset_config, split="test")
    else:
        raw = load_dataset(ood_cfg.dataset_name, split="test")

    if ood_cfg.max_eval_samples is not None:
        raw = raw.select(range(min(ood_cfg.max_eval_samples, len(raw))))

    if ood_cfg.test_format == "humaneval":
        fn = _tokenize_humaneval_eval
    else:
        raise ValueError(f"Unknown OOD test_format: {ood_cfg.test_format!r}")

    ds = raw.map(
        fn, batched=True,
        fn_kwargs={"tokenizer": tokenizer, "ood_cfg": ood_cfg, "max_length": ood_cfg.max_input_length},
        remove_columns=raw.column_names,
    )
    ds.set_format("torch", columns=["input_ids", "attention_mask"], output_all_columns=True)
    return ds


def _collate_with_strings(batch: list[dict]) -> dict:
    """Stack tensors; leave string/list fields as Python lists."""
    result: dict = {}
    for key in batch[0]:
        values = [ex[key] for ex in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result


_DECODER_TASK_TYPES = ("causal_lm", "code_generation", "generative_qa", "math_reasoning")


def build_dataset(
    task: TaskConfig,
    tokenizer: PreTrainedTokenizerBase,
    split: str,
    force_train_fmt: bool = False,
) -> Dataset:
    if task.task_type not in _DECODER_TASK_TYPES:
        from src.data_loader import build_dataset as _build
        return _build(task, tokenizer, split, force_train_fmt=force_train_fmt)

    raw = _load_decoder_dataset(task, split)
    if split == "train" and task.max_train_samples is not None:
        raw = raw.select(range(min(task.max_train_samples, len(raw))))
    elif split != "train" and task.max_eval_samples is not None:
        raw = raw.select(range(min(task.max_eval_samples, len(raw))))

    is_train = split == "train" or force_train_fmt
    if task.task_type == "code_generation":
        fn = _tokenize_code_gen_train if is_train else _tokenize_code_gen_eval
    elif task.task_type == "generative_qa":
        fn = _tokenize_generative_qa_train if is_train else _tokenize_generative_qa_eval
    elif task.task_type == "math_reasoning":
        fn = _tokenize_math_train if is_train else _tokenize_math_eval
    else:  # causal_lm
        fn = _tokenize_causal_lm_train if is_train else _tokenize_causal_lm_eval

    ds = raw.map(
        fn, batched=True,
        fn_kwargs={"tokenizer": tokenizer, "task": task, "max_length": task.max_input_length},
        remove_columns=raw.column_names,
    )
    if is_train:
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    else:
        ds.set_format("torch", columns=["input_ids", "attention_mask"], output_all_columns=True)
    return ds


def get_dataloaders(
    task: TaskConfig,
    tokenizer: PreTrainedTokenizerBase,
    training_cfg: TrainingConfig = TRAINING,
    eval_only: bool = False,
) -> dict:
    if task.task_type not in _DECODER_TASK_TYPES:
        from src.data_loader import get_dataloaders as _get
        return _get(task, tokenizer, training_cfg)

    eval_split = getattr(task, "eval_split", "validation")
    eval_ds = build_dataset(task, tokenizer, split=eval_split)
    ppl_ds = build_dataset(task, tokenizer, split=eval_split, force_train_fmt=True)

    eval_loader = DataLoader(
        eval_ds, batch_size=training_cfg.batch_size * 2,
        shuffle=False, num_workers=0, pin_memory=True,
        collate_fn=_collate_with_strings,
    )
    eval_ppl_loader = DataLoader(
        ppl_ds, batch_size=training_cfg.batch_size * 2,
        shuffle=False, num_workers=0, pin_memory=True,
    )
    result = {
        "train": None,
        "eval": eval_loader,
        "eval_ppl": eval_ppl_loader,
        "eval_dataset": None,
        "eval_ood": None,
    }

    if not eval_only:
        train_ds = build_dataset(task, tokenizer, split="train")
        result["train"] = DataLoader(
            train_ds, batch_size=training_cfg.batch_size,
            shuffle=True, num_workers=0, pin_memory=True,
        )

    if task.ood_eval is not None and not eval_only:
        ood_ds = build_ood_eval_dataset(task.ood_eval, tokenizer)
        result["eval_ood"] = DataLoader(
            ood_ds, batch_size=training_cfg.batch_size * 2,
            shuffle=False, num_workers=0, pin_memory=True,
            collate_fn=_collate_with_strings,
        )
    return result
