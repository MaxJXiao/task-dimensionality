"""
Model loading for the decoder LoRA experiment — plain fp16, no quantization.

_QUANT_MAP from src/model.py maps Llama to 4-bit BnB configs; this module
loads all decoder models in fp16 instead.  prepare_model_for_kbit_training is
not called because there is no quantization to prepare for.

Entry points (same signatures as src/model.py):
  get_lora_model(rank, model_name, task_type, num_labels, variant) → PeftModel
  get_full_model(model_name, task_type, num_labels)                → base model
  trainable_param_summary(model)                                   → dict
"""

from __future__ import annotations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from src_decoder.config import (
    LORA_ALPHA_MULTIPLIER,
    LORA_DROPOUT,
    MODEL_REGISTRY,
    DEFAULT_MODEL,
)


_CAUSAL_LM_TASK_TYPES = frozenset({"causal_lm", "code_generation", "generative_qa", "math_reasoning"})


def _load_base_model(model_name: str, task_type: str, num_labels: int):
    """Load the model in fp16 with no quantization config."""
    kwargs: dict = {"torch_dtype": torch.float16}

    if task_type == "classification":
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, **kwargs
        )
    elif task_type == "span_extraction":
        return AutoModelForQuestionAnswering.from_pretrained(model_name, **kwargs)
    elif task_type in _CAUSAL_LM_TASK_TYPES:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown task_type: {task_type!r}")


def _peft_task_type(task_type: str) -> TaskType:
    if task_type == "classification":
        return TaskType.SEQ_CLS
    elif task_type == "span_extraction":
        return TaskType.QUESTION_ANS
    elif task_type in _CAUSAL_LM_TASK_TYPES:
        return TaskType.CAUSAL_LM
    raise ValueError(f"Unknown task_type: {task_type!r}")


def get_lora_model(
    rank: int,
    model_name: str = DEFAULT_MODEL,
    task_type: str = "causal_lm",
    num_labels: int = 2,
    variant: str = "attn",
) -> PeftModel:
    """Return a plain LoRA-wrapped decoder model (no kbit preparation)."""
    model_cfg = MODEL_REGISTRY[model_name]
    base = _load_base_model(model_name, task_type, num_labels)

    if variant == "attn":
        target_modules = model_cfg.lora_attn_modules
    elif variant == "attn_mlp":
        target_modules = model_cfg.lora_attn_mlp_modules
    else:
        raise ValueError(f"Unknown LoRA variant: {variant!r}. Choose 'attn' or 'attn_mlp'.")

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=int(rank * LORA_ALPHA_MULTIPLIER),
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=target_modules,
        task_type=_peft_task_type(task_type),
    )
    return get_peft_model(base, lora_cfg)


def get_full_model(
    model_name: str = DEFAULT_MODEL,
    task_type: str = "causal_lm",
    num_labels: int = 2,
):
    """Return an fp16 decoder model with all parameters unfrozen."""
    model = _load_base_model(model_name, task_type, num_labels)
    for param in model.parameters():
        param.requires_grad = True
    return model


def trainable_param_summary(model) -> dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_pct": round(100 * trainable / total, 3),
    }
