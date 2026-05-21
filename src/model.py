"""
Model loading and LoRA / QLoRA configuration for the rank sweep experiment.

Supported models
----------------
  roberta-base        125 M   encoder   Standard LoRA   Q/K/V/dense
  roberta-large       355 M   encoder   Standard LoRA   Q/K/V/dense
  Llama-3.2-1B          1 B   decoder   QLoRA  8-bit    q_proj/v_proj
  Llama-3.2-3B          3 B   decoder   QLoRA  4-bit    q_proj/v_proj

Entry points
------------
  get_lora_model(rank, model_name, task_type, num_labels) → PeftModel
  get_full_model(model_name, task_type, num_labels)       → base model
  trainable_param_summary(model)                          → dict
"""

from __future__ import annotations

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training

from src.config import MODEL_REGISTRY, DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Quantization configs (bitsandbytes)
# ---------------------------------------------------------------------------

_BNBCONFIG_8BIT = BitsAndBytesConfig(load_in_8bit=True)

_BNBCONFIG_4BIT = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # matmuls run in fp16 even though weights are stored in 4-bit
    bnb_4bit_quant_type="nf4",             # NormalFloat4: optimal for normally-distributed LLM weights
    bnb_4bit_use_double_quant=True,        # quantise the quantisation constants too (~0.4 bits/param saved)
)

# Maps each supported model to its quantization config (None = no quantization).
_QUANT_MAP: dict[str, BitsAndBytesConfig | None] = {
    "roberta-base":             None,
    "roberta-large":            None,
    "meta-llama/Llama-3.2-1B": _BNBCONFIG_8BIT,
    "meta-llama/Llama-3.2-3B": _BNBCONFIG_4BIT,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_base_model(model_name: str, task_type: str, num_labels: int, quantize: bool = True):
    """
    Instantiate the correct head architecture for (model_name, task_type).

    Quantized models (Llama) are loaded with device_map="auto" so bitsandbytes
    can shard layers across available devices. RoBERTa models ignore the flag.
    Full fine-tuning runs for Llama pass quantize=False to load in fp16.
    """
    bnb_cfg = _QUANT_MAP.get(model_name) if quantize else None
    kwargs: dict = {}
    if bnb_cfg is not None:
        kwargs["quantization_config"] = bnb_cfg
        kwargs["device_map"] = "auto"

    if task_type == "classification":
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, **kwargs
        )
    elif task_type == "span_extraction":
        return AutoModelForQuestionAnswering.from_pretrained(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown task_type: {task_type!r}")


def _peft_task_type(task_type: str) -> TaskType:
    if task_type == "classification":
        return TaskType.SEQ_CLS
    elif task_type == "span_extraction":
        return TaskType.QUESTION_ANS
    raise ValueError(f"Unknown task_type: {task_type!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_lora_model(
    rank: int,
    model_name: str = DEFAULT_MODEL,
    task_type: str = "classification",
    num_labels: int = 2,
) -> PeftModel:
    """
    Return a LoRA-wrapped model for any supported model_name.

    Target modules are sourced from MODEL_REGISTRY so each architecture adapts
    the correct projection matrices. Llama models receive prepare_model_for_kbit_training
    before LoRA adapters are attached. lora_alpha = 2*rank (standard doubling heuristic).
    """
    model_cfg = MODEL_REGISTRY[model_name]
    base = _load_base_model(model_name, task_type, num_labels, quantize=True)

    if _QUANT_MAP.get(model_name) is not None:
        # Casts LayerNorms to fp32 and enables gradient checkpointing; required
        # before LoRA adapters are attached to a quantized model.
        base = prepare_model_for_kbit_training(base)

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,   # effective scale = alpha/r = 2; constant across all ranks
        lora_dropout=0.1,
        bias="none",           # training bias terms would break the low-rank structure
        target_modules=model_cfg.lora_target_modules,
        task_type=_peft_task_type(task_type),
    )
    return get_peft_model(base, lora_cfg)


def get_full_model(
    model_name: str = DEFAULT_MODEL,
    task_type: str = "classification",
    num_labels: int = 2,
):
    """
    Return a model with all parameters unfrozen for full fine-tuning.

    Llama models are loaded without quantization — bitsandbytes quantized
    weights are frozen by design and cannot be updated end-to-end.
    """
    model = _load_base_model(model_name, task_type, num_labels, quantize=False)
    for param in model.parameters():
        param.requires_grad = True
    return model


def trainable_param_summary(model) -> dict[str, int | float]:
    """Return a dict with total, trainable, and trainable-% parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_pct": round(100 * trainable / total, 3),
    }
