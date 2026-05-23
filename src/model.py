"""
Model loading and LoRA / QLoRA configuration for the rank sweep experiment.

Supported models
----------------
  roberta-base                   125 M   encoder   Standard LoRA   attn: Q/K/V   attn_mlp: Q/K/V/dense
  roberta-large                  355 M   encoder   Standard LoRA   attn: Q/K/V   attn_mlp: Q/K/V/dense
  bert_uncased_L-2_H-128_A-2      4 M   encoder   Standard LoRA   attn: Q/K/V   attn_mlp: Q/K/V/dense
  bert_uncased_L-4_H-256_A-4     11 M   encoder   Standard LoRA   attn: Q/K/V   attn_mlp: Q/K/V/dense
  bert_uncased_L-4_H-512_A-8     29 M   encoder   Standard LoRA   attn: Q/K/V   attn_mlp: Q/K/V/dense
  bert-base-uncased              110 M   encoder   Standard LoRA   attn: Q/K/V   attn_mlp: Q/K/V/dense
  bert-large-uncased             336 M   encoder   Standard LoRA   attn: Q/K/V   attn_mlp: Q/K/V/dense
  Llama-3.2-1B                     1 B   decoder   QLoRA  4-bit    attn: q/v_proj   attn_mlp: q/v/up/down/gate_proj
  Llama-3.2-3B                     3 B   decoder   QLoRA  4-bit    attn: q/v_proj   attn_mlp: q/v/up/down/gate_proj

Entry points
------------
  get_lora_model(rank, model_name, task_type, num_labels, variant) → PeftModel
  get_full_model(model_name, task_type, num_labels)       → base model
  trainable_param_summary(model)                          → dict
"""

from __future__ import annotations

# import torch  # re-enable for _BNBCONFIG_4BIT when Llama models are active
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    # BitsAndBytesConfig,  # re-enable when Llama models are active
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel  # , prepare_model_for_kbit_training

from src.config import MODEL_REGISTRY, DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Quantization configs (bitsandbytes) — re-enable when Llama models are active
# ---------------------------------------------------------------------------

# _BNBCONFIG_8BIT = BitsAndBytesConfig(load_in_8bit=True)

# _BNBCONFIG_4BIT = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )

# Maps each supported model to its quantization config (None = no quantization).
_QUANT_MAP: dict[str, None] = {
    "roberta-base":                          None,
    "roberta-large":                         None,
    "google/bert_uncased_L-2_H-128_A-2":    None,
    "google/bert_uncased_L-4_H-256_A-4":    None,
    "google/bert_uncased_L-4_H-512_A-8":    None,
    "bert-base-uncased":                     None,
    "bert-large-uncased":                    None,
    # "meta-llama/Llama-3.2-1B":            _BNBCONFIG_4BIT,
    # "meta-llama/Llama-3.2-3B":            _BNBCONFIG_4BIT,
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
    variant: str = "attn",
) -> PeftModel:
    """
    Return a LoRA-wrapped model for any supported model_name.

    variant="attn"     — QKV projections only
    variant="attn_mlp" — QKV + all dense layers (attention output + FFN up/down)
    """
    model_cfg = MODEL_REGISTRY[model_name]
    base = _load_base_model(model_name, task_type, num_labels, quantize=True)

    # if _QUANT_MAP.get(model_name) is not None:
    #     base = prepare_model_for_kbit_training(base)

    if variant == "attn":
        target_modules = model_cfg.lora_attn_modules
    elif variant == "attn_mlp":
        target_modules = model_cfg.lora_attn_mlp_modules
    else:
        raise ValueError(f"Unknown LoRA variant: {variant!r}. Choose 'attn' or 'attn_mlp'.")

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,   # effective scale = alpha/r = 2; constant across all ranks
        lora_dropout=0.1,
        bias="none",           # training bias terms would break the low-rank structure
        target_modules=target_modules,
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
