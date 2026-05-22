"""
Central configuration for the task dimensionality / LoRA requisite rank experiment.

Hypothesis: for any task there exists an optimal LoRA rank R* beyond which extra
parameters hurt rather than help, and R* is a property of the task, not the model.
"""

from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# LoRA sweep
# ---------------------------------------------------------------------------

LORA_RANKS: List[int] = [1, 2, 4, 8, 16, 32, 64]

LORA_ALPHA_MULTIPLIER: float = 2.0  # alpha = rank * multiplier; keeps effective scale constant across ranks
LORA_DROPOUT: float = 0.05


# ---------------------------------------------------------------------------
# Full-parameter baseline
# ---------------------------------------------------------------------------

INCLUDE_FULL_PARAM_BASELINE: bool = True  # train one full-finetune run per task/model for comparison


# ---------------------------------------------------------------------------
# Base models to sweep
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    hf_name: str
    params: str
    architecture: str                    # "encoder" | "decoder"
    lora_target_modules: List[str]
    learning_rate: Optional[float] = None  # overrides TrainingConfig.learning_rate when set


MODELS: List[ModelConfig] = [
    ModelConfig(
        hf_name="roberta-base",
        params="125M",
        architecture="encoder",
        lora_target_modules=["query", "key", "value", "dense"],
    ),
    ModelConfig(
        hf_name="roberta-large",
        params="355M",
        architecture="encoder",
        lora_target_modules=["query", "key", "value", "dense"],
    ),
    ModelConfig(
        hf_name="google/bert_uncased_L-2_H-128_A-2",
        params="4M",
        architecture="encoder",
        lora_target_modules=["query", "key", "value", "dense"],
    ),
    ModelConfig(
        hf_name="google/bert_uncased_L-4_H-256_A-4",
        params="11M",
        architecture="encoder",
        lora_target_modules=["query", "key", "value", "dense"],
    ),
    ModelConfig(
        hf_name="google/bert_uncased_L-4_H-512_A-8",
        params="29M",
        architecture="encoder",
        lora_target_modules=["query", "key", "value", "dense"],
    ),
    ModelConfig(
        hf_name="bert-base-uncased",
        params="110M",
        architecture="encoder",
        lora_target_modules=["query", "key", "value", "dense"],
    ),
    ModelConfig(
        hf_name="bert-large-uncased",
        params="336M",
        architecture="encoder",
        lora_target_modules=["query", "key", "value", "dense"],
    ),
    # ModelConfig(
    #     hf_name="meta-llama/Llama-3.2-1B",
    #     params="1B",
    #     architecture="decoder",
    #     lora_target_modules=["q_proj", "v_proj"],
    #     learning_rate=3e-4,
    # ),
    # ModelConfig(
    #     hf_name="meta-llama/Llama-3.2-3B",
    #     params="3B",
    #     architecture="decoder",
    #     lora_target_modules=["q_proj", "v_proj"],
    #     learning_rate=3e-4,
    # ),
]

MODEL_REGISTRY: dict[str, ModelConfig] = {m.hf_name: m for m in MODELS}

DEFAULT_MODEL: str = MODELS[0].hf_name


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5 #3e-4 might be too aggressive for roberta-base, but i should change it back for Llama models
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    num_epochs: int = 50 # train to overfit
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = True
    seed: int = 42
    eval_steps: int = 100          # evaluate every N optimizer steps
    # early stoppping is deliberately removed


TRAINING = TrainingConfig()


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    name: str
    display_name: str                          # human-readable label for plots/logs
    dataset_name: str
    dataset_config: Optional[str]
    text_column: str
    second_text_column: Optional[str]          # set for sentence-pair and QA tasks
    label_column: str
    num_labels: int                            # 0 for span-extraction tasks
    metric: str                                # primary evaluate metric name
    secondary_metric: Optional[str]            # e.g. "exact_match" for SQuAD 2.0
    sota_baseline: float                       # reported SOTA / human baseline score
    task_type: str                             # "classification" | "span_extraction"
    label_names: List[str] = field(default_factory=list)
    max_input_length: int = 128
    max_train_samples: Optional[int] = None    # None → full split
    max_eval_samples: Optional[int] = None


# MAX_TRAIN_SAMPLES: Optional[int] = None
# MAX_EVAL_SAMPLES:  Optional[int] = None
MAX_TRAIN_SAMPLES: Optional[int] = 2000
MAX_EVAL_SAMPLES:  Optional[int] = 500


TASKS: List[TaskConfig] = [
    TaskConfig(
        name="sst2",
        display_name="Sentiment Analysis",
        dataset_name="glue",
        dataset_config="sst2",
        text_column="sentence",
        second_text_column=None,
        label_column="label",
        num_labels=2,
        metric="accuracy",
        secondary_metric=None,
        sota_baseline=96.4,
        task_type="classification",
        label_names=["negative", "positive"],
        max_input_length=64,
        max_train_samples=MAX_TRAIN_SAMPLES,
        max_eval_samples=MAX_EVAL_SAMPLES,
    ),
    TaskConfig(
        name="cola",
        display_name="Grammatical Acceptability",
        dataset_name="glue",
        dataset_config="cola",
        text_column="sentence",
        second_text_column=None,
        label_column="label",
        num_labels=2,
        metric="matthews_correlation",
        secondary_metric=None,
        sota_baseline=63.6,
        task_type="classification",
        label_names=["unacceptable", "acceptable"],
        max_input_length=64,
        max_train_samples=MAX_TRAIN_SAMPLES,
        max_eval_samples=MAX_EVAL_SAMPLES,
    ),
    TaskConfig(
        name="snli",
        display_name="Natural Language Inference",
        dataset_name="snli",
        dataset_config=None,
        text_column="premise",
        second_text_column="hypothesis",
        label_column="label",
        num_labels=3,
        metric="accuracy",
        secondary_metric=None,
        sota_baseline=91.8,
        task_type="classification",
        label_names=["entailment", "neutral", "contradiction"],
        max_input_length=128,
        max_train_samples=MAX_TRAIN_SAMPLES,
        max_eval_samples=MAX_EVAL_SAMPLES,
    ),
    TaskConfig(
        name="squad2",
        display_name="Question Answering",
        dataset_name="rajpurkar/squad_v2",
        dataset_config=None,
        text_column="context",
        second_text_column="question",
        label_column="answers",
        num_labels=0,                          # span extraction — no fixed label set
        metric="f1",
        secondary_metric="exact_match",
        sota_baseline=86.5,                    # F1; EM baseline is 83.7
        task_type="span_extraction",
        label_names=[],
        max_input_length=384,                  # QA contexts are longer
        max_train_samples=MAX_TRAIN_SAMPLES,
        max_eval_samples=MAX_EVAL_SAMPLES,
    ),
]

TASK_REGISTRY: dict[str, TaskConfig] = {t.name: t for t in TASKS}


# ---------------------------------------------------------------------------
# Output / experiment tracking
# ---------------------------------------------------------------------------

RESULTS_DIR: str = "results"
CHECKPOINTS_DIR: str = "checkpoints"
FIGURES_DIR: str = "figures"

EXPERIMENT_LOG_FILE: str = "results/experiment_log.jsonl"
