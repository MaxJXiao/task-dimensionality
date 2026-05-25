"""
Central configuration for the decoder LoRA rank sweep experiment.

Same structure as src/config.py but restricted to decoder-only models loaded in
plain fp16 (no QLoRA).  Batch sizes are smaller to compensate for the higher VRAM
footprint of unquantized weights.  Task definitions are separate and will be added
once the decoder task suite is finalised.
"""

from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# LoRA sweep
# ---------------------------------------------------------------------------

LORA_RANKS: List[int] = [1, 4, 16, 64, 256]

LORA_ALPHA_MULTIPLIER: float = 2.0
LORA_DROPOUT: float = 0.05


# ---------------------------------------------------------------------------
# Full-parameter baseline
# ---------------------------------------------------------------------------

INCLUDE_FULL_PARAM_BASELINE: bool = True


# ---------------------------------------------------------------------------
# Smoke-test mode overrides (--test flag)
# ---------------------------------------------------------------------------

TEST_TRAIN_SAMPLES: int = 10
TEST_EVAL_SAMPLES: int = 1
TEST_EPOCHS: int = 1
TEST_EVAL_STEPS: int = 2


# ---------------------------------------------------------------------------
# Decoder models — plain fp16, no quantization
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    hf_name: str
    params: str
    architecture: str
    lora_attn_modules: List[str]
    lora_attn_mlp_modules: List[str]
    max_lora_rank: int = 512
    learning_rate: Optional[float] = None


MODELS: List[ModelConfig] = [
    ModelConfig(
        hf_name="meta-llama/Llama-3.2-1B",
        params="1B",
        architecture="decoder",
        lora_attn_modules=["q_proj", "v_proj"],
        lora_attn_mlp_modules=["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
        max_lora_rank=512,
        learning_rate=2e-5,
    ),
    # ModelConfig(
    #     hf_name="meta-llama/Llama-3.2-3B",
    #     params="3B",
    #     architecture="decoder",
    #     lora_attn_modules=["q_proj", "v_proj"],
    #     lora_attn_mlp_modules=["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
    #     max_lora_rank=512,
    #     learning_rate=2e-5,
    # ),
]

MODEL_REGISTRY: dict[str, ModelConfig] = {m.hf_name: m for m in MODELS}
DEFAULT_MODEL: str = MODELS[0].hf_name


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_epochs: int = 20
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = True
    seed: int = 42
    eval_steps: int = 100


TRAINING = TrainingConfig()


# ---------------------------------------------------------------------------
# Per-task batch sizes (smaller than encoder; fp16 uses more VRAM than 4-bit)
# ---------------------------------------------------------------------------

BATCH_SIZE_CLS: int = 8
BATCH_SIZE_SQUAD: int = 4
BATCH_SIZE_CAUSAL_LM: int = 8


# ---------------------------------------------------------------------------
# Task definitions — to be added
# ---------------------------------------------------------------------------

@dataclass
class OodEvalConfig:
    """Out-of-distribution evaluation dataset for a task."""
    dataset_name: str
    dataset_config: Optional[str]
    text_column: str
    test_format: str        # currently only "humaneval"
    max_eval_samples: Optional[int] = None
    max_input_length: int = 512


@dataclass
class TaskConfig:
    name: str
    display_name: str
    dataset_name: str
    dataset_config: Optional[str]
    text_column: str
    second_text_column: Optional[str]
    label_column: str
    num_labels: int
    metric: str
    secondary_metric: Optional[str]
    sota_baseline: Optional[float]
    task_type: str
    label_names: List[str] = field(default_factory=list)
    max_input_length: int = 128
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    num_epochs: Optional[int] = None
    eval_steps: Optional[int] = None
    eval_split: str = "validation"
    ood_eval: Optional[OodEvalConfig] = None
    # prompt template for causal_lm tasks; empty strings give bare text→label format
    prompt_prefix: str = ""
    prompt_suffix: str = ""


MAX_TRAIN_SAMPLES: Optional[int] = 10000
MAX_EVAL_SAMPLES: Optional[int] = 2000

TASKS: List[TaskConfig] = [
    TaskConfig(
        name="mbpp",
        display_name="MBPP Code Generation",
        dataset_name="google-research-datasets/mbpp",
        dataset_config=None,
        text_column="text",
        second_text_column=None,
        label_column="code",
        num_labels=1,
        metric="pass_at_1",
        secondary_metric=None,
        sota_baseline=None,
        task_type="code_generation",
        max_input_length=512,
        max_train_samples=300,
        max_eval_samples=74,
        num_epochs=20,
        eval_steps=50,
        ood_eval=OodEvalConfig(
            dataset_name="openai/openai_humaneval",
            dataset_config=None,
            text_column="prompt",
            test_format="humaneval",
            max_eval_samples=None,
            max_input_length=512,
        ),
    ),
    TaskConfig(
        name="gsm8k",
        display_name="GSM8K Math Word Problems",
        dataset_name="openai/gsm8k",
        dataset_config="main",
        text_column="question",
        second_text_column=None,
        label_column="answer",
        num_labels=1,
        metric="exact_match",
        secondary_metric=None,
        sota_baseline=None,
        task_type="math_reasoning",
        max_input_length=512,
        max_train_samples=2000,
        max_eval_samples=400,
        num_epochs=20,
        eval_steps=100,
        eval_split="test",
    ),
    TaskConfig(
        name="trivia_qa",
        display_name="TriviaQA Generative QA",
        dataset_name="trivia_qa",
        dataset_config="rc",
        text_column="question",
        second_text_column=None,
        label_column="answer",
        num_labels=1,
        metric="f1",
        secondary_metric="exact_match",
        sota_baseline=None,
        task_type="generative_qa",
        max_input_length=256,
        max_train_samples=2000,
        max_eval_samples=400,
        num_epochs=20,
        eval_steps=100,
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
