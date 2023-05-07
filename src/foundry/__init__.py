"""Foundry -- LLM fine-tuning and training utilities.

The package is deliberately dependency-free: dataset preparation, adapter
accounting, schedules, losses, and a deterministic training loop all run on the
Python standard library.
"""

from .data.dataset import (
    ChatTemplate,
    DatasetError,
    DatasetStats,
    DPORecord,
    Message,
    SFTRecord,
    Split,
    dataset_fingerprint,
    dataset_stats,
    detect_kind,
    estimate_tokens,
    load_dataset,
    load_dpo,
    load_sft,
    render_chat_template,
    split_records,
    write_jsonl,
)
from .export.export import (
    AdapterMetadata,
    finalize_adapter_metadata,
    render_model_card,
    write_adapter_metadata,
)
from .adapter.lora import LoRAConfig, LoRAPlan, ModulePlan, plan_adapter, trainable_parameters
from .objective.losses import (
    DPOReport,
    cross_entropy,
    dpo_loss,
    dpo_report,
    log_sigmoid,
    log_softmax,
    masked_cross_entropy,
    sigmoid,
)
from .schedules.schedule import LRSchedule, warmup_cosine
from .pipeline.trainer import (
    StepRecord,
    SyntheticModel,
    Trainer,
    TrainingConfig,
    TrainingHistory,
    TrainableModel,
)

__version__ = "0.1.0"

__all__ = [
    "AdapterMetadata",
    "ChatTemplate",
    "DPORecord",
    "DPOReport",
    "DatasetError",
    "DatasetStats",
    "LRSchedule",
    "LoRAConfig",
    "LoRAPlan",
    "Message",
    "ModulePlan",
    "SFTRecord",
    "Split",
    "StepRecord",
    "SyntheticModel",
    "TrainableModel",
    "Trainer",
    "TrainingConfig",
    "TrainingHistory",
    "cross_entropy",
    "dataset_fingerprint",
    "dataset_stats",
    "detect_kind",
    "dpo_loss",
    "dpo_report",
    "estimate_tokens",
    "finalize_adapter_metadata",
    "load_dataset",
    "load_dpo",
    "load_sft",
    "log_sigmoid",
    "log_softmax",
    "masked_cross_entropy",
    "plan_adapter",
    "render_chat_template",
    "render_model_card",
    "sigmoid",
    "split_records",
    "trainable_parameters",
    "warmup_cosine",
    "write_adapter_metadata",
    "write_jsonl",
]
