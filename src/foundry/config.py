"""Declarative pipeline configuration.

A Foundry run is described by a small JSON recipe rather than by a pile of flags.
``FoundryConfig`` loads that recipe, rejects unknown keys, and derives the adapter
shape and learning-rate schedule the rest of the pipeline consumes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from .adapter.lora import LoRAConfig
from .schedules.schedule import LRSchedule, warmup_cosine


class ConfigError(ValueError):
    """Raised when a recipe is unreadable or contains unknown keys."""


@dataclass(frozen=True)
class FoundryConfig:
    """A frozen recipe for one fine-tuning experiment."""

    dataset: str
    kind: str = "auto"
    val_ratio: float = 0.1
    seed: int = 17
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    lr: float = 2e-4
    steps: int = 1000
    warmup: int = 50
    min_lr_ratio: float = 0.1

    # ------------------------------------------------------------- loading
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FoundryConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"unknown config keys: {', '.join(unknown)}")
        payload = dict(data)
        if "target_modules" in payload:
            payload["target_modules"] = tuple(payload["target_modules"])
        try:
            return cls(**payload)
        except TypeError as exc:  # pragma: no cover - guarded by known keys
            raise ConfigError(str(exc)) from exc

    @classmethod
    def load(cls, path: str | Path) -> "FoundryConfig":
        location = Path(path)
        try:
            raw = location.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read config {location}: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON in {location}: {exc}") from exc
        if not isinstance(data, Mapping):
            raise ConfigError("config must be a JSON object")
        return cls.from_dict(data)

    # ------------------------------------------------------------ derived
    def to_lora(self) -> LoRAConfig:
        return LoRAConfig(
            r=self.lora_r,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
            target_modules=self.target_modules,
        )

    def to_schedule(self) -> LRSchedule:
        return LRSchedule(
            total_steps=self.steps,
            base_lr=self.lr,
            warmup_steps=self.warmup,
            min_lr_ratio=self.min_lr_ratio,
        )

    def lr_at(self, step: int) -> float:
        return warmup_cosine(
            step, self.steps, self.lr, self.warmup, self.min_lr_ratio
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "kind": self.kind,
            "val_ratio": self.val_ratio,
            "seed": self.seed,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": list(self.target_modules),
            "lr": self.lr,
            "steps": self.steps,
            "warmup": self.warmup,
            "min_lr_ratio": self.min_lr_ratio,
        }
