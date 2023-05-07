"""LoRA configuration and parameter accounting.

Low-rank adapters replace a full ``out x in`` weight update with two matrices:
``A`` of shape ``r x in`` and ``B`` of shape ``out x r``. Only those matrices
are trained, so the adapter holds ``r * (in + out)`` parameters per layer while
the update is scaled by ``alpha / r`` at merge time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_TARGET_MODULES = ("q_proj", "v_proj")


@dataclass(frozen=True)
class LoRAConfig:
    """Adapter shape and merge settings.

    ``target_modules`` names the linear projections to adapt; the names are
    opaque labels here, so they can match any model naming convention.
    """

    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES

    def __post_init__(self) -> None:
        if isinstance(self.r, bool) or not isinstance(self.r, int):
            raise ValueError("LoRA rank r must be an integer")
        if self.r <= 0:
            raise ValueError("LoRA rank r must be positive")
        if not math.isfinite(float(self.alpha)) or float(self.alpha) <= 0:
            raise ValueError("LoRA alpha must be a positive finite number")
        if not math.isfinite(float(self.dropout)) or not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("LoRA dropout must be in the range [0.0, 1.0)")
        modules = self.target_modules
        if isinstance(modules, str):
            modules = (modules,)
        if not isinstance(modules, (tuple, list)):
            raise ValueError("LoRA target_modules must be a sequence of names")
        normalized: list[str] = []
        for name in modules:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("LoRA target module names must be non-empty strings")
            cleaned = name.strip()
            if cleaned not in normalized:
                normalized.append(cleaned)
        if not normalized:
            raise ValueError("LoRA target_modules must contain at least one module")
        object.__setattr__(self, "target_modules", tuple(normalized))

    @property
    def scaling(self) -> float:
        """The ``alpha / r`` multiplier applied to the low-rank update."""

        return float(self.alpha) / float(self.r)

    def as_dict(self) -> dict[str, Any]:
        return {
            "r": self.r,
            "alpha": float(self.alpha),
            "dropout": float(self.dropout),
            "target_modules": list(self.target_modules),
            "scaling": self.scaling,
        }


def trainable_parameters(config: LoRAConfig, in_features: int, out_features: int) -> int:
    """Number of trainable adapter parameters for one linear layer."""

    if in_features <= 0 or out_features <= 0:
        raise ValueError("layer dimensions must be positive")
    return config.r * (int(in_features) + int(out_features))


@dataclass(frozen=True)
class ModulePlan:
    """Parameter accounting for one adapted projection."""

    name: str
    in_features: int
    out_features: int
    trainable: int
    base: int

    @property
    def ratio(self) -> float:
        return self.trainable / self.base if self.base else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "trainable_parameters": self.trainable,
            "base_parameters": self.base,
            "trainable_ratio": round(self.ratio, 8),
        }


@dataclass(frozen=True)
class LoRAPlan:
    """Aggregated adapter plan over a set of target modules."""

    config: LoRAConfig
    modules: tuple[ModulePlan, ...]

    @property
    def total_trainable(self) -> int:
        return sum(module.trainable for module in self.modules)

    @property
    def total_base(self) -> int:
        return sum(module.base for module in self.modules)

    @property
    def overhead_ratio(self) -> float:
        """Trainable share of the targeted full-precision projections."""

        return self.total_trainable / self.total_base if self.total_base else 0.0

    @property
    def scaling(self) -> float:
        return self.config.scaling

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "modules": {module.name: module.as_dict() for module in self.modules},
            "total_trainable_parameters": self.total_trainable,
            "total_base_parameters": self.total_base,
            "overhead_ratio": round(self.overhead_ratio, 8),
        }


def plan_adapter(
    config: LoRAConfig, module_shapes: Mapping[str, tuple[int, int]]
) -> LoRAPlan:
    """Build a parameter plan for the configured target modules.

    ``module_shapes`` maps module names to ``(in_features, out_features)``.
    Every target module must be present; extra shapes are ignored.
    """

    if not isinstance(module_shapes, Mapping):
        raise ValueError("module_shapes must be a mapping of name -> (in, out)")
    missing = [name for name in config.target_modules if name not in module_shapes]
    if missing:
        raise ValueError(
            "missing shapes for target modules: " + ", ".join(sorted(missing))
        )
    modules: list[ModulePlan] = []
    for name in config.target_modules:
        shape = module_shapes[name]
        if not isinstance(shape, Sequence) or len(shape) != 2:
            raise ValueError(f"shape for {name!r} must be a (in_features, out_features) pair")
        in_features, out_features = int(shape[0]), int(shape[1])
        if in_features <= 0 or out_features <= 0:
            raise ValueError(f"shape for {name!r} must use positive dimensions")
        modules.append(
            ModulePlan(
                name=name,
                in_features=in_features,
                out_features=out_features,
                trainable=trainable_parameters(config, in_features, out_features),
                base=in_features * out_features,
            )
        )
    return LoRAPlan(config=config, modules=tuple(modules))
