"""Linear warmup followed by cosine decay.

The schedule is defined for discrete optimizer steps ``0..total_steps``.
Step ``0`` is the learning rate used by the first update, not a "before
training" value. During warmup the rate rises linearly from
``base_lr / warmup_steps`` to ``base_lr``; afterwards it follows a cosine curve
down to ``min_lr_ratio * base_lr`` at ``total_steps``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LRSchedule:
    """A warmup + cosine learning-rate schedule.

    Attributes:
        total_steps: Final optimizer step, inclusive.
        base_lr: Peak learning rate reached at the end of warmup.
        warmup_steps: Number of linear warmup steps (``0`` disables warmup).
        min_lr_ratio: Floor as a fraction of ``base_lr``, in ``[0, 1]``.
    """

    total_steps: int
    base_lr: float
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.total_steps, bool) or not isinstance(self.total_steps, int):
            raise ValueError("total_steps must be an integer")
        if self.total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        if not math.isfinite(float(self.base_lr)) or float(self.base_lr) <= 0:
            raise ValueError("base_lr must be a positive finite number")
        if isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int):
            raise ValueError("warmup_steps must be an integer")
        if not 0 <= self.warmup_steps <= self.total_steps:
            raise ValueError("warmup_steps must be between 0 and total_steps")
        if not math.isfinite(float(self.min_lr_ratio)) or not 0.0 <= float(
            self.min_lr_ratio
        ) <= 1.0:
            raise ValueError("min_lr_ratio must be in the range [0.0, 1.0]")

    @property
    def min_lr(self) -> float:
        return float(self.base_lr) * float(self.min_lr_ratio)

    def __call__(self, step: int) -> float:
        """Learning rate at ``step``; negative steps are rejected, large ones clamp."""

        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError("step must be an integer")
        if step < 0:
            raise ValueError("step must not be negative")
        position = min(step, self.total_steps)
        if self.warmup_steps > 0 and position < self.warmup_steps:
            return float(self.base_lr) * (position + 1) / self.warmup_steps
        if self.total_steps == self.warmup_steps:
            return float(self.base_lr)
        progress = (position - self.warmup_steps) / (
            self.total_steps - self.warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (float(self.base_lr) - self.min_lr) * cosine

    def values(self) -> tuple[float, ...]:
        """Every learning rate from step 0 through ``total_steps``."""

        return tuple(self(step) for step in range(self.total_steps + 1))

    def sample(self, count: int) -> list[tuple[int, float]]:
        """Return ``count`` evenly spaced ``(step, lr)`` pairs, endpoints included."""

        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError("count must be an integer")
        if count < 2:
            raise ValueError("count must be at least 2 to include both endpoints")
        if count > self.total_steps + 1:
            raise ValueError("count cannot exceed total_steps + 1")
        spacing = self.total_steps / (count - 1)
        steps = sorted({int(round(index * spacing)) for index in range(count)})
        return [(step, self(step)) for step in steps]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": "linear-warmup-cosine",
            "total_steps": self.total_steps,
            "base_lr": float(self.base_lr),
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": float(self.min_lr_ratio),
            "min_lr": self.min_lr,
        }


def warmup_cosine(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> float:
    """Convenience wrapper around :class:`LRSchedule` for a single lookup."""

    return LRSchedule(
        total_steps=total_steps,
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )(step)
