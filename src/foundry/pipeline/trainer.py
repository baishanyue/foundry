"""A minimal, deterministic training-loop abstraction.

This module deliberately separates loop mechanics from model math:

``TrainableModel`` protocol
    A model exposes gradient accumulation, gradient scaling, an optimizer step,
    and JSON-serializable state. ``SyntheticModel`` below is the offline default.

``Trainer``
    Owns batching, epoch order, gradient accumulation, global-norm clipping,
    the learning-rate schedule, loss history, and JSON checkpoints.

The bundled ``SyntheticModel`` is a bigram transition table over a tiny
vocabulary. It exists so the loop can be exercised end to end in milliseconds,
without torch, without a GPU, and without network access. It is a plumbing
fixture, not a language model. A real backend can implement the same protocol
without changing the trainer.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..objective.losses import log_softmax
from ..schedules.schedule import LRSchedule

CHECKPOINT_FORMAT = "foundry.checkpoint"
CHECKPOINT_VERSION = 1


@runtime_checkable
class TrainableModel(Protocol):
    """Contract between :class:`Trainer` and any model backend.

    ``accumulate`` must add per-token gradients for the batch and return
    ``(mean_loss, token_count)``. ``scale_gradients`` multiplies every gradient
    by a scalar. ``apply_gradients`` performs one update and clears gradients.
    """

    def accumulate(self, batch: Sequence[Sequence[int]]) -> tuple[float, int]: ...

    def evaluate(self, batch: Sequence[Sequence[int]]) -> tuple[float, int]: ...

    def gradient_norm(self) -> float: ...

    def scale_gradients(self, factor: float) -> None: ...

    def apply_gradients(self, lr: float) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class SyntheticModel:
    """A deterministic next-token model backed by a dense transition table.

    Rows are logits for the token that follows each vocabulary item, so the
    model can only learn bigram statistics. Gradients are the exact softmax
    minus one-hot cross-entropy gradient, which keeps the fixture honest.
    """

    def __init__(self, vocab_size: int = 16, seed: int = 0) -> None:
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
            raise ValueError("vocab_size must be an integer")
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.vocab_size = vocab_size
        self.seed = seed
        generator = random.Random(seed)
        self.weights = [
            [generator.uniform(-0.1, 0.1) for _ in range(vocab_size)]
            for _ in range(vocab_size)
        ]
        self._gradients = [[0.0] * vocab_size for _ in range(vocab_size)]

    @property
    def num_parameters(self) -> int:
        return self.vocab_size * self.vocab_size

    def _validate_batch(self, batch: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for sequence in batch:
            if len(sequence) < 2:
                raise ValueError("every sequence must contain at least two tokens")
            for token in sequence:
                if isinstance(token, bool) or not isinstance(token, int):
                    raise ValueError("token ids must be integers")
                if not 0 <= token < self.vocab_size:
                    raise ValueError(
                        f"token id {token} is outside the vocabulary of size {self.vocab_size}"
                    )
            pairs.extend(zip(sequence, sequence[1:]))
        if not pairs:
            raise ValueError("batch contains no next-token targets")
        return pairs

    def accumulate(self, batch: Sequence[Sequence[int]]) -> tuple[float, int]:
        pairs = self._validate_batch(batch)
        loss_total = 0.0
        for source, target in pairs:
            row = self.weights[source]
            log_probabilities = log_softmax(row)
            loss_total -= log_probabilities[target]
            gradient_row = self._gradients[source]
            for index, log_probability in enumerate(log_probabilities):
                gradient_row[index] += math.exp(log_probability)
            gradient_row[target] -= 1.0
        return loss_total / len(pairs), len(pairs)

    def evaluate(self, batch: Sequence[Sequence[int]]) -> tuple[float, int]:
        pairs = self._validate_batch(batch)
        loss_total = 0.0
        for source, target in pairs:
            loss_total -= log_softmax(self.weights[source])[target]
        return loss_total / len(pairs), len(pairs)

    def gradient_norm(self) -> float:
        return math.sqrt(
            sum(
                gradient * gradient
                for row in self._gradients
                for gradient in row
            )
        )

    def scale_gradients(self, factor: float) -> None:
        if not math.isfinite(factor) or factor < 0.0:
            raise ValueError("gradient scale must be a non-negative finite number")
        for row in self._gradients:
            for index in range(len(row)):
                row[index] *= factor

    def apply_gradients(self, lr: float) -> None:
        if not math.isfinite(lr) or lr < 0.0:
            raise ValueError("learning rate must be a non-negative finite number")
        for row, gradient_row in zip(self.weights, self._gradients):
            for index in range(len(row)):
                row[index] -= lr * gradient_row[index]
                gradient_row[index] = 0.0

    def state_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "seed": self.seed,
            "weights": [row[:] for row in self.weights],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("model state must be a mapping")
        if state.get("vocab_size") != self.vocab_size:
            raise ValueError(
                "checkpoint vocabulary size does not match the model "
                f"({state.get('vocab_size')!r} != {self.vocab_size})"
            )
        weights = state.get("weights")
        if not isinstance(weights, list) or len(weights) != self.vocab_size:
            raise ValueError("checkpoint weights must match the model shape")
        restored: list[list[float]] = []
        for row in weights:
            if not isinstance(row, list) or len(row) != self.vocab_size:
                raise ValueError("checkpoint weights must match the model shape")
            values: list[float] = []
            for value in row:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("checkpoint weights must be numbers")
                values.append(float(value))
            restored.append(values)
        self.weights = restored
        self._gradients = [[0.0] * self.vocab_size for _ in range(self.vocab_size)]


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class TrainingConfig:
    """Loop settings shared by every backend."""

    epochs: int = 1
    batch_size: int = 4
    grad_accum_steps: int = 1
    lr: float = 3e-4
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0
    max_grad_norm: float | None = 1.0
    shuffle: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        _positive_int("epochs", self.epochs)
        _positive_int("batch_size", self.batch_size)
        _positive_int("grad_accum_steps", self.grad_accum_steps)
        if not math.isfinite(float(self.lr)) or float(self.lr) <= 0.0:
            raise ValueError("lr must be a positive finite number")
        if isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int):
            raise ValueError("warmup_steps must be an integer")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must not be negative")
        if not 0.0 <= float(self.min_lr_ratio) <= 1.0:
            raise ValueError("min_lr_ratio must be in the range [0.0, 1.0]")
        if self.max_grad_norm is not None:
            if (
                isinstance(self.max_grad_norm, bool)
                or not isinstance(self.max_grad_norm, (int, float))
                or not math.isfinite(float(self.max_grad_norm))
                or float(self.max_grad_norm) <= 0.0
            ):
                raise ValueError("max_grad_norm must be None or a positive finite number")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")

    def as_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "lr": float(self.lr),
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": float(self.min_lr_ratio),
            "max_grad_norm": None
            if self.max_grad_norm is None
            else float(self.max_grad_norm),
            "shuffle": self.shuffle,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class StepRecord:
    """One completed optimizer update."""

    step: int
    epoch: int
    lr: float
    loss: float
    grad_norm: float
    clipped_norm: float
    tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "lr": self.lr,
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "clipped_norm": self.clipped_norm,
            "tokens": self.tokens,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StepRecord":
        return cls(
            step=int(payload["step"]),
            epoch=int(payload["epoch"]),
            lr=float(payload["lr"]),
            loss=float(payload["loss"]),
            grad_norm=float(payload["grad_norm"]),
            clipped_norm=float(payload["clipped_norm"]),
            tokens=int(payload["tokens"]),
        )


@dataclass
class TrainingHistory:
    """Update-level records plus the raw micro-batch losses behind them."""

    config: TrainingConfig
    records: list[StepRecord]
    micro_losses: list[float]

    @property
    def losses(self) -> list[float]:
        return [record.loss for record in self.records]

    @property
    def final_loss(self) -> float | None:
        return self.records[-1].loss if self.records else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "records": [record.to_dict() for record in self.records],
            "micro_losses": list(self.micro_losses),
            "final_loss": self.final_loss,
        }

    @classmethod
    def from_dict(cls, config: TrainingConfig, payload: Mapping[str, Any]) -> "TrainingHistory":
        records = [
            StepRecord.from_dict(item) for item in payload.get("records", [])
        ]
        micro_losses = [float(value) for value in payload.get("micro_losses", [])]
        return cls(config=config, records=records, micro_losses=micro_losses)


class Trainer:
    """Deterministic loop plumbing over a :class:`TrainableModel`."""

    def __init__(
        self,
        model: TrainableModel,
        config: TrainingConfig,
        schedule: LRSchedule | None = None,
    ) -> None:
        if not isinstance(config, TrainingConfig):
            raise ValueError("config must be a TrainingConfig instance")
        if not isinstance(model, TrainableModel):
            raise ValueError("model does not implement the TrainableModel contract")
        self.model = model
        self.config = config
        self.history = TrainingHistory(config=config, records=[], micro_losses=[])
        self.step = 0
        self.schedule = schedule

    def _batches(
        self, examples: Sequence[Sequence[int]], epoch: int
    ) -> list[list[Sequence[int]]]:
        indices = list(range(len(examples)))
        if self.config.shuffle:
            random.Random(self.config.seed + epoch).shuffle(indices)
        ordered = [examples[index] for index in indices]
        size = self.config.batch_size
        return [ordered[start : start + size] for start in range(0, len(ordered), size)]

    def _schedule_for(self, total_updates: int) -> LRSchedule:
        total_steps = self.step + total_updates
        warmup = min(self.config.warmup_steps, total_steps)
        return LRSchedule(
            total_steps=total_steps,
            base_lr=self.config.lr,
            warmup_steps=warmup,
            min_lr_ratio=self.config.min_lr_ratio,
        )

    def fit(self, examples: Sequence[Sequence[int]]) -> TrainingHistory:
        """Train for ``config.epochs`` passes over ``examples``.

        Gradient accumulation sums per-token gradients and divides by the total
        token count in the accumulation window, so a short final window is not
        up-weighted. The gradient norm is recorded after that normalization and
        before clipping.
        """

        items = list(examples)
        if not items:
            raise ValueError("fit() requires at least one example")
        batches_per_epoch = math.ceil(len(items) / self.config.batch_size)
        updates_per_epoch = math.ceil(
            batches_per_epoch / self.config.grad_accum_steps
        )
        schedule = self._schedule_for(updates_per_epoch * self.config.epochs)
        self.schedule = schedule

        for epoch in range(1, self.config.epochs + 1):
            batches = self._batches(items, epoch)
            accum_batches = 0
            accum_tokens = 0
            accum_loss = 0.0
            for batch_index, batch in enumerate(batches):
                loss, tokens = self.model.accumulate(batch)
                self.history.micro_losses.append(loss)
                accum_loss += loss * tokens
                accum_tokens += tokens
                accum_batches += 1
                is_last_batch = batch_index == len(batches) - 1
                if (
                    accum_batches < self.config.grad_accum_steps
                    and not is_last_batch
                ):
                    continue
                if accum_tokens == 0:
                    raise RuntimeError("accumulation window contained no target tokens")
                self.model.scale_gradients(1.0 / accum_tokens)
                grad_norm = self.model.gradient_norm()
                clipped_norm = grad_norm
                if (
                    self.config.max_grad_norm is not None
                    and grad_norm > self.config.max_grad_norm
                ):
                    self.model.scale_gradients(self.config.max_grad_norm / grad_norm)
                    clipped_norm = self.config.max_grad_norm
                lr = schedule(self.step)
                self.model.apply_gradients(lr)
                self.history.records.append(
                    StepRecord(
                        step=self.step,
                        epoch=epoch,
                        lr=lr,
                        loss=accum_loss / accum_tokens,
                        grad_norm=grad_norm,
                        clipped_norm=clipped_norm,
                        tokens=accum_tokens,
                    )
                )
                self.step += 1
                accum_batches = 0
                accum_tokens = 0
                accum_loss = 0.0
        return self.history

    def evaluate(self, examples: Sequence[Sequence[int]]) -> float:
        """Return token-weighted mean loss over ``examples`` without updating."""

        items = list(examples)
        if not items:
            raise ValueError("evaluate() requires at least one example")
        total_loss = 0.0
        total_tokens = 0
        size = self.config.batch_size
        for start in range(0, len(items), size):
            loss, tokens = self.model.evaluate(items[start : start + size])
            total_loss += loss * tokens
            total_tokens += tokens
        if total_tokens == 0:
            raise ValueError("evaluation data contains no target tokens")
        return total_loss / total_tokens

    def save_checkpoint(self, path: str | Path) -> Path:
        """Write model state, config, schedule step, and history as JSON."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "step": self.step,
            "config": self.config.as_dict(),
            "model": {
                "type": type(self.model).__name__,
                "state": self.model.state_dict(),
            },
            "history": self.history.to_dict(),
        }
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load_checkpoint(
        cls, path: str | Path, model: TrainableModel
    ) -> "Trainer":
        """Restore a trainer (including model weights) from a JSON checkpoint."""

        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"checkpoint not found: {source}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"checkpoint is not valid JSON: {source} ({exc})") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("checkpoint must be a JSON object")
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("unrecognized checkpoint format")
        if payload.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"unsupported checkpoint version: {payload.get('version')!r}"
            )
        config_payload = payload.get("config")
        if not isinstance(config_payload, Mapping):
            raise ValueError("checkpoint is missing its training config")
        try:
            config = TrainingConfig(**config_payload)
        except TypeError as exc:
            raise ValueError(f"checkpoint has an invalid training config: {exc}") from exc
        trainer = cls(model=model, config=config)
        model_payload = payload.get("model")
        if not isinstance(model_payload, Mapping) or "state" not in model_payload:
            raise ValueError("checkpoint is missing model state")
        model.load_state_dict(model_payload["state"])
        step = payload.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("checkpoint step must be a non-negative integer")
        history_payload = payload.get("history")
        if not isinstance(history_payload, Mapping):
            raise ValueError("checkpoint is missing training history")
        trainer.history = TrainingHistory.from_dict(config, history_payload)
        if len(trainer.history.records) != step:
            raise ValueError(
                "checkpoint step does not match the number of history records"
            )
        trainer.step = step
        return trainer
