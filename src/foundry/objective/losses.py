"""Numerically stable losses implemented with :mod:`math`.

Everything here works on plain Python floats and sequences. The functions avoid
``exp`` overflow by subtracting the maximum logit before exponentiation and by
using a branch-free log-sigmoid expression for the DPO term.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


def _check_finite(values: Sequence[float], label: str) -> None:
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(f"{label}[{index}] must be finite, got {value!r}")


def log_softmax(logits: Sequence[float]) -> list[float]:
    """Return ``log_softmax(logits)`` without overflow.

    ``logits`` must be a non-empty sequence of finite floats.
    """

    values = list(logits)
    if not values:
        raise ValueError("logits must not be empty")
    _check_finite(values, "logits")
    maximum = max(values)
    shifted = [value - maximum for value in values]
    log_denominator = math.log(sum(math.exp(value) for value in shifted))
    return [value - log_denominator for value in shifted]


def cross_entropy(logits: Sequence[float], target: int) -> float:
    """Cross-entropy for a single next-token target index."""

    values = list(logits)
    if isinstance(target, bool) or not isinstance(target, int):
        raise ValueError("target must be an integer index")
    if not 0 <= target < len(values):
        raise ValueError(
            f"target index {target} is outside the vocabulary of size {len(values)}"
        )
    return -log_softmax(values)[target]


def sigmoid(value: float) -> float:
    """Numerically stable logistic function."""

    if not math.isfinite(value):
        raise ValueError("sigmoid input must be finite")
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def log_sigmoid(value: float) -> float:
    """Stable ``log(sigmoid(value))`` for arbitrary finite inputs."""

    if not math.isfinite(value):
        raise ValueError("log_sigmoid input must be finite")
    if value >= 0.0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def masked_cross_entropy(
    logits: Sequence[Sequence[float]],
    targets: Sequence[int],
    mask: Sequence[bool | int],
) -> float:
    """Mean token cross-entropy over positions where ``mask`` is true.

    ``logits`` is a sequence of per-position logit vectors, ``targets`` the
    matching token ids, and ``mask`` a per-position boolean/int selector. The
    reduction is a plain mean over selected positions; if a batch has no
    selected positions the result is undefined and this raises ``ValueError``.
    """

    rows = list(logits)
    target_list = list(targets)
    mask_list = list(mask)
    if not rows:
        raise ValueError("logits must contain at least one position")
    if len(rows) != len(target_list) or len(rows) != len(mask_list):
        raise ValueError("logits, targets, and mask must have the same length")
    total = 0.0
    selected = 0
    for index, (row, target, keep) in enumerate(zip(rows, target_list, mask_list)):
        if isinstance(keep, bool):
            keep_position = keep
        elif isinstance(keep, int) and keep in (0, 1):
            keep_position = bool(keep)
        else:
            raise ValueError(f"mask[{index}] must be a bool or 0/1")
        if not keep_position:
            continue
        if isinstance(target, bool) or not isinstance(target, int):
            raise ValueError(f"targets[{index}] must be an integer index")
        total += cross_entropy(row, target)
        selected += 1
    if selected == 0:
        raise ValueError("mask selects no positions; loss is undefined")
    return total / selected


def dpo_loss(
    policy_chosen_logp: float,
    policy_rejected_logp: float,
    ref_chosen_logp: float,
    ref_rejected_logp: float,
    beta: float = 0.1,
) -> float:
    """Direct Preference Optimization loss from log-probability margins.

    The implicit reward margin is

    ``beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))``

    and the loss is ``-log(sigmoid(margin))``, computed in a stable form.
    """

    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be a positive finite number")
    values = [policy_chosen_logp, policy_rejected_logp, ref_chosen_logp, ref_rejected_logp]
    _check_finite([float(value) for value in values], "log probability")
    for label, value in zip(
        ("policy_chosen_logp", "policy_rejected_logp", "ref_chosen_logp", "ref_rejected_logp"),
        values,
    ):
        if value > 0.0:
            raise ValueError(f"{label} must not be positive (log probabilities are <= 0)")
    margin = beta * (
        (policy_chosen_logp - policy_rejected_logp)
        - (ref_chosen_logp - ref_rejected_logp)
    )
    return -log_sigmoid(margin)


@dataclass(frozen=True)
class DPOReport:
    """Loss and diagnostics for one preference pair or batch average."""

    loss: float
    margin: float
    chosen_reward: float
    rejected_reward: float
    preference_accuracy: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "margin": self.margin,
            "chosen_reward": self.chosen_reward,
            "rejected_reward": self.rejected_reward,
            "preference_accuracy": self.preference_accuracy,
        }


def dpo_report(
    policy_chosen_logp: float,
    policy_rejected_logp: float,
    ref_chosen_logp: float,
    ref_rejected_logp: float,
    beta: float = 0.1,
) -> DPOReport:
    """Return the DPO loss plus implicit rewards and pairwise accuracy."""

    loss = dpo_loss(
        policy_chosen_logp,
        policy_rejected_logp,
        ref_chosen_logp,
        ref_rejected_logp,
        beta=beta,
    )
    margin = beta * (
        (policy_chosen_logp - policy_rejected_logp)
        - (ref_chosen_logp - ref_rejected_logp)
    )
    chosen_reward = beta * (policy_chosen_logp - ref_chosen_logp)
    rejected_reward = beta * (policy_rejected_logp - ref_rejected_logp)
    if margin > 0.0:
        accuracy = 1.0
    elif margin < 0.0:
        accuracy = 0.0
    else:
        accuracy = 0.5
    return DPOReport(
        loss=loss,
        margin=margin,
        chosen_reward=chosen_reward,
        rejected_reward=rejected_reward,
        preference_accuracy=accuracy,
    )
