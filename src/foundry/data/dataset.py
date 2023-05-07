"""Dataset loading, schema validation, chat rendering, and length statistics.

Foundry supports two JSONL shapes:

* SFT -- ``{"messages": [{"role": "user", "content": "..."}, ...]}``
* DPO -- ``{"prompt": "...", "chosen": "...", "rejected": "..."}``

The toolkit ships no tokenizer. Token counts in this module are **character
based estimates** (``ceil(characters / 4)``) and are only useful for budgeting
memory and sequence length. They are not tokenizer output and will differ from
the counts used by a real training stack.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHARS_PER_TOKEN = 4.0
ALLOWED_ROLES = ("system", "user", "assistant", "tool")
SUPPORTED_KINDS = ("sft", "dpo")


class DatasetError(ValueError):
    """Raised when a file or record does not match the expected schema."""


@dataclass(frozen=True)
class Message:
    """One chat turn with a validated role and non-empty text content."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in ALLOWED_ROLES:
            raise DatasetError(
                f"unknown role {self.role!r}; expected one of {', '.join(ALLOWED_ROLES)}"
            )
        if not isinstance(self.content, str) or not self.content.strip():
            raise DatasetError("message content must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class SFTRecord:
    """A supervised fine-tuning conversation whose final turn is an assistant."""

    messages: tuple[Message, ...]
    record_id: str | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            raise DatasetError("SFT record must contain at least one message")
        if not all(isinstance(message, Message) for message in self.messages):
            raise DatasetError("SFT messages must be Message instances")
        if not any(message.role == "assistant" for message in self.messages):
            raise DatasetError("SFT record must contain at least one assistant turn")
        if self.messages[-1].role != "assistant":
            raise DatasetError("the final SFT message must be an assistant turn")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": [message.to_dict() for message in self.messages]
        }
        if self.record_id is not None:
            payload["id"] = self.record_id
        return payload


@dataclass(frozen=True)
class DPORecord:
    """A preference pair: one prompt with a chosen and a rejected completion."""

    prompt: str
    chosen: str
    rejected: str
    record_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("prompt", "chosen", "rejected"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise DatasetError(f"DPO field {name!r} must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }
        if self.record_id is not None:
            payload["id"] = self.record_id
        return payload


@dataclass(frozen=True)
class Split:
    """A deterministic train/validation partition."""

    train: tuple[Any, ...]
    val: tuple[Any, ...]

    @property
    def total(self) -> int:
        return len(self.train) + len(self.val)


def _error(path: Path, lineno: int, message: str) -> DatasetError:
    return DatasetError(f"{path}:{lineno}: {message}")


def _read_jsonl(path: str | Path) -> list[tuple[int, Any]]:
    source = Path(path)
    if not source.exists():
        raise DatasetError(f"{source}: file not found")
    if not source.is_file():
        raise DatasetError(f"{source}: not a regular file")
    rows: list[tuple[int, Any]] = []
    try:
        with source.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rows.append((lineno, json.loads(stripped)))
                except json.JSONDecodeError as exc:
                    raise _error(
                        source,
                        lineno,
                        f"invalid JSON ({exc.msg} at column {exc.colno})",
                    ) from exc
    except UnicodeDecodeError as exc:
        raise DatasetError(f"{source}: file is not valid UTF-8 ({exc})") from exc
    return rows


def _optional_id(raw: Mapping[str, Any], path: Path, lineno: int) -> str | None:
    value = raw.get("id")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _error(path, lineno, "record 'id' must be a non-empty string when present")
    return value


def _parse_message(raw: Any, path: Path, lineno: int, index: int) -> Message:
    if not isinstance(raw, Mapping):
        raise _error(path, lineno, f"messages[{index}] must be a JSON object")
    role = raw.get("role")
    if not isinstance(role, str):
        raise _error(path, lineno, f"messages[{index}].role must be a string")
    if role not in ALLOWED_ROLES:
        raise _error(
            path,
            lineno,
            f"messages[{index}].role is {role!r}; expected one of "
            f"{', '.join(ALLOWED_ROLES)}",
        )
    content = raw.get("content")
    if not isinstance(content, str):
        raise _error(path, lineno, f"messages[{index}].content must be a string")
    if not content.strip():
        raise _error(path, lineno, f"messages[{index}].content must not be empty")
    return Message(role=role, content=content)


def _parse_sft(raw: Any, path: Path, lineno: int) -> SFTRecord:
    if not isinstance(raw, Mapping):
        raise _error(path, lineno, "SFT record must be a JSON object")
    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        raise _error(path, lineno, "'messages' must be a non-empty array")
    parsed = tuple(
        _parse_message(item, path, lineno, index)
        for index, item in enumerate(messages)
    )
    if not any(message.role == "assistant" for message in parsed):
        raise _error(path, lineno, "SFT record must contain at least one assistant turn")
    if parsed[-1].role != "assistant":
        raise _error(
            path,
            lineno,
            "the final SFT message must be an assistant turn (the training target)",
        )
    return SFTRecord(messages=parsed, record_id=_optional_id(raw, path, lineno))


def _parse_dpo(raw: Any, path: Path, lineno: int) -> DPORecord:
    if not isinstance(raw, Mapping):
        raise _error(path, lineno, "DPO record must be a JSON object")
    values: dict[str, str] = {}
    for name in ("prompt", "chosen", "rejected"):
        value = raw.get(name)
        if not isinstance(value, str):
            raise _error(path, lineno, f"DPO field {name!r} must be a string")
        if not value.strip():
            raise _error(path, lineno, f"DPO field {name!r} must not be empty")
        values[name] = value
    return DPORecord(
        prompt=values["prompt"],
        chosen=values["chosen"],
        rejected=values["rejected"],
        record_id=_optional_id(raw, path, lineno),
    )


def load_sft(path: str | Path) -> list[SFTRecord]:
    """Load and validate a JSONL file of supervised fine-tuning records."""

    source = Path(path)
    return [_parse_sft(raw, source, lineno) for lineno, raw in _read_jsonl(source)]


def load_dpo(path: str | Path) -> list[DPORecord]:
    """Load and validate a JSONL file of preference pairs."""

    source = Path(path)
    return [_parse_dpo(raw, source, lineno) for lineno, raw in _read_jsonl(source)]


def _read_first_jsonl(path: str | Path) -> tuple[int, Any]:
    source = Path(path)
    if not source.exists():
        raise DatasetError(f"{source}: file not found")
    if not source.is_file():
        raise DatasetError(f"{source}: not a regular file")
    try:
        with source.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    return lineno, json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise _error(
                        source,
                        lineno,
                        f"invalid JSON ({exc.msg} at column {exc.colno})",
                    ) from exc
    except UnicodeDecodeError as exc:
        raise DatasetError(f"{source}: file is not valid UTF-8 ({exc})") from exc
    raise DatasetError(f"{source}: dataset is empty")


def detect_kind(path: str | Path) -> str:
    """Return ``"sft"`` or ``"dpo"`` by inspecting the first JSON object."""

    source = Path(path)
    lineno, raw = _read_first_jsonl(source)
    if not isinstance(raw, Mapping):
        raise _error(source, lineno, "record must be a JSON object")
    if "messages" in raw:
        return "sft"
    if {"prompt", "chosen", "rejected"}.issubset(raw.keys()):
        return "dpo"
    raise _error(
        source,
        lineno,
        "cannot detect dataset kind; expected 'messages' (SFT) or "
        "'prompt', 'chosen', 'rejected' (DPO)",
    )


def load_dataset(
    path: str | Path, kind: str = "auto"
) -> tuple[str, list[SFTRecord] | list[DPORecord]]:
    """Load a dataset, detecting its kind when ``kind`` is ``"auto"``."""

    if kind not in (*SUPPORTED_KINDS, "auto"):
        raise DatasetError(
            f"unknown dataset kind {kind!r}; expected 'auto', 'sft', or 'dpo'"
        )
    resolved = detect_kind(path) if kind == "auto" else kind
    if resolved == "sft":
        return "sft", load_sft(path)
    return "dpo", load_dpo(path)


@dataclass(frozen=True)
class ChatTemplate:
    """A ChatML-style template with explicit special tokens.

    Each turn is rendered as ``<role_token>content<end_token>`` followed by a
    separator. A BOS token opens the sequence and an EOS token closes it, unless
    ``add_generation_prompt`` is used (an open assistant turn is emitted instead).
    """

    bos_token: str = "<|bos|>"
    eos_token: str = "<|eos|>"
    system_token: str = "<|system|>"
    user_token: str = "<|user|>"
    assistant_token: str = "<|assistant|>"
    tool_token: str = "<|tool|>"
    end_token: str = "<|end|>"
    separator: str = "\n"

    def role_token(self, role: str) -> str:
        tokens = {
            "system": self.system_token,
            "user": self.user_token,
            "assistant": self.assistant_token,
            "tool": self.tool_token,
        }
        if role not in tokens:
            raise DatasetError(
                f"unknown role {role!r}; expected one of {', '.join(ALLOWED_ROLES)}"
            )
        return tokens[role]

    def render(
        self, messages: Sequence[Message], add_generation_prompt: bool = False
    ) -> str:
        if not messages:
            raise DatasetError("cannot render an empty conversation")
        parts: list[str] = [self.bos_token]
        for message in messages:
            parts.append(self.role_token(message.role))
            parts.append(message.content)
            parts.append(self.end_token)
            parts.append(self.separator)
        if add_generation_prompt:
            parts.append(self.assistant_token)
            parts.append(self.separator)
        else:
            parts.append(self.eos_token)
        return "".join(parts)


DEFAULT_CHAT_TEMPLATE = ChatTemplate()


def render_chat_template(
    messages: Sequence[Message],
    template: ChatTemplate | None = None,
    add_generation_prompt: bool = False,
) -> str:
    """Render messages with special tokens using :class:`ChatTemplate`."""

    return (template or DEFAULT_CHAT_TEMPLATE).render(
        messages, add_generation_prompt=add_generation_prompt
    )


def estimate_tokens(text: str, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Return a character-based token estimate: ``ceil(len(text) / 4)``."""

    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    return int(math.ceil(len(text) / chars_per_token)) if text else 0


def record_estimated_tokens(record: SFTRecord | DPORecord) -> int:
    """Estimate the number of model tokens a record consumes.

    SFT uses the rendered conversation. DPO counts the prompt plus both
    completions, because a preference step reads all three fields.
    """

    if isinstance(record, SFTRecord):
        return estimate_tokens(render_chat_template(record.messages))
    if isinstance(record, DPORecord):
        return (
            estimate_tokens(record.prompt)
            + estimate_tokens(record.chosen)
            + estimate_tokens(record.rejected)
        )
    raise DatasetError(f"unsupported record type: {type(record).__name__}")


@dataclass(frozen=True)
class DatasetStats:
    """Aggregate character and estimated-token statistics for a dataset."""

    kind: str
    records: int
    messages: int
    characters: int
    estimated_tokens_total: int
    mean_tokens: float
    median_tokens: float
    p95_tokens: float
    max_tokens: int
    roles: Mapping[str, int]
    chars_per_token: float = CHARS_PER_TOKEN

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "records": self.records,
            "messages": self.messages,
            "characters": self.characters,
            "estimated_tokens_total": self.estimated_tokens_total,
            "mean_tokens": round(self.mean_tokens, 2),
            "median_tokens": round(self.median_tokens, 2),
            "p95_tokens": round(self.p95_tokens, 2),
            "max_tokens": self.max_tokens,
            "roles": dict(sorted(self.roles.items())),
            "chars_per_token": self.chars_per_token,
            "token_estimate": "characters / 4 (estimate, not tokenizer output)",
        }


def _p95(values: Sequence[int]) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return float(ordered[index])


def dataset_stats(
    records: Sequence[SFTRecord] | Sequence[DPORecord],
) -> DatasetStats:
    """Compute record counts, role counts, characters, and estimated tokens."""

    items = list(records)
    if not items:
        raise DatasetError("dataset contains no records")
    first = items[0]
    if isinstance(first, SFTRecord):
        kind = "sft"
        if not all(isinstance(item, SFTRecord) for item in items):
            raise DatasetError("cannot mix SFT and DPO records in one dataset")
        roles: dict[str, int] = {}
        characters = 0
        messages = 0
        for record in items:
            for message in record.messages:
                roles[message.role] = roles.get(message.role, 0) + 1
                characters += len(message.content)
            messages += len(record.messages)
    elif isinstance(first, DPORecord):
        kind = "dpo"
        if not all(isinstance(item, DPORecord) for item in items):
            raise DatasetError("cannot mix SFT and DPO records in one dataset")
        characters = sum(
            len(record.prompt) + len(record.chosen) + len(record.rejected)
            for record in items
        )
        messages = 3 * len(items)
        roles = {"prompt": len(items), "chosen": len(items), "rejected": len(items)}
    else:
        raise DatasetError(
            f"unsupported record type: {type(first).__name__}; "
            "expected SFTRecord or DPORecord"
        )

    token_counts = [record_estimated_tokens(record) for record in items]
    return DatasetStats(
        kind=kind,
        records=len(items),
        messages=messages,
        characters=characters,
        estimated_tokens_total=sum(token_counts),
        mean_tokens=statistics.fmean(token_counts),
        median_tokens=float(statistics.median(token_counts)),
        p95_tokens=_p95(token_counts),
        max_tokens=max(token_counts),
        roles=roles,
    )


def split_records(
    records: Sequence[SFTRecord] | Sequence[DPORecord],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Split:
    """Shuffle and split records with a locally seeded RNG.

    ``val_ratio`` of the records are held out. When the ratio is positive and
    there are at least two records, the validation split gets at least one
    record, and the training split is never emptied.
    """

    if not 0.0 <= val_ratio < 1.0:
        raise DatasetError("val_ratio must be in the range [0.0, 1.0)")
    items = list(records)
    if not items:
        return Split(train=(), val=())
    indices = list(range(len(items)))
    random.Random(seed).shuffle(indices)
    val_count = int(round(len(items) * val_ratio))
    if val_ratio > 0.0 and len(items) >= 2:
        val_count = max(1, val_count)
    val_count = min(val_count, len(items) - 1) if len(items) >= 2 else 0
    val = tuple(items[index] for index in indices[:val_count])
    train = tuple(items[index] for index in indices[val_count:])
    return Split(train=train, val=val)


def dataset_fingerprint(
    records: Sequence[SFTRecord] | Sequence[DPORecord],
) -> str:
    """Return a stable SHA-256 fingerprint of the canonical records."""

    payload = [record.to_dict() for record in records]
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Write mappings as JSONL; returns the number of rows written."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count
