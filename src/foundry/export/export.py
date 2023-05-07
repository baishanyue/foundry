"""Adapter metadata finalization and model-card generation.

The exporter does not invent training results. It merges the declared LoRA
configuration, the training declaration, and measured dataset statistics into a
stable JSON manifest, then renders a Markdown model card from that manifest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from ..data.dataset import DatasetStats
from ..adapter.lora import LoRAConfig, LoRAPlan, plan_adapter

ADAPTER_FORMAT = "foundry.adapter"
ADAPTER_VERSION = 1


def _format_number(value: int | float) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    number = float(value)
    if number.is_integer():
        return f"{int(number):,}"
    if 0 < abs(number) < 0.01:
        return f"{number:.6g}"
    return f"{number:,.2f}"


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return _format_number(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, Mapping):
        return json.dumps(dict(value), sort_keys=True)
    return str(value)


@dataclass(frozen=True)
class AdapterMetadata:
    """Inputs required to finalize an adapter manifest.

    ``training`` is a declared configuration (for example epochs, batch size,
    learning rate). It is copied verbatim into the manifest; no metrics are
    added unless the caller supplies them.
    """

    name: str
    base_model: str
    lora: LoRAConfig
    dataset: DatasetStats
    training: Mapping[str, Any] = field(default_factory=dict)
    module_shapes: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    dataset_files: tuple[str, ...] = ()
    dataset_fingerprint: str | None = None
    created: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("adapter name must be a non-empty string")
        if not isinstance(self.base_model, str) or not self.base_model.strip():
            raise ValueError("base_model must be a non-empty string")
        if not isinstance(self.lora, LoRAConfig):
            raise ValueError("lora must be a LoRAConfig instance")
        if not isinstance(self.dataset, DatasetStats):
            raise ValueError("dataset must be a DatasetStats instance")
        if not isinstance(self.training, Mapping):
            raise ValueError("training must be a mapping")
        if not isinstance(self.module_shapes, Mapping):
            raise ValueError("module_shapes must be a mapping")
        if self.created is not None:
            try:
                date.fromisoformat(self.created)
            except (TypeError, ValueError) as exc:
                raise ValueError("created must be an ISO date (YYYY-MM-DD)") from exc
        files = self.dataset_files
        if isinstance(files, str):
            files = (files,)
        if not isinstance(files, (tuple, list)) or any(
            not isinstance(item, str) or not item for item in files
        ):
            raise ValueError("dataset_files must be a sequence of non-empty strings")
        notes = self.notes
        if isinstance(notes, str):
            notes = (notes,)
        if not isinstance(notes, (tuple, list)) or any(
            not isinstance(item, str) for item in notes
        ):
            raise ValueError("notes must be a sequence of strings")
        object.__setattr__(self, "dataset_files", tuple(files))
        object.__setattr__(self, "notes", tuple(notes))
        object.__setattr__(self, "training", dict(self.training))
        object.__setattr__(self, "module_shapes", dict(self.module_shapes))

    def plan(self) -> LoRAPlan | None:
        """Parameter plan when module shapes are available, else ``None``."""

        if not self.module_shapes:
            return None
        return plan_adapter(self.lora, self.module_shapes)

    def finalize(self) -> dict[str, Any]:
        """Merge configs into a canonical, fingerprinted manifest."""

        plan = self.plan()
        method: dict[str, Any] = {
            "type": "lora",
            "r": self.lora.r,
            "alpha": float(self.lora.alpha),
            "scaling": self.lora.scaling,
            "dropout": float(self.lora.dropout),
            "target_modules": list(self.lora.target_modules),
        }
        if plan is not None:
            method.update(
                {
                    "trainable_parameters": plan.total_trainable,
                    "targeted_base_parameters": plan.total_base,
                    "overhead_ratio": round(plan.overhead_ratio, 8),
                    "modules": {
                        module.name: module.as_dict() for module in plan.modules
                    },
                }
            )
        dataset = self.dataset.to_dict()
        dataset["files"] = list(self.dataset_files)
        if self.dataset_fingerprint is not None:
            if not isinstance(self.dataset_fingerprint, str) or not self.dataset_fingerprint:
                raise ValueError("dataset_fingerprint must be a non-empty string")
            dataset["fingerprint"] = self.dataset_fingerprint
        payload: dict[str, Any] = {
            "format": ADAPTER_FORMAT,
            "version": ADAPTER_VERSION,
            "name": self.name,
            "base_model": self.base_model,
            "method": method,
            "dataset": dataset,
            "training": dict(self.training),
        }
        if self.created is not None:
            payload["created"] = self.created
        if self.notes:
            payload["notes"] = list(self.notes)
        try:
            canonical = json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"adapter metadata is not JSON serializable: {exc}") from exc
        payload["fingerprint"] = "sha256:" + hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        return payload


def finalize_adapter_metadata(metadata: AdapterMetadata) -> dict[str, Any]:
    """Public wrapper around :meth:`AdapterMetadata.finalize`."""

    if not isinstance(metadata, AdapterMetadata):
        raise ValueError("metadata must be an AdapterMetadata instance")
    return metadata.finalize()


def write_adapter_metadata(metadata: AdapterMetadata, path: str | Path) -> Path:
    """Write the finalized manifest as stable, sorted JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = finalize_adapter_metadata(metadata)
    target.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def _training_rows(training: Mapping[str, Any]) -> str:
    if not training:
        return "| _No training configuration supplied_ | |"
    rows = []
    for key in sorted(training):
        rows.append(f"| {key} | {_format_value(training[key])} |")
    return "\n".join(rows)


def render_model_card(metadata: AdapterMetadata) -> str:
    """Render a Markdown model card from finalized adapter metadata."""

    payload = finalize_adapter_metadata(metadata)
    method = payload["method"]
    dataset = payload["dataset"]
    plan = metadata.plan()
    data_path = (
        metadata.dataset_files[0]
        if metadata.dataset_files
        else "examples/sft.jsonl"
    )

    if plan is not None:
        parameter_line = (
            f"{_format_number(plan.total_trainable)} of "
            f"{_format_number(plan.total_base)} parameters in the targeted "
            f"projections ({plan.overhead_ratio * 100:.2f}%)"
        )
    else:
        parameter_line = "not estimated (no layer shapes supplied)"

    lines: list[str] = [
        f"# {payload['name']}",
        "",
        "Adapter card generated by **Foundry**. It describes a declared "
        "fine-tuning configuration and measured dataset statistics; it makes no "
        "claims about downstream quality.",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Base model | `{payload['base_model']}` |",
        f"| Method | LoRA, r={method['r']}, alpha={_format_number(method['alpha'])}, "
        f"scaling={method['scaling']:.4f} |",
        f"| Trainable parameters | {parameter_line} |",
        f"| Dataset | {_format_number(dataset['records'])} records, "
        f"~{_format_number(dataset['estimated_tokens_total'])} estimated tokens |",
    ]
    if payload.get("created"):
        lines.append(f"| Created | {payload['created']} |")
    lines.append(f"| Manifest fingerprint | `{payload['fingerprint']}` |")
    lines.extend(
        [
            "",
            "## Method",
            "",
            f"Low-rank adaptation with rank {method['r']}, alpha "
            f"{_format_number(method['alpha'])}, and dropout "
            f"{method['dropout']}. The update applied at merge time is scaled by "
            f"`alpha / r = {method['scaling']:.4f}`.",
            "",
            "| Parameter | Value |",
            "| --- | --- |",
            f"| Target modules | {_format_value(method['target_modules'])} |",
            f"| Rank | {method['r']} |",
            f"| Alpha | {_format_number(method['alpha'])} |",
            f"| Dropout | {method['dropout']} |",
            f"| Scaling (alpha / r) | {method['scaling']:.4f} |",
        ]
    )
    if plan is not None:
        lines.extend(
            [
                f"| Trainable parameters | {_format_number(plan.total_trainable)} |",
                f"| Targeted base parameters | {_format_number(plan.total_base)} |",
                f"| Adapter share | {plan.overhead_ratio * 100:.2f}% |",
                "",
                "| Module | In | Out | Trainable | Base |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for module in plan.modules:
            lines.append(
                f"| `{module.name}` | {_format_number(module.in_features)} | "
                f"{_format_number(module.out_features)} | "
                f"{_format_number(module.trainable)} | {_format_number(module.base)} |"
            )
    lines.extend(
        [
            "",
            "## Dataset",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Format | {dataset['kind'].upper()} |",
            f"| Records | {_format_number(dataset['records'])} |",
            f"| Turns / fields | {_format_number(dataset['messages'])} |",
            f"| Characters | {_format_number(dataset['characters'])} |",
            f"| Estimated tokens | {_format_number(dataset['estimated_tokens_total'])} |",
            f"| Mean estimated tokens | {dataset['mean_tokens']:.2f} |",
            f"| Median estimated tokens | {dataset['median_tokens']:.2f} |",
            f"| p95 estimated tokens | {dataset['p95_tokens']:.2f} |",
            f"| Max estimated tokens | {_format_number(dataset['max_tokens'])} |",
            "",
            "Roles: " + _format_value(dataset["roles"]) + ".",
            "",
            "> Token counts are character-based estimates (`ceil(characters / 4)`), "
            "not tokenizer output.",
        ]
    )
    if metadata.dataset_files:
        lines.extend(
            [
                "",
                "Dataset files: " + ", ".join(f"`{name}`" for name in metadata.dataset_files),
            ]
        )
    if "fingerprint" in dataset:
        lines.extend(["", f"Dataset fingerprint: `{dataset['fingerprint']}`"])
    lines.extend(
        [
            "",
            "## Training configuration",
            "",
            "| Parameter | Value |",
            "| --- | --- |",
            _training_rows(payload["training"]),
            "",
            "## Intended use",
            "",
            "Research and engineering evaluation of parameter-efficient "
            "fine-tuning workflows on instruction-style data. The manifest is "
            "meant to travel with the adapter weights so runs stay auditable.",
            "",
            "## Limitations",
            "",
            "- This card contains metadata only; adapter weights are stored separately.",
            "- Token statistics are estimates and will not match a specific tokenizer.",
            "- No evaluation results are implied by the training configuration.",
            "- Dataset quality, licensing, and deduplication remain the operator's responsibility.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "# Inspect the dataset statistics used above",
            f"python -m foundry stats --data {data_path}",
            "",
            "# Regenerate this card",
            f"python -m foundry card --data {data_path} \\",
            f"  --base-model {payload['base_model']} --out model-card.md",
            "```",
            "",
        ]
    )
    if metadata.notes:
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in metadata.notes)
        lines.append("")
    return "\n".join(lines)
