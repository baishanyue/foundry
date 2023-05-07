import json
from pathlib import Path

import pytest

from foundry.data.dataset import dataset_stats, load_sft
from foundry.export.export import (
    AdapterMetadata,
    finalize_adapter_metadata,
    render_model_card,
    write_adapter_metadata,
)
from foundry.adapter.lora import LoRAConfig

ROOT = Path(__file__).resolve().parents[1]
SFT_PATH = ROOT / "examples" / "sft.jsonl"


def make_metadata(**overrides):
    records = load_sft(SFT_PATH)
    base = dict(
        name="foundry-adapter",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        lora=LoRAConfig(r=16, alpha=32, target_modules=("q_proj", "v_proj")),
        dataset=dataset_stats(records),
        training={
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 0.0002,
            "schedule": "linear warmup + cosine decay",
        },
        module_shapes={"q_proj": (4096, 4096), "v_proj": (4096, 4096)},
        dataset_files=("examples/sft.jsonl",),
        dataset_fingerprint="a" * 64,
        created="2026-11-06",
        notes=("Synthetic example metadata.",),
    )
    base.update(overrides)
    return AdapterMetadata(**base)


def test_finalize_manifest_merges_configs():
    metadata = make_metadata()
    payload = finalize_adapter_metadata(metadata)
    assert payload["format"] == "foundry.adapter"
    assert payload["version"] == 1
    assert payload["base_model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert payload["created"] == "2026-11-06"
    assert payload["dataset"]["records"] == 8
    assert payload["dataset"]["files"] == ["examples/sft.jsonl"]
    assert payload["dataset"]["fingerprint"] == "a" * 64
    assert payload["method"]["type"] == "lora"
    assert payload["method"]["scaling"] == pytest.approx(2.0)
    assert payload["method"]["trainable_parameters"] == 16 * (4096 + 4096) * 2
    assert payload["method"]["overhead_ratio"] == pytest.approx(0.0078125)
    assert payload["training"]["epochs"] == 3
    assert payload["notes"] == ["Synthetic example metadata."]
    assert payload["fingerprint"].startswith("sha256:")
    assert len(payload["fingerprint"]) == len("sha256:") + 64
    assert payload["fingerprint"] == make_metadata().finalize()["fingerprint"]


def test_manifest_without_shapes_omits_parameter_estimates():
    metadata = make_metadata(module_shapes={})
    payload = metadata.finalize()
    assert "trainable_parameters" not in payload["method"]
    assert metadata.plan() is None
    assert "not estimated" in render_model_card(metadata)


def test_write_manifest(tmp_path):
    path = write_adapter_metadata(make_metadata(), tmp_path / "nested" / "adapter.json")
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["name"] == "foundry-adapter"
    assert payload["fingerprint"].startswith("sha256:")


def test_model_card_contains_required_sections():
    card = render_model_card(make_metadata())
    assert card.startswith("# foundry-adapter")
    assert "Qwen/Qwen2.5-7B-Instruct" in card
    assert "LoRA" in card
    assert "## Method" in card
    assert "## Dataset" in card
    assert "## Training configuration" in card
    assert "## Intended use" in card
    assert "## Limitations" in card
    assert "## Reproduction" in card
    assert "character-based estimates" in card
    assert "`q_proj`" in card


def test_metadata_validation():
    with pytest.raises(ValueError, match="name"):
        make_metadata(name="  ")
    with pytest.raises(ValueError, match="base_model"):
        make_metadata(base_model="")
    with pytest.raises(ValueError, match="lora"):
        make_metadata(lora="r=16")
    with pytest.raises(ValueError, match="dataset"):
        make_metadata(dataset={})
    with pytest.raises(ValueError, match="created"):
        make_metadata(created="November 6")
    with pytest.raises(ValueError, match="dataset_files"):
        make_metadata(dataset_files=("",))
    with pytest.raises(ValueError, match="JSON serializable"):
        make_metadata(training={"bad": object()}).finalize()


def test_finalize_rejects_wrong_type():
    with pytest.raises(ValueError, match="AdapterMetadata"):
        finalize_adapter_metadata({"name": "x"})
