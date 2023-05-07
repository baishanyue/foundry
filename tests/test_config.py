import json

import pytest

from foundry.config import ConfigError, FoundryConfig


@pytest.mark.parametrize("name,r,alpha,steps", [
    ("sft-default", 8, 16.0, 1000),
    ("lora-small", 4, 8.0, 600),
])
def test_shipped_recipes_load(name, r, alpha, steps):
    config = FoundryConfig.load(f"configs/{name}.json")
    assert config.lora_r == r
    assert config.lora_alpha == alpha
    assert config.steps == steps
    assert config.to_lora().r == r
    assert config.to_schedule().total_steps == steps


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"dataset": "x.jsonl", "learning_rate": 1e-4}), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown config keys"):
        FoundryConfig.load(path)


def test_invalid_json_is_rejected(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid JSON"):
        FoundryConfig.load(path)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="cannot read config"):
        FoundryConfig.load(tmp_path / "absent.json")


def test_round_trip():
    config = FoundryConfig.load("configs/lora-small.json")
    assert FoundryConfig.from_dict(config.to_dict()) == config
