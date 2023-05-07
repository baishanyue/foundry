import pytest

from foundry.adapter.lora import (
    LoRAConfig,
    plan_adapter,
    trainable_parameters,
)


def test_default_config_scaling():
    config = LoRAConfig()
    assert config.r == 16
    assert config.alpha == 32
    assert config.dropout == pytest.approx(0.05)
    assert config.target_modules == ("q_proj", "v_proj")
    assert config.scaling == pytest.approx(2.0)
    payload = config.as_dict()
    assert payload["scaling"] == pytest.approx(2.0)
    assert payload["target_modules"] == ["q_proj", "v_proj"]


def test_trainable_parameter_formula():
    config = LoRAConfig(r=8, alpha=16)
    assert trainable_parameters(config, 4096, 4096) == 8 * (4096 + 4096)
    assert trainable_parameters(config, 4096, 1024) == 8 * (4096 + 1024)
    with pytest.raises(ValueError):
        trainable_parameters(config, 0, 4096)


def test_plan_adapter_accounting():
    config = LoRAConfig(r=16, alpha=32, target_modules=("q_proj", "v_proj"))
    plan = plan_adapter(config, {"q_proj": (4096, 4096), "v_proj": (4096, 1024)})
    assert plan.total_trainable == 16 * (4096 + 4096) + 16 * (4096 + 1024)
    assert plan.total_base == 4096 * 4096 + 4096 * 1024
    assert plan.overhead_ratio == pytest.approx(plan.total_trainable / plan.total_base)
    assert plan.scaling == pytest.approx(2.0)
    assert [module.name for module in plan.modules] == ["q_proj", "v_proj"]
    payload = plan.as_dict()
    assert payload["total_trainable_parameters"] == plan.total_trainable
    assert set(payload["modules"]) == {"q_proj", "v_proj"}


def test_plan_ignores_extra_shapes():
    config = LoRAConfig(target_modules=("q_proj",))
    plan = plan_adapter(
        config,
        {"q_proj": (64, 32), "k_proj": (64, 32), "embed": (100, 64)},
    )
    assert len(plan.modules) == 1
    assert plan.modules[0].trainable == config.r * (64 + 32)
    assert plan.modules[0].base == 64 * 32


def test_target_modules_are_normalized():
    config = LoRAConfig(target_modules=[" q_proj ", "q_proj", "v_proj"])
    assert config.target_modules == ("q_proj", "v_proj")
    single = LoRAConfig(target_modules="gate_proj")
    assert single.target_modules == ("gate_proj",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"r": 0},
        {"r": -4},
        {"r": 4.5},
        {"alpha": 0},
        {"alpha": -1.0},
        {"dropout": 1.0},
        {"dropout": -0.1},
        {"target_modules": ()},
        {"target_modules": ("",)},
    ],
)
def test_invalid_configs_raise(kwargs):
    with pytest.raises(ValueError):
        LoRAConfig(**kwargs)


def test_plan_requires_target_shapes():
    config = LoRAConfig(target_modules=("q_proj", "v_proj"))
    with pytest.raises(ValueError, match="missing shapes"):
        plan_adapter(config, {"q_proj": (8, 8)})
    with pytest.raises(ValueError, match="pair"):
        plan_adapter(config, {"q_proj": (8,), "v_proj": (8, 8)})
    with pytest.raises(ValueError, match="positive"):
        plan_adapter(config, {"q_proj": (0, 8), "v_proj": (8, 8)})
