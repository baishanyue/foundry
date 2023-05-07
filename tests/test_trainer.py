import json

import pytest

from foundry.pipeline.trainer import (
    SyntheticModel,
    Trainer,
    TrainingConfig,
    TrainableModel,
)

DATA = ((1, 2, 3, 4), (1, 3, 2, 4), (2, 3, 4, 5), (1, 2, 4, 5))


def make_config(**overrides):
    base = dict(
        epochs=2,
        batch_size=2,
        grad_accum_steps=1,
        lr=0.2,
        warmup_steps=2,
        min_lr_ratio=0.1,
        max_grad_norm=2.0,
        shuffle=True,
        seed=0,
    )
    base.update(overrides)
    return TrainingConfig(**base)


def test_synthetic_model_is_deterministic_and_serializable():
    first = SyntheticModel(vocab_size=8, seed=3)
    second = SyntheticModel(vocab_size=8, seed=3)
    assert first.weights == second.weights
    assert first.num_parameters == 64
    assert isinstance(first, TrainableModel)
    state = first.state_dict()
    clone = SyntheticModel(vocab_size=8, seed=99)
    clone.load_state_dict(state)
    assert clone.weights == first.weights
    assert clone.gradient_norm() == 0.0


def test_gradient_accumulation_matches_full_batch():
    full = SyntheticModel(vocab_size=8, seed=5)
    loss_full, tokens_full = full.accumulate(DATA[:2])
    expected = [row[:] for row in full._gradients]

    split = SyntheticModel(vocab_size=8, seed=5)
    loss_one, tokens_one = split.accumulate(DATA[:1])
    loss_two, tokens_two = split.accumulate(DATA[1:2])
    assert tokens_one + tokens_two == tokens_full
    split.scale_gradients(1.0 / (tokens_one + tokens_two))
    for actual_row, expected_row in zip(split._gradients, expected):
        for actual, target in zip(actual_row, expected_row):
            assert actual == pytest.approx(target / tokens_full, abs=1e-12)
    weighted = (loss_one * tokens_one + loss_two * tokens_two) / tokens_full
    assert weighted == pytest.approx(loss_full)


def test_training_reduces_loss_and_records_history():
    trainer = Trainer(
        SyntheticModel(vocab_size=8, seed=2),
        make_config(epochs=40, lr=0.5, warmup_steps=4, max_grad_norm=5.0),
    )
    before = trainer.evaluate(DATA)
    history = trainer.fit(DATA)
    after = trainer.evaluate(DATA)
    assert len(history.records) == 80
    assert trainer.step == 80
    assert history.losses[-1] < history.losses[0]
    assert after < before
    assert [record.step for record in history.records] == list(range(80))
    assert history.records[0].lr == pytest.approx(0.5 / 4)
    assert history.records[1].lr == pytest.approx(0.5 / 2)
    assert history.records[4].lr == pytest.approx(0.5)
    assert all(record.tokens > 0 for record in history.records)
    assert len(history.micro_losses) == 80


def test_gradient_clipping_caps_norm():
    trainer = Trainer(
        SyntheticModel(vocab_size=8, seed=4),
        make_config(
            epochs=1,
            batch_size=4,
            lr=0.1,
            warmup_steps=0,
            max_grad_norm=0.01,
            shuffle=False,
        ),
    )
    history = trainer.fit(DATA)
    record = history.records[0]
    assert record.grad_norm > 0.01
    assert record.clipped_norm == pytest.approx(0.01)
    assert trainer.model.gradient_norm() == 0.0


def test_without_clipping_norms_match():
    trainer = Trainer(
        SyntheticModel(vocab_size=8, seed=4),
        make_config(max_grad_norm=None, epochs=1, warmup_steps=0),
    )
    record = trainer.fit(DATA).records[0]
    assert record.clipped_norm == pytest.approx(record.grad_norm)


def test_gradient_accumulation_steps_change_update_count():
    trainer = Trainer(
        SyntheticModel(vocab_size=8, seed=1),
        make_config(epochs=3, batch_size=1, grad_accum_steps=2),
    )
    history = trainer.fit(DATA)
    assert len(history.records) == 6
    assert len(history.micro_losses) == 12


def test_checkpoint_roundtrip(tmp_path):
    config = make_config(epochs=3, batch_size=2)
    trainer = Trainer(SyntheticModel(vocab_size=8, seed=7), config)
    trainer.fit(DATA)
    path = trainer.save_checkpoint(tmp_path / "checkpoints" / "step.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "foundry.checkpoint"
    assert payload["step"] == trainer.step

    restored_model = SyntheticModel(vocab_size=8, seed=123)
    restored = Trainer.load_checkpoint(path, restored_model)
    assert restored.step == trainer.step
    assert restored.config == config
    assert restored_model.weights == trainer.model.weights
    assert restored.history.losses == trainer.history.losses
    assert restored.history.micro_losses == trainer.history.micro_losses


def test_checkpoint_validation(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        Trainer.load_checkpoint(tmp_path / "missing.json", SyntheticModel(8))
    broken = tmp_path / "broken.json"
    broken.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="format"):
        Trainer.load_checkpoint(broken, SyntheticModel(8))
    model = SyntheticModel(8)
    assert model.vocab_size == 8


def test_training_is_reproducible():
    config = make_config(epochs=4)
    first = Trainer(SyntheticModel(vocab_size=8, seed=11), config)
    second = Trainer(SyntheticModel(vocab_size=8, seed=11), config)
    first.fit(DATA)
    second.fit(DATA)
    assert first.history.records == second.history.records
    assert first.model.weights == second.model.weights


def test_validation_errors():
    with pytest.raises(ValueError):
        SyntheticModel(vocab_size=1)
    with pytest.raises(ValueError):
        SyntheticModel(vocab_size=8, seed=True)
    model = SyntheticModel(8)
    with pytest.raises(ValueError, match="at least two tokens"):
        model.accumulate([[1]])
    with pytest.raises(ValueError, match="outside the vocabulary"):
        model.accumulate([[1, 99]])
    with pytest.raises(ValueError, match="no next-token targets"):
        model.accumulate([])
    with pytest.raises(ValueError):
        model.scale_gradients(-1.0)
    with pytest.raises(ValueError):
        model.apply_gradients(-0.1)

    trainer = Trainer(SyntheticModel(8), make_config())
    with pytest.raises(ValueError):
        trainer.fit([])
    with pytest.raises(ValueError):
        trainer.evaluate([])


@pytest.mark.parametrize(
    "overrides",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"grad_accum_steps": 0},
        {"lr": 0.0},
        {"warmup_steps": -1},
        {"min_lr_ratio": 2.0},
        {"max_grad_norm": 0.0},
        {"seed": 1.5},
    ],
)
def test_invalid_training_config(overrides):
    with pytest.raises(ValueError):
        make_config(**overrides)


def test_config_as_dict_is_json_safe():
    config = make_config()
    payload = config.as_dict()
    assert payload["max_grad_norm"] == 2.0
    assert json.loads(json.dumps(payload)) == payload
