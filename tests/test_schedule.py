import math

import pytest

from foundry.schedules.schedule import LRSchedule, warmup_cosine


def test_warmup_is_linear_and_peaks_at_base():
    schedule = LRSchedule(
        total_steps=10, base_lr=1.0, warmup_steps=4, min_lr_ratio=0.1
    )
    assert schedule(0) == pytest.approx(0.25)
    assert schedule(1) == pytest.approx(0.5)
    assert schedule(2) == pytest.approx(0.75)
    assert schedule(3) == pytest.approx(1.0)
    assert schedule(4) == pytest.approx(1.0)


def test_cosine_decays_to_floor():
    schedule = LRSchedule(
        total_steps=10, base_lr=1.0, warmup_steps=4, min_lr_ratio=0.1
    )
    values = schedule.values()
    assert len(values) == 11
    assert values[-1] == pytest.approx(0.1)
    assert values[4] >= values[5] >= values[6] >= values[10]
    assert values[5] < values[4]
    midpoint = schedule(7)
    assert 0.1 < midpoint < 1.0


def test_no_warmup_starts_at_peak():
    schedule = LRSchedule(total_steps=4, base_lr=2e-4, warmup_steps=0)
    assert schedule(0) == pytest.approx(2e-4)
    assert schedule(4) == pytest.approx(0.0)
    assert schedule(2) == pytest.approx(1e-4)


def test_steps_clamp_and_negative_rejected():
    schedule = LRSchedule(total_steps=10, base_lr=1.0, warmup_steps=2)
    assert schedule(999) == pytest.approx(schedule(10))
    with pytest.raises(ValueError):
        schedule(-1)
    with pytest.raises(ValueError):
        schedule(1.5)
    with pytest.raises(ValueError):
        schedule(True)


def test_sample_and_metadata():
    schedule = LRSchedule(
        total_steps=10, base_lr=1.0, warmup_steps=2, min_lr_ratio=0.05
    )
    samples = schedule.sample(5)
    assert [step for step, _ in samples] == [0, 2, 5, 8, 10]
    assert samples[0][1] == pytest.approx(schedule(0))
    payload = schedule.as_dict()
    assert payload["name"] == "linear-warmup-cosine"
    assert payload["min_lr"] == pytest.approx(0.05)
    with pytest.raises(ValueError):
        schedule.sample(1)
    with pytest.raises(ValueError):
        schedule.sample(12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_steps": 0, "base_lr": 1.0},
        {"total_steps": 10, "base_lr": 0.0},
        {"total_steps": 10, "base_lr": -1.0},
        {"total_steps": 10, "base_lr": 1.0, "warmup_steps": 11},
        {"total_steps": 10, "base_lr": 1.0, "warmup_steps": -1},
        {"total_steps": 10, "base_lr": 1.0, "min_lr_ratio": 1.5},
        {"total_steps": 10, "base_lr": 1.0, "min_lr_ratio": -0.1},
    ],
)
def test_invalid_schedules_raise(kwargs):
    with pytest.raises(ValueError):
        LRSchedule(**kwargs)


def test_wrapper_matches_class():
    for step in (0, 3, 10, 25):
        assert warmup_cosine(
            step,
            total_steps=20,
            base_lr=3e-4,
            warmup_steps=5,
            min_lr_ratio=0.1,
        ) == pytest.approx(
            LRSchedule(
                total_steps=20,
                base_lr=3e-4,
                warmup_steps=5,
                min_lr_ratio=0.1,
            )(step)
        )


def test_full_warmup_edge_case():
    schedule = LRSchedule(total_steps=5, base_lr=1.0, warmup_steps=5)
    assert schedule(4) == pytest.approx(1.0)
    assert schedule(5) == pytest.approx(1.0)
    assert all(math.isfinite(value) for value in schedule.values())
