import math

import pytest

from foundry.objective.losses import (
    cross_entropy,
    dpo_loss,
    dpo_report,
    log_sigmoid,
    log_softmax,
    masked_cross_entropy,
    sigmoid,
)


def test_log_softmax_uniform():
    values = log_softmax([0.0, 0.0, 0.0])
    assert values == pytest.approx([-math.log(3.0)] * 3)
    assert sum(math.exp(value) for value in values) == pytest.approx(1.0)


def test_log_softmax_is_stable_for_large_logits():
    values = log_softmax([1000.0, 0.0, -1000.0])
    assert all(math.isfinite(value) for value in values)
    assert sum(math.exp(value) for value in values) == pytest.approx(1.0)
    assert values[0] == pytest.approx(0.0, abs=1e-12)
    assert cross_entropy([1000.0, 0.0, -1000.0], 2) == pytest.approx(2000.0)
    with pytest.raises(ValueError):
        log_softmax([])
    with pytest.raises(ValueError):
        log_softmax([0.0, math.inf])


def test_cross_entropy_validation():
    assert cross_entropy([0.0, 0.0], 0) == pytest.approx(math.log(2.0))
    with pytest.raises(ValueError):
        cross_entropy([0.0, 0.0], 2)
    with pytest.raises(ValueError):
        cross_entropy([0.0, 0.0], -1)
    with pytest.raises(ValueError):
        cross_entropy([0.0, 0.0], True)


def test_masked_cross_entropy_averages_only_selected_tokens():
    logits = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    targets = [0, 1, 1]
    result = masked_cross_entropy(logits, targets, [True, False, True])
    assert result == pytest.approx(math.log(2.0))
    changed = [[0.0, 0.0], [99.0, -99.0], [0.0, 0.0]]
    assert masked_cross_entropy(changed, targets, [1, 0, 1]) == pytest.approx(
        math.log(2.0)
    )


def test_masked_cross_entropy_validation():
    with pytest.raises(ValueError, match="no positions"):
        masked_cross_entropy([[0.0]], [0], [False])
    with pytest.raises(ValueError, match="same length"):
        masked_cross_entropy([[0.0]], [0, 0], [True])
    with pytest.raises(ValueError, match="logits"):
        masked_cross_entropy([], [], [])
    with pytest.raises(ValueError, match="bool or 0/1"):
        masked_cross_entropy([[0.0]], [0], [2])


def test_sigmoid_and_log_sigmoid_are_stable():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(1000.0) == 1.0
    assert sigmoid(-1000.0) == pytest.approx(0.0)
    assert log_sigmoid(0.0) == pytest.approx(-math.log(2.0))
    assert log_sigmoid(1000.0) == pytest.approx(0.0, abs=1e-12)
    assert log_sigmoid(-1000.0) == pytest.approx(-1000.0)
    assert math.isfinite(log_sigmoid(-500.0))
    with pytest.raises(ValueError):
        sigmoid(math.nan)
    with pytest.raises(ValueError):
        log_sigmoid(math.inf)


def test_dpo_loss_zero_margin_is_log_two():
    loss = dpo_loss(-1.0, -2.0, -1.5, -2.5)
    assert loss == pytest.approx(math.log(2.0))


def test_dpo_loss_responds_to_margin():
    good = dpo_loss(-0.1, -10.0, -5.0, -5.0, beta=0.5)
    bad = dpo_loss(-10.0, -0.1, -5.0, -5.0, beta=0.5)
    assert 0.0 <= good < math.log(2.0) < bad
    margin = 0.5 * ((-0.1 + 10.0) - 0.0)
    assert good == pytest.approx(-math.log(sigmoid(margin)))


def test_dpo_loss_validation():
    with pytest.raises(ValueError):
        dpo_loss(-1.0, -2.0, -1.0, -2.0, beta=0.0)
    with pytest.raises(ValueError):
        dpo_loss(-1.0, math.nan, -1.0, -2.0)
    with pytest.raises(ValueError):
        dpo_loss(1.0, -2.0, -1.0, -2.0)


def test_dpo_report_metrics():
    report = dpo_report(-0.1, -10.0, -5.0, -5.0, beta=0.5)
    assert report.preference_accuracy == 1.0
    assert report.chosen_reward > report.rejected_reward
    assert report.margin > 0
    payload = report.as_dict()
    assert payload["preference_accuracy"] == 1.0

    tied = dpo_report(-1.0, -2.0, -1.5, -2.5)
    assert tied.preference_accuracy == 0.5
    assert tied.margin == pytest.approx(0.0)

    wrong = dpo_report(-10.0, -0.1, -5.0, -5.0, beta=0.5)
    assert wrong.preference_accuracy == 0.0
