"""The degrees-instead-of-radians class of bug, caught two ways: a static
transform check and the runtime InputStatsGuard."""

import math

import pytest
import torch

from src.callbacks import InputStatsGuard
from src.utils.checks import assert_radians


def test_radians_pass():
    assert_radians(torch.linspace(-math.pi, math.pi, 10))


def test_degrees_fail():
    with pytest.raises(ValueError, match="radians"):
        assert_radians(torch.tensor([45.0, 90.0, 180.0]), name="lat")


def test_guard_raises_on_big_numbers():
    guard = InputStatsGuard(absmax_limit=100.0, mode="raise")
    bad_batch = (torch.tensor([[1.0, 2.0, 1e4]]), torch.zeros(1, 1))
    with pytest.raises(ValueError, match="absmax"):
        guard.check_batch(bad_batch)


def test_guard_raises_on_nan():
    guard = InputStatsGuard(mode="raise")
    bad_batch = (torch.tensor([[1.0, float("nan")]]),)
    with pytest.raises(ValueError, match="NaN"):
        guard.check_batch(bad_batch)


def test_guard_passes_clean_batch():
    guard = InputStatsGuard(mode="raise")
    metrics = guard.check_batch((torch.randn(8, 4), torch.randn(8, 2)))
    assert any(k.endswith("_absmax") for k in metrics)
