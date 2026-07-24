"""Data-contract checks: the degrees-instead-of-radians class of bug and its
physics-simulation cousins (raw Pascals where standardized values were expected,
out-of-bounds concentrations), caught statically and by the runtime guard."""

import math

import pytest
import torch

from src.callbacks import InputStatsGuard
from src.utils.checks import (
    assert_absmax,
    assert_in_range,
    assert_radians,
    assert_standardized,
)


def test_radians_pass():
    assert_radians(torch.linspace(-math.pi, math.pi, 10))


def test_standardized_field_passes():
    torch.manual_seed(0)
    assert_standardized(torch.randn(1024), name="u_velocity")


def test_raw_pascals_fail_standardized_check():
    # Sea-level pressure fed raw (~1e5 Pa) instead of standardized.
    raw_pressure = 101325.0 + 500.0 * torch.randn(1024)
    with pytest.raises(ValueError, match="standardized"):
        assert_standardized(raw_pressure, name="mslp")


def test_absmax_passes_normalized():
    torch.manual_seed(0)
    assert_absmax(torch.randn(1024), limit=100.0, name="normalized_field")


def test_absmax_catches_gradient_bombs():
    # Unnormalized elevation in meters entering a model expecting ~N(0, 1).
    elevation_m = torch.tensor([0.0, 1250.0, 8848.0])
    with pytest.raises(ValueError, match="destabilize gradients"):
        assert_absmax(elevation_m, limit=100.0, name="elevation")


def test_concentration_in_bounds():
    assert_in_range(torch.rand(256), 0.0, 1.0, name="sea_ice_concentration")


def test_kelvin_vs_celsius_caught():
    # Surface temperature in Celsius sneaking into a pipeline expecting Kelvin.
    celsius = torch.tensor([15.0, 22.5, -5.0])
    with pytest.raises(ValueError, match="physical range"):
        assert_in_range(celsius, 180.0, 340.0, name="t2m_kelvin")


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
