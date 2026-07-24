"""Data-contract checks: one assert per convention your data must satisfy.

Use them inside transforms/datamodules (fail at load time, not after an hour of
GPU) and in tests. `assert_radians` is one instance of the pattern; add one
checker per physical convention your project relies on.
"""

import math

import torch


def assert_finite(t: torch.Tensor, name: str = "tensor") -> None:
    """No NaN/Inf anywhere."""
    bad = int(torch.isnan(t).sum() + torch.isinf(t).sum())
    if bad:
        raise ValueError(f"{name}: {bad} NaN/Inf values out of {t.numel()}")


def assert_radians(t: torch.Tensor, name: str = "coords") -> None:
    """Fail loudly if angular coordinates look like degrees instead of radians."""
    absmax = t.abs().max().item()
    if absmax > math.pi + 1e-6:
        raise ValueError(
            f"{name}: |max| = {absmax:.4g} > pi. Angular inputs must be radians; "
            "this looks like degrees."
        )


def assert_in_range(
    t: torch.Tensor, lo: float, hi: float, name: str = "tensor"
) -> None:
    """Hard physical bounds for raw (pre-normalization) quantities.

    Examples: fractions/concentrations in [0, 1]; surface temperature in
    Kelvin in [180, 340]; salinity in [0, 45] psu. Violations mean wrong units,
    a corrupt shard, or a bad transform.
    """
    tmin, tmax = t.min().item(), t.max().item()
    if tmin < lo or tmax > hi:
        raise ValueError(
            f"{name}: values in [{tmin:.4g}, {tmax:.4g}] outside physical range "
            f"[{lo:.4g}, {hi:.4g}]: check units and preprocessing"
        )


def assert_standardized(
    t: torch.Tensor,
    name: str = "tensor",
    mean_tol: float = 0.5,
    std_range: tuple[float, float] = (0.3, 3.0),
) -> None:
    """Check a field was actually standardized (~zero mean, ~unit std).

    Catches normalization silently not applied: e.g. pressure fed in raw Pascals
    (~1e5) instead of standardized, or stats computed on the wrong variable.
    Tolerances are loose on purpose: single batches wobble around the dataset
    statistics.
    """
    assert_finite(t, name)
    mean, std = t.float().mean().item(), t.float().std().item()
    if abs(mean) > mean_tol or not (std_range[0] <= std <= std_range[1]):
        raise ValueError(
            f"{name}: mean={mean:.4g}, std={std:.4g}: does not look standardized "
            f"(expected |mean| <= {mean_tol}, std in {std_range}). "
            "Was normalization applied, with the right statistics?"
        )


def find_unused_parameters(pl_module, batch) -> list[str]:
    """Run one training step + backward; return parameters that received no grad.

    Catches disconnected graph parts (a head that's never called, a frozen block
    you forgot about). Anything returned should either be fixed or explicitly
    excluded.
    """
    pl_module.zero_grad()
    loss = pl_module.training_step(batch, 0)
    if isinstance(loss, dict):
        loss = loss["loss"]
    loss.backward()
    unused = [
        n for n, p in pl_module.named_parameters() if p.requires_grad and p.grad is None
    ]
    pl_module.zero_grad()
    return unused
