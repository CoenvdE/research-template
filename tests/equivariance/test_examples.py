"""Examples of the harness. Replace with tests for your actual model's claimed
symmetries (SO(3), permutation, translation, ...).

Note the shape of every example: a positive assertion paired with a negative
control. The control is what makes the positive assertion mean something.
"""

import torch

from tests.equivariance.utils import (
    assert_equivariant,
    assert_invariant,
    assert_not_equivariant,
    assert_not_invariant,
    random_rotation,
)


def test_mean_pool_is_permutation_invariant():
    x = torch.randn(4, 10, 8)  # (batch, set, feat)
    perm = torch.randperm(10)
    assert_invariant(
        fn=lambda t: t.mean(dim=1),
        x=x,
        act_in=lambda t: t[:, perm],
        name="mean-pool",
    )


def test_first_element_is_not_permutation_invariant():
    """Negative control: proves the invariance check can fail."""
    x = torch.randn(4, 10, 8)
    perm = torch.randperm(10)
    assert_not_invariant(
        fn=lambda t: t[:, 0],
        x=x,
        act_in=lambda t: t[:, perm],
        name="take-first",
    )


def test_linear_scaling_is_rotation_equivariant():
    g = torch.Generator().manual_seed(0)
    rot = random_rotation(3, g)
    x = torch.randn(16, 3, generator=g)
    assert_equivariant(
        fn=lambda t: 2.0 * t,  # any isotropic scaling commutes with rotation
        x=x,
        act_in=lambda t: t @ rot.T,
        act_out=lambda t: t @ rot.T,
        name="scale-by-2",
    )


def test_anisotropic_scaling_is_not_rotation_equivariant():
    """Negative control: per-axis scaling picks out a frame, so it must fail."""
    g = torch.Generator().manual_seed(0)
    rot = random_rotation(3, g)
    x = torch.randn(16, 3, generator=g)
    weights = torch.tensor([1.0, 2.0, 3.0])
    assert_not_equivariant(
        fn=lambda t: t * weights,
        x=x,
        act_in=lambda t: t @ rot.T,
        act_out=lambda t: t @ rot.T,
        name="anisotropic-scale",
    )
