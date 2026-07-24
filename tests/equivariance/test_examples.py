"""Examples of the harness. Replace with tests for your actual model's claimed
symmetries (SO(3), permutation, translation, ...)."""

import torch

from tests.equivariance.utils import assert_equivariant, assert_invariant, random_rotation


def test_mean_pool_is_permutation_invariant():
    x = torch.randn(4, 10, 8)  # (batch, set, feat)
    perm = torch.randperm(10)
    assert_invariant(
        fn=lambda t: t.mean(dim=1),
        x=x,
        act_in=lambda t: t[:, perm],
        name="mean-pool",
    )


def test_linear_scaling_is_rotation_equivariant():
    g = torch.Generator().manual_seed(0)
    rot = random_rotation(3, g)
    x = torch.randn(16, 3, generator=g)
    assert_equivariant(
        fn=lambda t: 2.0 * t,           # any pointwise linear map commutes with rotation
        x=x,
        act_in=lambda t: t @ rot.T,
        act_out=lambda t: t @ rot.T,
        name="scale-by-2",
    )
