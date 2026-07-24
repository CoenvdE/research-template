"""Generic equivariance test harness (generalizes geometric-weather's
_equivariance_utils). A function f is equivariant when f(g . x) == g . f(x) for
group actions g; invariance is the special case where the output action is the
identity."""

from typing import Callable

import torch


def assert_equivariant(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    act_in: Callable[[torch.Tensor], torch.Tensor],
    act_out: Callable[[torch.Tensor], torch.Tensor],
    atol: float = 1e-5,
    name: str = "fn",
):
    out_transformed_input = fn(act_in(x))
    transformed_output = act_out(fn(x))
    err = (out_transformed_input - transformed_output).abs().max().item()
    assert err <= atol, f"{name} not equivariant: max error {err:.3e} > {atol:.1e}"


def assert_invariant(fn, x, act_in, atol: float = 1e-5, name: str = "fn"):
    assert_equivariant(fn, x, act_in, act_out=lambda t: t, atol=atol, name=name)


def random_rotation(dim: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Uniform random rotation matrix via QR decomposition."""
    a = torch.randn(dim, dim, generator=generator)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:  # ensure proper rotation (det +1)
        q[:, 0] = -q[:, 0]
    return q
