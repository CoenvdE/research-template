"""Generic equivariance test harness. A function f is equivariant when
f(g . x) == g . f(x) for group actions g; invariance is the special case where
the output action is the identity.

Every positive check here has a negative counterpart, and that pairing is the
point: a function that ignored its inputs entirely would pass any equivariance
assertion trivially. Pair `assert_equivariant` with `assert_not_equivariant`
under a transformation the model must NOT be equivariant to (a random rotation
when only a discrete subgroup is claimed, a non-group permutation), so the test
is proven able to fail.
"""

from typing import Callable

import torch


def equivariance_error(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    act_in: Callable[[torch.Tensor], torch.Tensor],
    act_out: Callable[[torch.Tensor], torch.Tensor],
) -> float:
    """Max |f(g.x) - g.f(x)|. Zero for a perfectly equivariant f."""
    return (fn(act_in(x)) - act_out(fn(x))).abs().max().item()


def assert_equivariant(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    act_in: Callable[[torch.Tensor], torch.Tensor],
    act_out: Callable[[torch.Tensor], torch.Tensor],
    atol: float = 1e-5,
    name: str = "fn",
):
    err = equivariance_error(fn, x, act_in, act_out)
    assert err <= atol, f"{name} not equivariant: max error {err:.3e} > {atol:.1e}"


def assert_invariant(fn, x, act_in, atol: float = 1e-5, name: str = "fn"):
    assert_equivariant(fn, x, act_in, act_out=lambda t: t, atol=atol, name=name)


def assert_not_equivariant(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    act_in: Callable[[torch.Tensor], torch.Tensor],
    act_out: Callable[[torch.Tensor], torch.Tensor],
    atol: float = 1e-5,
    name: str = "fn",
):
    """Negative control: the property must NOT hold here.

    Failing this means the positive test proves nothing, because the assertion
    passes for transformations the model never claimed to respect.
    """
    err = equivariance_error(fn, x, act_in, act_out)
    assert err > atol, (
        f"{name} appears equivariant to a transformation it should not be "
        f"(error {err:.3e} <= {atol:.1e}): the positive test cannot discriminate, "
        "so it proves nothing"
    )


def assert_not_invariant(fn, x, act_in, atol: float = 1e-5, name: str = "fn"):
    assert_not_equivariant(fn, x, act_in, act_out=lambda t: t, atol=atol, name=name)


def random_rotation(dim: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Uniform random rotation matrix via QR decomposition."""
    a = torch.randn(dim, dim, generator=generator)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:  # ensure proper rotation (det +1)
        q[:, 0] = -q[:, 0]
    return q
