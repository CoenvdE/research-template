"""Standalone correctness checks used by tests and available at runtime."""

import math

import torch


def assert_radians(t: torch.Tensor, name: str = "coords") -> None:
    """Fail loudly if angular coordinates look like degrees instead of radians."""
    absmax = t.abs().max().item()
    if absmax > math.pi + 1e-6:
        raise ValueError(
            f"{name}: |max| = {absmax:.4g} > pi. Angular inputs must be radians; "
            "this looks like degrees."
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
