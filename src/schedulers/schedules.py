"""LR schedules specified in *fractions* of the total run length.

Both schedules work with `interval: step` or `interval: epoch`; the total number of
units is resolved from the Trainer at `configure_optimizers` time, so percentage
specs survive resumes as long as max_steps / max_epochs stay the same.
"""

import math
from typing import Callable

import lightning.pytorch as pl


def resolve_total_units(trainer: "pl.Trainer", interval: str) -> int:
    if interval == "step":
        total = int(trainer.estimated_stepping_batches)
    elif interval == "epoch":
        if trainer.max_epochs is None or trainer.max_epochs < 0:
            raise ValueError("interval='epoch' requires trainer.max_epochs to be set")
        total = int(trainer.max_epochs)
    else:
        raise ValueError(f"interval must be 'step' or 'epoch', got {interval!r}")
    if total <= 0:
        raise ValueError(f"resolved non-positive schedule length: {total}")
    return total


def wsd_lambda(
    total: int, warmup_frac: float = 0.05, decay_frac: float = 0.2, min_lr_ratio: float = 0.0
) -> Callable[[int], float]:
    """Warmup-Stable-Decay: linear warmup, flat plateau, linear decay to min_lr_ratio."""
    warmup = max(1, round(warmup_frac * total))
    decay_start = round(total * (1.0 - decay_frac))

    def fn(t: int) -> float:
        if t < warmup:
            return (t + 1) / warmup
        if t < decay_start:
            return 1.0
        frac = (t - decay_start) / max(1, total - decay_start)
        return 1.0 - (1.0 - min_lr_ratio) * min(frac, 1.0)

    return fn


def cosine_lambda(
    total: int, warmup_frac: float = 0.05, min_lr_ratio: float = 0.0
) -> Callable[[int], float]:
    """Linear warmup then cosine annealing to min_lr_ratio."""
    warmup = max(1, round(warmup_frac * total))

    def fn(t: int) -> float:
        if t < warmup:
            return (t + 1) / warmup
        frac = min((t - warmup) / max(1, total - warmup), 1.0)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * frac))

    return fn


def build_lambda(
    name: str,
    total: int,
    warmup_frac: float,
    decay_frac: float,
    min_lr_ratio: float,
) -> Callable[[int], float]:
    if name == "wsd":
        return wsd_lambda(total, warmup_frac, decay_frac, min_lr_ratio)
    if name == "cosine":
        return cosine_lambda(total, warmup_frac, min_lr_ratio)
    raise ValueError(f"unknown scheduler {name!r} (expected 'wsd' or 'cosine')")
