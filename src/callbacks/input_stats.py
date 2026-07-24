"""Guard against bad numbers entering the model (degrees instead of radians,
unnormalized fields, NaNs). Checks batch statistics every N steps; logs them
always, and warns or raises when thresholds are violated.
"""

from typing import Literal

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_warn


def _iter_tensors(batch, prefix="t"):
    if isinstance(batch, torch.Tensor):
        yield prefix, batch
    elif isinstance(batch, (tuple, list)):
        for i, b in enumerate(batch):
            yield from _iter_tensors(b, f"{prefix}{i}")
    elif isinstance(batch, dict):
        for k, b in batch.items():
            yield from _iter_tensors(b, f"{prefix}.{k}")


class InputStatsGuard(Callback):
    def __init__(
        self,
        every_n_steps: int = 100,
        absmax_limit: float = 1e3,
        std_range: tuple[float, float] = (1e-3, 1e3),
        mode: Literal["warn", "raise"] = "raise",
        check_first_n_steps: int = 3,
    ):
        self.every_n_steps = every_n_steps
        self.absmax_limit = absmax_limit
        self.std_range = std_range
        self.mode = mode
        self.check_first_n_steps = check_first_n_steps

    def check_batch(self, batch, step: int | None = None) -> dict[str, float]:
        metrics, problems = {}, []
        for name, t in _iter_tensors(batch):
            if not t.is_floating_point() or t.numel() == 0:
                continue
            t = t.detach().float()
            absmax = t.abs().max().item()
            std = t.std().item() if t.numel() > 1 else 0.0
            nans = int(torch.isnan(t).sum() + torch.isinf(t).sum())
            metrics[f"input/{name}_absmax"] = absmax
            metrics[f"input/{name}_std"] = std
            if nans:
                problems.append(f"{name}: {nans} NaN/Inf values")
            if absmax > self.absmax_limit:
                problems.append(f"{name}: absmax {absmax:.3g} > {self.absmax_limit:.3g}")
            if std > 0 and not (self.std_range[0] <= std <= self.std_range[1]):
                problems.append(f"{name}: std {std:.3g} outside {self.std_range}")
        if problems:
            msg = f"InputStatsGuard tripped at step {step}: " + "; ".join(problems)
            if self.mode == "raise":
                raise ValueError(msg)
            rank_zero_warn(msg)
        return metrics

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        step = trainer.global_step
        if step >= self.check_first_n_steps and step % self.every_n_steps != 0:
            return
        metrics = self.check_batch(batch, step=step)
        if metrics:
            pl_module.log_dict(metrics, on_step=True, on_epoch=False)
