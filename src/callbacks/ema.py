"""Exponential moving average of model weights.

Validation and checkpoints see the EMA weights (swapped in around val). State is
checkpointed via state_dict/load_state_dict so resume is exact.
"""

import torch
from lightning.pytorch.callbacks import Callback


class EMACallback(Callback):
    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self._ema: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}

    def on_fit_start(self, trainer, pl_module):
        params = {n: p for n, p in pl_module.named_parameters() if p.requires_grad}
        if not self._ema:
            self._ema = {n: p.detach().clone() for n, p in params.items()}
        else:  # restored from checkpoint: move to wherever the params live now
            self._ema = {n: t.to(params[n].device) for n, t in self._ema.items() if n in params}

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for n, p in pl_module.named_parameters():
            if n in self._ema:
                self._ema[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def _swap_in(self, pl_module):
        for n, p in pl_module.named_parameters():
            if n in self._ema:
                self._backup[n] = p.detach().clone()
                p.copy_(self._ema[n])

    @torch.no_grad()
    def _swap_out(self, pl_module):
        for n, p in pl_module.named_parameters():
            if n in self._backup:
                p.copy_(self._backup[n])
        self._backup = {}

    def on_validation_start(self, trainer, pl_module):
        if self._ema:
            self._swap_in(pl_module)

    def on_validation_end(self, trainer, pl_module):
        self._swap_out(pl_module)

    def state_dict(self):
        return {"decay": self.decay, "ema": self._ema}

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self._ema = state_dict["ema"]
