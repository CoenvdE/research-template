"""Log activation RMS per module: catches explosions *inside* the net (bad init,
missing norm layer) that clean inputs won't reveal."""

import torch
from lightning.pytorch.callbacks import Callback
from torch import nn


class ActivationMonitor(Callback):
    # Not a config arg (jsonargparse can't serialize classes); override in a subclass.
    module_types: tuple = (nn.Linear, nn.Conv2d)

    def __init__(self, every_n_steps: int = 100):
        self.every_n_steps = every_n_steps
        self._handles = []
        self._rms: dict[str, float] = {}
        self._armed = False

    def on_fit_start(self, trainer, pl_module):
        def make_hook(name):
            def hook(module, args, output):
                if self._armed and isinstance(output, torch.Tensor):
                    rms = output.detach().float().pow(2).mean().sqrt().item()
                    self._rms[f"act_rms/{name}"] = rms

            return hook

        for name, module in pl_module.named_modules():
            if isinstance(module, self.module_types):
                self._handles.append(module.register_forward_hook(make_hook(name)))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._armed = trainer.global_step % self.every_n_steps == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._armed and self._rms:
            pl_module.log_dict(self._rms, on_step=True, on_epoch=False)
            self._rms = {}
        self._armed = False

    def teardown(self, trainer, pl_module, stage):
        for h in self._handles:
            h.remove()
        self._handles = []
