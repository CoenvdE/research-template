"""Split step time into dataloader-wait vs forward vs backward.

If `time/data_frac` is high, the dataloader is the bottleneck (raise num_workers,
prefetch, or move preprocessing offline).
"""

import time

import torch
from lightning.pytorch.callbacks import Callback


class TimingCallback(Callback):
    def __init__(self, log_every_n_steps: int = 50, cuda_sync: bool = True):
        self.log_every_n_steps = log_every_n_steps
        self.cuda_sync = cuda_sync
        self._reset()
        self._prev_batch_end = None
        self._t_batch_start = None
        self._t_bwd_start = None

    def _reset(self):
        self._acc = {"data": 0.0, "forward": 0.0, "backward": 0.0, "total": 0.0}
        self._n = 0

    def _now(self):
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._t_batch_start = self._now()
        if self._prev_batch_end is not None:
            self._acc["data"] += self._t_batch_start - self._prev_batch_end

    def on_before_backward(self, trainer, pl_module, loss):
        self._t_bwd_start = self._now()
        if self._t_batch_start is not None:
            self._acc["forward"] += self._t_bwd_start - self._t_batch_start

    def on_after_backward(self, trainer, pl_module):
        if self._t_bwd_start is not None:
            self._acc["backward"] += self._now() - self._t_bwd_start

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        end = self._now()
        if self._t_batch_start is not None:
            self._acc["total"] += end - self._t_batch_start
        self._prev_batch_end = end
        self._n += 1
        if self._n >= self.log_every_n_steps:
            step_total = self._acc["total"] + self._acc["data"]
            metrics = {
                "time/data_s": self._acc["data"] / self._n,
                "time/forward_s": self._acc["forward"] / self._n,
                "time/backward_s": self._acc["backward"] / self._n,
                "time/step_s": step_total / self._n,
                "time/data_frac": self._acc["data"] / max(step_total, 1e-12),
            }
            pl_module.log_dict(metrics, on_step=True, on_epoch=False)
            self._reset()
