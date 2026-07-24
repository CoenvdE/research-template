"""Samples/sec, GPU memory, and MFU (model FLOPs utilization).

MFU = achieved FLOPs/sec / peak FLOPs/sec of the hardware. Rule of thumb for a
training step: ~3x the forward FLOPs (forward + 2x backward). Set
`peak_flops_per_sec` for your GPU (e.g. H100 bf16 dense ~= 990e12) to get MFU;
leave None to skip.
"""

import time

import torch
from lightning.pytorch.callbacks import Callback

from src.callbacks.flops_params import FlopsParamsCallback


class ThroughputCallback(Callback):
    def __init__(self, log_every_n_steps: int = 50, peak_flops_per_sec: float | None = None):
        self.log_every_n_steps = log_every_n_steps
        self.peak_flops_per_sec = peak_flops_per_sec
        self._window_start = None
        self._samples = 0
        self._steps = 0

    def _flops_per_sample(self, trainer) -> float | None:
        for cb in trainer.callbacks:
            if isinstance(cb, FlopsParamsCallback):
                return cb.flops_per_sample
        return None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._window_start is None:
            self._window_start = time.perf_counter()
        first = batch[0] if isinstance(batch, (tuple, list)) else batch
        self._samples += int(first.shape[0])
        self._steps += 1
        if self._steps < self.log_every_n_steps:
            return

        elapsed = time.perf_counter() - self._window_start
        sps = self._samples / max(elapsed, 1e-12)
        metrics = {"perf/samples_per_sec": sps}

        flops = self._flops_per_sample(trainer)
        if flops is not None and self.peak_flops_per_sec is not None:
            achieved = 3.0 * flops * sps  # fwd + bwd approximation
            metrics["perf/mfu"] = achieved / self.peak_flops_per_sec

        if torch.cuda.is_available():
            metrics["perf/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()

        pl_module.log_dict(metrics, on_step=True, on_epoch=False)
        self._window_start = time.perf_counter()
        self._samples = 0
        self._steps = 0
