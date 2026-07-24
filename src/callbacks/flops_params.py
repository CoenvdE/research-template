"""One generic FLOPs + parameter-count callback (replaces per-model copies).

Counts parameters exactly; measures forward FLOPs per sample with
torch.utils.flop_counter on a single real batch. ThroughputCallback picks up
`flops_per_sample` from here to compute MFU.
"""

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_info, rank_zero_warn


class FlopsParamsCallback(Callback):
    def __init__(self):
        self.flops_per_sample: float | None = None

    def on_fit_start(self, trainer, pl_module):
        total = sum(p.numel() for p in pl_module.parameters())
        trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        metrics = {"model/params_total": float(total), "model/params_trainable": float(trainable)}

        try:
            from torch.utils.flop_counter import FlopCounterMode

            batch = next(iter(trainer.datamodule.train_dataloader()))
            x = batch[0][:1].to(pl_module.device)
            was_training = pl_module.training
            pl_module.eval()
            with FlopCounterMode(display=False) as counter:
                pl_module(x)
            if was_training:
                pl_module.train()
            self.flops_per_sample = float(counter.get_total_flops())
            metrics["model/forward_flops_per_sample"] = self.flops_per_sample
        except Exception as e:  # counting must never kill a run
            rank_zero_warn(f"FLOP counting failed ({e}); continuing without it.")

        rank_zero_info(f"[flops_params] {metrics}")
        if trainer.logger is not None:
            trainer.logger.log_hyperparams(metrics)
