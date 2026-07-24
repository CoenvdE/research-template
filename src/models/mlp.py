"""Example LightningModule. Replace the network; keep the optimizer/scheduler wiring."""

from typing import Literal, Optional

import lightning.pytorch as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from src.schedulers.schedules import build_lambda, resolve_total_units


class TemplateModule(pl.LightningModule):
    def __init__(
        self,
        in_dim: int = 8,
        hidden_dim: int = 64,
        out_dim: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        scheduler: Optional[Literal["wsd", "cosine"]] = "wsd",
        warmup_frac: float = 0.05,
        decay_frac: float = 0.2,
        min_lr_ratio: float = 0.0,
        scheduler_interval: Literal["step", "epoch"] = "step",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        if self.hparams.scheduler is None:
            return opt
        interval = self.hparams.scheduler_interval
        total = resolve_total_units(self.trainer, interval)
        fn = build_lambda(
            self.hparams.scheduler,
            total,
            self.hparams.warmup_frac,
            self.hparams.decay_frac,
            self.hparams.min_lr_ratio,
        )
        sched = LambdaLR(opt, fn)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": interval},
        }
