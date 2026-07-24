"""Synthetic regression data, fully deterministic from a seed.

Ships with the template so every test and smoke run works on any machine, CPU-only,
with no data download. Also demonstrates the split-metadata pattern that the
leakage tests assert against: replace it with real sample ids / timestamps.
"""

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


class SyntheticDataModule(pl.LightningDataModule):
    def __init__(
        self,
        n_train: int = 512,
        n_val: int = 128,
        in_dim: int = 8,
        out_dim: int = 4,
        batch_size: int = 64,
        num_workers: int = 0,
        seed: int = 0,
        temporal_gap: int = 8,
        shuffle: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        h = self.hparams
        g = torch.Generator().manual_seed(h.seed)
        # Fixed random linear map, scaled so targets are ~N(0, 1): initial MSE of a
        # near-zero-output model should then sit near 1.0 (the initial-loss test).
        w = torch.randn(h.in_dim, h.out_dim, generator=g) / (h.in_dim**0.5)
        x = torch.randn(h.n_train + h.n_val, h.in_dim, generator=g)
        y = x @ w
        self.train_ds = TensorDataset(x[: h.n_train], y[: h.n_train])
        self.val_ds = TensorDataset(x[h.n_train :], y[h.n_train :])
        # Split metadata: pretend samples are a time series. Real datamodules should
        # populate these from actual sample ids / timestamps.
        self.train_ids = list(range(h.n_train))
        self.val_ids = list(range(h.n_train + h.temporal_gap, h.n_train + h.temporal_gap + h.n_val))
        self.train_times = torch.arange(h.n_train, dtype=torch.float64)
        self.val_times = self.train_times[-1] + h.temporal_gap + torch.arange(h.n_val).double()
        self.temporal_gap = h.temporal_gap

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle,
            num_workers=self.hparams.num_workers,
            generator=torch.Generator().manual_seed(self.hparams.seed),
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
