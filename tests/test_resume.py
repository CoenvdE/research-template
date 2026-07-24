"""Interrupt-and-resume must equal an uninterrupted run: weights, EMA state, LR.

This is the test that catches scheduler-state bugs, callbacks without state_dict,
and every other 'weird bug after resume'.
"""

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from src.callbacks import EMACallback
from src.data.synthetic import SyntheticDataModule
from src.models.mlp import TemplateModule


class StopAfterSteps(Callback):
    """Simulate a preemption: request a graceful stop after N optimizer steps."""

    def __init__(self, n: int):
        self.n = n

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step >= self.n:
            trainer.should_stop = True


def _make(seed=0):
    pl.seed_everything(seed)
    # shuffle=False: Lightning does not checkpoint sampler RNG (the documented
    # mid-epoch/shuffle caveat), so exact-equality resume testing needs a fixed
    # batch order. Optimizer, scheduler, and callback state are what's under test.
    dm = SyntheticDataModule(n_train=24, n_val=16, batch_size=8, seed=0, shuffle=False)
    model = TemplateModule(in_dim=8, hidden_dim=32, out_dim=4, lr=1e-2, scheduler="wsd")
    return model, dm


def _trainer(tmp_path, callbacks, max_epochs=2):
    return pl.Trainer(
        max_epochs=max_epochs,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        deterministic=True,
    )


def test_resume_matches_uninterrupted(tmp_path):
    # Run A: 2 epochs (6 steps) in one go.
    model_a, dm = _make()
    ema_a = EMACallback(decay=0.99)
    _trainer(tmp_path / "a", [ema_a]).fit(model_a, dm)

    # Run B: stop after epoch 1 (step 3), then resume from last.ckpt to the end.
    model_b, dm = _make()
    ema_b = EMACallback(decay=0.99)
    ckpt = ModelCheckpoint(dirpath=tmp_path / "ckpt", save_last=True, every_n_epochs=1)
    _trainer(tmp_path / "b1", [ema_b, ckpt, StopAfterSteps(3)]).fit(model_b, dm)
    assert (tmp_path / "ckpt" / "last.ckpt").exists()

    model_b2, dm = _make()
    ema_b2 = EMACallback(decay=0.99)
    trainer2 = _trainer(tmp_path / "b2", [ema_b2])
    trainer2.fit(model_b2, dm, ckpt_path=str(tmp_path / "ckpt" / "last.ckpt"))

    assert trainer2.global_step == 6
    # Weights identical to the uninterrupted run.
    for (n, pa), (_, pb) in zip(
        model_a.state_dict().items(), model_b2.state_dict().items()
    ):
        assert torch.allclose(pa, pb, atol=1e-6), f"weight mismatch after resume: {n}"
    # EMA state restored and identical.
    for n in ema_a._ema:
        assert torch.allclose(ema_a._ema[n], ema_b2._ema[n], atol=1e-6), (
            f"EMA mismatch after resume: {n}"
        )
    # Scheduler continued (same final LR).
    lr_a = model_a.trainer.optimizers[0].param_groups[0]["lr"]
    lr_b = trainer2.optimizers[0].param_groups[0]["lr"]
    assert abs(lr_a - lr_b) < 1e-12
