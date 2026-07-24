"""The most diagnostic smoke test there is: a model that cannot drive the loss to
~0 on a single small batch has a broken loss, data pipeline, or optimizer wiring."""

import lightning.pytorch as pl

from src.data.synthetic import SyntheticDataModule
from src.models.mlp import TemplateModule


def test_overfit_one_batch(tmp_path):
    pl.seed_everything(0)
    dm = SyntheticDataModule(n_train=8, n_val=8, batch_size=8, seed=0)
    model = TemplateModule(hidden_dim=64, lr=1e-2, scheduler=None)
    trainer = pl.Trainer(
        overfit_batches=1,
        max_epochs=300,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, dm)
    final_loss = trainer.callback_metrics["train/loss"].item()
    assert final_loss < 0.05, f"could not overfit one batch: final loss {final_loss:.4f}"
