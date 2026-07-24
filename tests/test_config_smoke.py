"""The committed configs must always run: one fast_dev_run batch through the
full CLI stack (model, data, every callback in base.yaml). Catches config drift
the moment a rename breaks a class_path or init_arg."""

from pathlib import Path

from lightning.pytorch.cli import LightningCLI

from src.data.synthetic import SyntheticDataModule
from src.models.mlp import TemplateModule

ROOT = Path(__file__).resolve().parent.parent


def test_base_config_runs_fast_dev_run(tmp_path):
    LightningCLI(
        TemplateModule,
        SyntheticDataModule,
        save_config_callback=None,
        args=[
            "fit",
            "--config",
            str(ROOT / "configs" / "base.yaml"),
            "--trainer.fast_dev_run",
            "true",
            "--trainer.logger",
            "false",
            "--trainer.default_root_dir",
            str(tmp_path),
            "--trainer.enable_progress_bar",
            "false",
        ],
    )
