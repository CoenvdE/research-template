"""Training entrypoint via LightningCLI.

Usage:
    uv run python -m src.train fit --config configs/base.yaml
    uv run python -m src.train fit --config configs/base.yaml \
        --config configs/experiment/wandb.yaml
Resume (same wandb run, see README "Resume" section):
    uv run python -m src.train fit --config configs/base.yaml \
        --ckpt_path outputs/last.ckpt
"""

from lightning.pytorch.cli import LightningCLI

from src.data.synthetic import SyntheticDataModule
from src.models.mlp import TemplateModule


def main():
    LightningCLI(
        TemplateModule,
        SyntheticDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
