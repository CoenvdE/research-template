"""Evaluate a checkpoint and write metrics to outputs/eval/metrics.json.

    uv run python -m src.eval --config configs/base.yaml \
        --ckpt_path outputs/last.ckpt

(LightningCLI also provides `validate`/`test`/`predict` subcommands on
src.train directly; this wrapper additionally persists the metrics as JSON so
results land in a file the anti-drift hook watches, nudging an EXPERIMENTS.md
entry.)
"""

import json
from pathlib import Path
from typing import Optional

from lightning.pytorch.cli import LightningCLI

from src.data.synthetic import SyntheticDataModule
from src.models.mlp import TemplateModule


class EvalCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--ckpt_path", type=Optional[str], default=None)


def main():
    cli = EvalCLI(
        TemplateModule, SyntheticDataModule, run=False, save_config_callback=None
    )
    results = cli.trainer.validate(
        cli.model, datamodule=cli.datamodule, ckpt_path=cli.config["ckpt_path"]
    )
    out_dir = Path(cli.trainer.default_root_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(results[0], f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
