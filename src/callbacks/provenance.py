"""Log git SHA, dirty flag, and seed to the logger at fit start, so every run is
traceable to the exact code and config that produced it."""

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_info, rank_zero_warn

from src.utils.provenance import git_info


class ProvenanceCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        info = git_info()
        if info["git_dirty"]:
            rank_zero_warn(
                "Running with uncommitted changes (git_dirty=True): this run is not "
                "exactly reproducible from the logged SHA."
            )
        rank_zero_info(f"[provenance] {info}")
        if trainer.logger is not None:
            trainer.logger.log_hyperparams(info)
