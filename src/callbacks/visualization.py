"""Visualize predictions on the SAME fixed val batch every epoch, so panels are
comparable across the run. Default plot is pred-vs-target scatter; subclass and
override `make_figure` for domain-specific plots (maps, spectra, ...)."""

import torch
from lightning.pytorch.callbacks import Callback


class VisualizationCallback(Callback):
    def __init__(self, every_n_epochs: int = 1, max_points: int = 512):
        self.every_n_epochs = every_n_epochs
        self.max_points = max_points
        self._fixed_batch = None

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if self._fixed_batch is None and batch_idx == 0:
            self._fixed_batch = tuple(
                b.detach().cpu() if isinstance(b, torch.Tensor) else b for b in batch
            )

    def make_figure(self, pl_module, batch):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x, y = batch
        with torch.no_grad():
            pred = pl_module(x.to(pl_module.device)).cpu()
        fig, ax = plt.subplots(figsize=(4, 4))
        yf, pf = y.flatten()[: self.max_points], pred.flatten()[: self.max_points]
        ax.scatter(yf, pf, s=4, alpha=0.5)
        lims = [min(yf.min(), pf.min()), max(yf.max(), pf.max())]
        ax.plot(lims, lims, "k--", lw=0.8)
        ax.set_xlabel("target")
        ax.set_ylabel("prediction")
        ax.set_title(f"epoch {pl_module.current_epoch}")
        fig.tight_layout()
        return fig

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or self._fixed_batch is None:
            return
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        fig = self.make_figure(pl_module, self._fixed_batch)
        logged = False
        try:
            import wandb

            if wandb.run is not None:
                wandb.log({"viz/fixed_val_batch": wandb.Image(fig)}, step=trainer.global_step)
                logged = True
        except ImportError:
            pass
        if not logged:
            import os

            out = os.path.join(trainer.default_root_dir, "viz")
            os.makedirs(out, exist_ok=True)
            fig.savefig(os.path.join(out, f"epoch_{trainer.current_epoch:04d}.png"), dpi=120)
        import matplotlib.pyplot as plt

        plt.close(fig)
