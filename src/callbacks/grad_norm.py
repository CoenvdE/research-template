"""Log the total gradient norm (and optionally the largest per-parameter norms).
Spikes here are the early-warning signal for instability."""

from lightning.pytorch.callbacks import Callback


class GradNormCallback(Callback):
    def __init__(self, every_n_steps: int = 50, top_k: int = 0):
        self.every_n_steps = every_n_steps
        self.top_k = top_k

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if trainer.global_step % self.every_n_steps != 0:
            return
        norms = {
            name: p.grad.detach().norm(2).item()
            for name, p in pl_module.named_parameters()
            if p.grad is not None
        }
        if not norms:
            return
        total = sum(v**2 for v in norms.values()) ** 0.5
        metrics = {"grad/total_norm": total}
        if self.top_k:
            for name, v in sorted(norms.items(), key=lambda kv: -kv[1])[: self.top_k]:
                metrics[f"grad/norm.{name}"] = v
        pl_module.log_dict(metrics, on_step=True, on_epoch=False)
