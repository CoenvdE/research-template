"""Loss at init should sit near its theoretical value. For MSE on ~N(0,1) targets
with a near-zero-output init, that's ~Var(y) ~= 1. Far off means broken
normalization or a bad init."""

import torch

from src.models.mlp import TemplateModule


def test_initial_loss_near_theoretical(datamodule):
    torch.manual_seed(0)
    model = TemplateModule(in_dim=8, hidden_dim=32, out_dim=4)
    x, y = next(iter(datamodule.train_dataloader()))
    with torch.no_grad():
        loss = model.loss_fn(model(x), y).item()
    assert 0.2 < loss < 5.0, (
        f"initial MSE {loss:.3f} far from ~Var(y)~=1: check target normalization / init"
    )
