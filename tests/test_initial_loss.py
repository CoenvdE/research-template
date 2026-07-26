"""Loss at init should sit near its theoretical know-nothing value: Var(y) for
MSE, which is ~1 for standardized targets. Far above means broken normalization
or init; far below at step zero means suspect leakage.

The expected value is derived from the targets, never recorded from what the
model happens to produce. See tests/test_guardrails_detect_bugs.py for the proof
that this check can fail.
"""

import torch

from src.models.mlp import TemplateModule
from src.utils.checks import assert_loss_near_theoretical, expected_mse_at_init


def test_initial_loss_near_theoretical(datamodule):
    torch.manual_seed(0)
    model = TemplateModule(in_dim=8, hidden_dim=32, out_dim=4)
    x, y = next(iter(datamodule.train_dataloader()))
    with torch.no_grad():
        loss = model.loss_fn(model(x), y).item()
    assert_loss_near_theoretical(loss, expected_mse_at_init(y), name="initial MSE")
