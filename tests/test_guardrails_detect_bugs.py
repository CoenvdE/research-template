"""Tests of the tests: prove the guardrails can actually fail.

A test that cannot fail is worse than no test, because it converts an unknown
into a false assurance. The classic ways a suite becomes vacuous: an expected
value recorded from the code under test, a tolerance loosened until it passes, a
property that a model ignoring its inputs would satisfy anyway. None of those
announce themselves; the suite just stays green.

So every guardrail here is paired with a planted defect it must catch. If you
add a guardrail, add its sensitivity proof next to it, or write down why no
control is possible.

Controls that live with their subject rather than here:
- tests/test_unused_params.py: plants a layer forward never calls
- tests/test_input_ranges.py: plants degrees, raw Pascals, Celsius, NaNs
- tests/equivariance/test_examples.py: pairs each symmetry with a transformation
  the function must NOT respect
"""

import lightning.pytorch as pl
import pytest
import torch

from src.data.synthetic import SyntheticDataModule
from src.models.mlp import TemplateModule
from src.utils.checks import assert_loss_near_theoretical, expected_mse_at_init


def _trainer(root, **kwargs):
    defaults = dict(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=root,
        num_sanity_val_steps=0,
        deterministic=True,
    )
    defaults.update(kwargs)
    return pl.Trainer(**defaults)


def test_overfit_criterion_rejects_a_model_that_cannot_learn(tmp_path):
    """The overfit-one-batch criterion must discriminate.

    Planted defect: lr=0, so the optimizer never updates anything (the shape of a
    misconfigured schedule or an optimizer that never steps). The loss must stay
    near its initial value, i.e. the criterion in test_overfit_one_batch must NOT
    be satisfied. If this model "passed", that test would be proving nothing.
    """
    pl.seed_everything(0)
    dm = SyntheticDataModule(n_train=8, n_val=8, batch_size=8, seed=0)
    model = TemplateModule(hidden_dim=64, lr=0.0, scheduler=None)
    trainer = _trainer(tmp_path, overfit_batches=1, max_epochs=50)
    trainer.fit(model, dm)

    final_loss = trainer.callback_metrics["train/loss"].item()
    assert final_loss >= 0.05, (
        f"a model with lr=0 reached loss {final_loss:.4f}, satisfying the overfit "
        "criterion: the criterion does not discriminate and proves nothing"
    )


def test_initial_loss_check_rejects_a_badly_scaled_init():
    """Planted defect: every parameter scaled 50x, the shape of a bad init."""
    torch.manual_seed(0)
    model = TemplateModule(in_dim=8, hidden_dim=32, out_dim=4)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(50.0)

    x, y = torch.randn(64, 8), torch.randn(64, 4)
    with torch.no_grad():
        loss = model.loss_fn(model(x), y).item()

    with pytest.raises(ValueError, match="theoretical"):
        assert_loss_near_theoretical(loss, expected_mse_at_init(y), name="initial MSE")


def test_initial_loss_check_rejects_uncentered_targets():
    """Planted defect: targets left in raw physical units (a Kelvin-like offset).

    A model at init predicts ~0, so its loss is E[y^2] = Var(y) + mean^2. A large
    offset makes mean^2 dominate and the check fires.
    """
    torch.manual_seed(0)
    model = TemplateModule(in_dim=8, hidden_dim=32, out_dim=4)
    x = torch.randn(64, 8)
    y_kelvin = torch.randn(64, 4) + 288.0
    with torch.no_grad():
        loss = model.loss_fn(model(x), y_kelvin).item()

    with pytest.raises(ValueError, match="theoretical"):
        assert_loss_near_theoretical(loss, expected_mse_at_init(y_kelvin), name="initial MSE")


def test_initial_loss_check_is_blind_to_centered_rescaling():
    """The documented blind spot, asserted so it stays documented.

    Scaling *centered* targets moves Var(y) and E[y^2] together, so the ratio
    stays at 1 and this check cannot see it. That is not a bug to fix here; it is
    a limit to know, and it is why `assert_standardized` exists separately.
    """
    torch.manual_seed(0)
    model = TemplateModule(in_dim=8, hidden_dim=32, out_dim=4)
    x = torch.randn(64, 8)
    y_scaled = 100.0 * torch.randn(64, 4)
    with torch.no_grad():
        loss = model.loss_fn(model(x), y_scaled).item()

    # Passes despite wildly unnormalized targets: the known limitation.
    assert_loss_near_theoretical(loss, expected_mse_at_init(y_scaled), name="initial MSE")


def _train_briefly(root, seed: int) -> torch.Tensor:
    pl.seed_everything(seed)
    dm = SyntheticDataModule(n_train=24, n_val=8, batch_size=8, seed=0, shuffle=False)
    model = TemplateModule(in_dim=8, hidden_dim=32, out_dim=4, lr=1e-2, scheduler=None)
    _trainer(root, max_epochs=1).fit(model, dm)
    return model.net[0].weight.detach().clone()


def test_weight_equality_comparison_discriminates(tmp_path):
    """test_resume asserts allclose(atol=1e-6) between an uninterrupted run and a
    resumed one. That assertion means something only if two premises hold: fixed
    seeds give bit-identical results, and genuinely different runs do NOT compare
    equal. Prove both, or the resume guarantee is unfalsifiable.
    """
    same_a = _train_briefly(tmp_path / "a", seed=0)
    same_b = _train_briefly(tmp_path / "b", seed=0)
    different = _train_briefly(tmp_path / "c", seed=1)

    assert torch.allclose(same_a, same_b, atol=1e-6), (
        "training is not deterministic under a fixed seed, so the resume test's "
        "exact-equality assertion cannot mean what it claims"
    )
    assert not torch.allclose(same_a, different, atol=1e-6), (
        "runs that genuinely differ still compare equal at atol=1e-6: the resume "
        "test's comparison cannot fail and proves nothing"
    )
