import pytest

from src.schedulers.schedules import cosine_lambda, wsd_lambda


def test_wsd_shape():
    fn = wsd_lambda(total=100, warmup_frac=0.1, decay_frac=0.2, min_lr_ratio=0.1)
    assert fn(0) == pytest.approx(0.1)      # first warmup step
    assert fn(9) == pytest.approx(1.0)      # warmup done
    assert fn(50) == 1.0                    # stable plateau
    assert fn(79) == 1.0                    # last plateau step
    assert fn(99) == pytest.approx(0.1, abs=0.05)  # decayed to min ratio


def test_cosine_shape():
    fn = cosine_lambda(total=100, warmup_frac=0.1, min_lr_ratio=0.05)
    assert fn(0) == pytest.approx(0.1)
    assert fn(9) == pytest.approx(1.0)
    assert fn(99) == pytest.approx(0.05, abs=0.01)


def test_monotone_decay():
    fn = wsd_lambda(total=1000, warmup_frac=0.05, decay_frac=0.3, min_lr_ratio=0.0)
    values = [fn(t) for t in range(700, 1000)]
    assert all(a >= b for a, b in zip(values, values[1:]))
