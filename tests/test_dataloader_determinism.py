"""Same seed => same batches, regardless of num_workers. Catches seeding bugs
that make runs irreproducible in a way nobody notices."""

import torch

from src.data.synthetic import SyntheticDataModule


def _first_batch(num_workers: int):
    dm = SyntheticDataModule(n_train=64, n_val=16, batch_size=16, seed=7, num_workers=num_workers)
    dm.setup()
    return next(iter(dm.train_dataloader()))


def test_same_seed_same_batches():
    (x1, y1), (x2, y2) = _first_batch(0), _first_batch(0)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)


def test_num_workers_do_not_change_batches():
    (x1, y1), (x2, y2) = _first_batch(0), _first_batch(2)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
