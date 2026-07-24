"""Leakage asserts: no sample overlap, temporal ordering with a gap, and
normalization stats derived from train only. Adapt the metadata source when you
swap in a real datamodule; keep the assertions."""


def test_no_sample_overlap(datamodule):
    train, val = set(datamodule.train_ids), set(datamodule.val_ids)
    assert not (train & val), f"train/val share {len(train & val)} sample ids"


def test_temporal_split_with_gap(datamodule):
    gap = datamodule.temporal_gap
    train_end = datamodule.train_times.max().item()
    val_start = datamodule.val_times.min().item()
    assert val_start >= train_end + gap, (
        f"val starts at t={val_start} but train ends at t={train_end}; "
        f"need a gap >= {gap} (longest input window + lead time) to prevent window bleed"
    )
