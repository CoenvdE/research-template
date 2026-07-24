"""Every trainable parameter must receive a gradient after one step; anything
that doesn't is a disconnected part of the graph."""

from torch import nn

from src.utils.checks import find_unused_parameters


def test_no_unused_parameters(model, datamodule):
    batch = next(iter(datamodule.train_dataloader()))
    unused = find_unused_parameters(model, batch)
    assert unused == [], f"parameters received no gradient: {unused}"


def test_detects_planted_unused_parameter(model, datamodule):
    model.dead_head = nn.Linear(4, 4)  # never called in forward
    batch = next(iter(datamodule.train_dataloader()))
    unused = find_unused_parameters(model, batch)
    assert any("dead_head" in n for n in unused)
