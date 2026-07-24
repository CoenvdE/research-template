import pytest

from src.data.synthetic import SyntheticDataModule
from src.models.mlp import TemplateModule


@pytest.fixture
def datamodule():
    dm = SyntheticDataModule(n_train=24, n_val=16, in_dim=8, out_dim=4, batch_size=8, seed=0)
    dm.setup()
    return dm


@pytest.fixture
def model():
    return TemplateModule(in_dim=8, hidden_dim=32, out_dim=4, lr=1e-2, scheduler="wsd")
