import torch
from datasets import MRIDataset

from models.utils import predict
import pytest


class StateFunc():
    def __init__(self):
        self.n = 0

    def __call__(self, x):
        self.n += 1
        return torch.ones(x.squeeze(0).shape) * self.n


@pytest.mark.parametrize("slice_dim", [0, 1, 2])
def test_predict3d(slice_dim):
    crop_size = 256

    dataset = MRIDataset(
        "data/harp_tiny",
        subfolder="preprocessed",
        out_dims=3,
        slice_dim=slice_dim,
        crop_size=crop_size
    )

    x, y = dataset[0]

    y_hat = predict.predict3d(x.unsqueeze(0), StateFunc(), slice_dim)

    assert y_hat.shape == y.shape
    assert y_hat.dtype == torch.int64

    # assert all empty indicies in y_hat are also empty in x
    sd = slice_dim
    assert len(set(empty_idx(y_hat, 0, sd)).difference(set(empty_idx(x, -1, sd)))) == 0
    # assert all empty indicies in y_hat are also empty in y (but y contains many more empty idx)
    assert len(set(empty_idx(y_hat, 0, sd)).difference(set(empty_idx(y, 0, sd)))) == 0


def empty_idx(x, empty_val, slice_dim):
    if slice_dim == 0:
        dim = (1, 2)
    elif slice_dim == 1:
        dim = (0, 2)
    else:
        dim = (0, 1)
    return (torch.amax(x, dim=dim) == empty_val).nonzero().ravel().tolist()
