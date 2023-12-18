import torch
import torch.nn as nn

from datasets import MRIDataset
from models import UNetBase

from models.utils import predict

import pytest

props = {
    'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
    'dropout_op': None,
    'dropout_op_kwargs': {'p': 0, 'inplace': True},
    'conv_op_kwargs': {'stride': 1, 'dilation': 1, 'bias': False},
    'nonlin': nn.ReLU,
    'nonlin_kwargs': {'inplace': True}
}


@pytest.mark.skip(reason="Havent been updated to mddunet yet...")
def test_predict3d():
    unet = UNetBase(1, 32, 2, 2, 4, props, 3, False, 512, slice_dim=1)

    dataset = MRIDataset("data/lpba40", out_dims=3)
    x, y = dataset[0]
    y_hat = predict.predict3d(x.unsqueeze(0), unet.predict, unet.slice_dim)

    assert y_hat.shape == y.shape
    assert y_hat.dtype == torch.int64

    # assert all empty indicies in y_hat are also empty in x
    assert len(set(empty_idx(y_hat, 0)).difference(set(empty_idx(x, -1)))) == 0
    # assert all empty indicies in y_hat are also empty in y (but y contains many more empty idx)
    assert len(set(empty_idx(y_hat, 0)).difference(set(empty_idx(y, 0)))) == 0


def empty_idx(x, empty_val):
    return (torch.amax(x, dim=(0, 2)) == empty_val).nonzero().ravel().tolist()
