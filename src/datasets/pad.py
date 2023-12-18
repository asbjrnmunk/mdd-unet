import math
from torch.nn import functional as F


def center_pad(x, shape, slice_dim=None, empty_val=0):
    if slice_dim is None:
        assert len(x.shape) == 2
    else:
        assert slice_dim in [0, 1, 2]
        assert len(x.shape) == 3

    if len(x.shape) == 3:
        assert slice_dim is not None

    # F.pad will pad the last two dimensions, so we have to move slice_dim (sigh)
    x = x.movedim(slice_dim, 0) if slice_dim is not None else x

    pad_x = (shape[0] - x.shape[-2]) / 2

    pad_top = math.floor(pad_x)
    pad_bottom = math.ceil(pad_x)

    pad_y = (shape[1] - x.shape[-1]) / 2
    pad_left = math.floor(pad_y)
    pad_right = math.ceil(pad_y)

    pad_amount = (pad_left, pad_right, pad_top, pad_bottom)

    x_pad = F.pad(x, pad_amount, value=empty_val)

    if slice_dim is not None:
        assert x_pad.shape[1:] == shape, x_pad.shape
        # and move slice_dim back
        x_pad = x_pad.movedim(0, slice_dim)
        assert x_pad.shape[slice_dim] == x.shape[0]
    else:
        assert x_pad.shape == shape

    return x_pad
