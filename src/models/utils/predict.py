import torch
from typing import Callable


def predict3d(x: torch.Tensor, predict2d: Callable[[torch.Tensor], torch.Tensor], slice_dim: int):
    assert len(x.shape) == 4
    assert x.shape[0] == 1  # batch of one

    slice_dim = slice_dim + 1  # slice_dim is without batch dimension

    preds = []
    for i in torch.arange(x.shape[slice_dim], device=x.device):
        s = torch.index_select(x, slice_dim, i)
        s = s.squeeze().unsqueeze(0)  # (1, H, W)
        assert len(s.shape) == 3

        # ignore empty slices and just predict zero
        if s.max() == s.min():
            pred = torch.zeros(s.shape, dtype=torch.int64, device=s.device)
        else:
            s = s.unsqueeze(0)  # (B, C, H, W) where B == 1 and C == 1 (required by forward)
            pred = predict2d(s)

        assert len(pred.shape) == 3  # (B, H, W)
        preds.append(pred)

    y_hat = torch.vstack(preds)            # (length of slice_dim, H, W)
    y_hat = y_hat.movedim(0, slice_dim-1)  # (H, length of slice_dim, W) (or wherever slice_dim is)

    return y_hat.long()
