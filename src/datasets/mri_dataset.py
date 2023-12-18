import torch
from torch.utils.data import Dataset
import torchio as tio
from bisect import bisect_right
import os

from . import preprocess_volumes, center_pad


def calculate_weights(class_counts, num_classes):
    """
        Calculate weights for Weighted Cross Entropy
    """
    weights = torch.ones(num_classes)

    if class_counts is not None:
        voxels = sum(class_counts.values())
        weights[0] = 1 - (class_counts[0]/voxels)

        for k in range(1, num_classes):
            weights[k] = 1/(num_classes-1) - (class_counts[k] / voxels)

        assert torch.sum(weights).data.item() - 1 < 1e-8

    return weights


class MRIDataset(Dataset):
    def __init__(
        self,
        path: str,
        subfolder: str,
        out_dims: int,
        slice_dim: bool,
        crop_size: int,
        ignore_empty: bool = None,
        consider_fg: bool = None,
        num_classes: int = 3,
        epoch_size: int = None,
        p_pre_aug: float = 0.,
        preprocess_augmentation: tio.transforms.Transform = None,
        online_augmentation: tio.transforms.Transform = None,
        force_preprocess: bool = False
    ):
        """
        Dataset which handles MRI data. It can do a lot. But be careful!

        Params:
            path: path to images and labels folders (str).
            epoch_size: number of samples per epoch
            out_dims: amount of dimensions to output pr. brain. Either 2 or 3.
            slice_dim: dimension to slice from (int). 0: sagittal, 1: coronal, 2: axial. Default: 1.
            """
        self.images_path = os.path.join(os.getcwd(), path, "images")
        self.labels_path = os.path.join(os.getcwd(), path, "labels")

        self.images_path = os.path.join(self.images_path, subfolder)
        self.labels_path = os.path.join(self.labels_path, subfolder)

        assert out_dims in [2, 3], "This is science fiction."
        self.out_dims = out_dims
        assert slice_dim is not None  # otherwise we dont know what dimension to pad
        assert slice_dim in [0, 1, 2], "This requires a red pill or are you Neo?!."
        self.slice_dim = slice_dim
        self.crop_size = crop_size
        self.augmentation = online_augmentation

        if out_dims == 2:
            assert ignore_empty is not None
            assert consider_fg is not None

        self.files, slice_infos = preprocess_volumes(
            path,
            subfolder,
            p_aug=p_pre_aug,
            augmentation=preprocess_augmentation,
            out_dims=out_dims,
            crop_size=crop_size,
            num_classes=num_classes,
            ignore_empty=ignore_empty,
            consider_fg=consider_fg,
            force=force_preprocess
        )

        assert len(self.files) > 0

        # the way we index of dataset is very different from 2d and 3d
        # in 2d each slice is considered independently
        # in 3d each volume is considered independently
        if out_dims == 2:
            assert slice_infos is not None, slice_infos
            slice_info = slice_infos[slice_dim]
            self.lens = slice_info["lens"]
            self.fg_idx = slice_info["fg_idx"]
            self.empty_idx = slice_info["empty_idx"]
            self.counts = slice_info["counts"]
            self.len = slice_info["n_slices"]  # length of our dataset is amount of slices
            self.ce_weights = calculate_weights(self.counts, num_classes)
        else:
            assert slice_infos is None, slice_infos
            self.len = len(self.files)  # length of our dataset is amount of volumes

        # Epoch size can be set independently because we over- and undersample the dataset
        if epoch_size is not None:
            self.epoch_size = epoch_size
        elif ignore_empty:
            assert out_dims == 2, "3d volumes are never empty you dummy."
            self.epoch_size = self.len - len(self.empty_idx)
        else:
            self.epoch_size = self.len

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        # In the following shapes we assume slice_dim == 1.
        if self.out_dims == 2:
            x, y = self._getitem_2d(idx)
            x = x.unsqueeze(0)  # add empty channel dimension

            assert len(x.shape) == 3
            assert len(y.shape) == 2

            if self.augmentation is not None:
                x = x.unsqueeze(self.slice_dim + 1)               # make tensor 4d for torchio
                y = y.unsqueeze(0).unsqueeze(self.slice_dim + 1)  # make tensor 4d for torchio

                assert x.shape == y.shape
                assert len(x.shape) == 4
                assert len(y.shape) == 4

                subj = tio.Subject({
                    "mri": tio.ScalarImage(tensor=x),
                    "seg": tio.LabelMap(tensor=y)
                })
                subj = self.augmentation(subj)

                x = subj.mri.tensor.squeeze(self.slice_dim + 1)  # (1, H, W)
                y = subj.seg.tensor.squeeze().long()             # (H, W)
            else:
                y = y.long()
        else:
            x, y = self._getitem_3d(idx)

        assert len(x.shape) == 3, x.shape
        assert len(y.shape) == self.out_dims, y.shape

        return x, y  # Both (H, D, W) in 3d or (1, H, W) in 2d

    def _getitem_3d(self, idx):
        image_path = os.path.join(self.images_path, self.files[idx])
        label_path = os.path.join(self.labels_path, self.files[idx])
        x = torch.load(image_path)
        y = torch.load(label_path)

        assert len(x.shape) == 3
        assert len(y.shape) == 3

        shape = (self.crop_size, self.crop_size)

        x = center_pad(x, shape, slice_dim=self.slice_dim, empty_val=x.min())
        y = center_pad(y, shape, slice_dim=self.slice_dim, empty_val=y.min())

        return x, y  # shape (H, D, W)

    def _getitem_2d(self, idx):
        # perform binary search on lens to get idx of corresponding file
        fileidx = bisect_right(self.lens, idx) - 1
        assert fileidx < len(self.files)

        image_path = os.path.join(self.images_path, self.files[fileidx])
        label_path = os.path.join(self.labels_path, self.files[fileidx])

        img_data = torch.load(image_path)
        label_data = torch.load(label_path)

        assert len(img_data.shape) == 3
        assert len(label_data.shape) == 3

        # for all fileidx : self.lens[file_idx] <= idx, so k >= 0
        k = torch.tensor(idx - self.lens[fileidx])

        assert k >= 0, f"k: {k}, idx: {idx}, fileidx: [fileidx], file len: {self.lens[fileidx]}"
        assert k < img_data.shape[self.slice_dim], (k, img_data.shape[self.slice_dim])

        x = torch.index_select(img_data, self.slice_dim, k)
        y = torch.index_select(label_data, self.slice_dim, k)

        x = x.squeeze()
        y = y.squeeze()

        assert len(x.shape) == 2, x.shape
        assert len(y.shape) == 2, y.shape

        shape = (self.crop_size, self.crop_size)

        x = center_pad(x, shape, empty_val=x.min())
        y = center_pad(y, shape, empty_val=y.min())

        assert x.shape == shape
        return x, y  # shape: (H, W)
