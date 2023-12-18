import torch
import os
import copy
import numpy as np
from tqdm import tqdm
import torchio as tio
import pickle
import gc


def preprocess_volumes(
    path,
    subfolder,
    out_dims: int,
    crop_size: int,
    num_classes: int,
    ignore_empty: bool,
    consider_fg: bool,
    p_aug: float = 0.,
    augmentation: tio.transforms.Transform = None,
    force: bool = False,
):
    """
        Preprocess the volumes invariant to the slide_dim. This also means we will recompute
        slide_info for each possible slide_dim.
    """

    print(f"Preprocessing: {path}, output subfolder: {subfolder}")

    if p_aug != 0:
        assert augmentation is not None

    in_images_path = os.path.join(os.getcwd(), path, "images")
    in_labels_path = os.path.join(os.getcwd(), path, "labels")
    out_images_path = os.path.join(in_images_path, subfolder)
    out_labels_path = os.path.join(in_labels_path, subfolder)
    slice_infos_path = os.path.join(os.getcwd(), path, "slice_infos.pkl")

    in_files = [file for file in next(os.walk(in_images_path))[2] if file.endswith(".nii")]

    in_files.sort()

    n_files = len(in_files)
    n_augmented = int(p_aug * n_files)
    precomputed = os.path.exists(out_images_path) and os.path.exists(out_labels_path)
    slice_infos_exists = os.path.isfile(slice_infos_path)

    if precomputed:
        precomputed_files = [
            file for file in next(os.walk(out_images_path))[2] if file.endswith(".pkl")
        ]
        if len(precomputed_files) == n_files + n_augmented and not force and slice_infos_exists:
            print("Volumes already preprocessed. Reusing...")
            slice_infos = load_pickle_from(slice_infos_path) if out_dims == 2 else None
            return precomputed_files, slice_infos
        else:
            print("Deleting previous preprocessed volumes and reprocessing.")
            for precomputed_file in precomputed_files:
                os.remove(f"{out_images_path}/{precomputed_file}")
                os.remove(f"{out_labels_path}/{precomputed_file}")
    else:
        os.mkdir(out_images_path)
        os.mkdir(out_labels_path)

    out_files, slice_infos = prepare_volumes(
        in_files,
        n_augmented=n_augmented,
        in_images_path=in_images_path,
        in_labels_path=in_labels_path,
        out_images_path=out_images_path,
        out_labels_path=out_labels_path,
        augmentation=augmentation,
        out_dims=out_dims,
        crop_size=crop_size,
        num_classes=num_classes,
        ignore_empty=ignore_empty,
        consider_fg=consider_fg,
    )

    if slice_infos is not None:
        if slice_infos_exists:
            os.remove(slice_infos_path)
        store_pickle_to(slice_infos_path, slice_infos)

    return out_files, slice_infos


def prepare_volumes(files, n_augmented, in_images_path, in_labels_path, out_images_path,
                    out_labels_path, augmentation, out_dims, crop_size, num_classes, ignore_empty,
                    consider_fg):

    out_files = []

    if out_dims == 2:
        dim_info = {
            # `lens` contains the accumulated image count on dimension `dim` ordered as
            # in `files`. So if we need image number `i` we can do a binary search on
            # `lens` and the index of the element matching `i` will be the index into
            # `files`.
            # NB: The value is the accumulated length that the volume _starts_ at, so first has
            # len 0 and so on
            "lens": [],

            # Global slice id's of slices with foreground
            "fg_idx": [],

            # Global slice id's of slices without any non_zero pixel values
            "empty_idx": [],

            # Count of number of pixels in the labels with different class values
            "counts": dict([(i, 0) for i in range(1, num_classes)]),

            # Accumulated number of slices
            "n_slices": 0
        }

        # contains one dim info per possible slice_dim
        slice_infos = {
            0: copy.deepcopy(dim_info),
            1: copy.deepcopy(dim_info),
            2: copy.deepcopy(dim_info),
        }
    else:
        slice_infos = None

    for i, in_file in tqdm(enumerate(files), desc="Preprocessing volumes", total=len(files)):
        subj = tio.Subject({
            "mri": tio.ScalarImage(f"{in_images_path}/{in_file}"),
            "seg": tio.LabelMap(f"{in_labels_path}/{in_file}")
        })

        out_file = in_file.rstrip(".nii") + ".pkl"

        subj = transform_volume(subj, crop_size=crop_size)

        torch.save(subj.mri.tensor.squeeze(), f"{out_images_path}/{out_file}")
        torch.save(subj.seg.tensor.squeeze(), f"{out_labels_path}/{out_file}")

        if out_dims == 2:
            for slice_dim in [0, 1, 2]:
                update_dim_info(
                    slice_dim,
                    slice_infos[slice_dim],  # dicts are pass by ref, so "global" dict is updated
                    x=subj.mri.tensor.squeeze().numpy(),
                    y=subj.seg.tensor.squeeze().numpy(),
                    num_classes=num_classes,
                    ignore_empty=ignore_empty,
                    consider_fg=consider_fg,
                )

        if i < n_augmented and augmentation is not None:
            aug_subj = augmentation(subj)
            torch.save(aug_subj.mri.tensor.squeeze(), f"{out_images_path}/aug_{out_file}")
            torch.save(aug_subj.seg.tensor.squeeze(), f"{out_labels_path}/aug_{out_file}")

        subj.unload()
        del subj
        gc.collect()
        out_files.append(out_file)

    if out_dims == 2:
        for slice_dim in [0, 1, 2]:
            voxels = slice_infos[slice_dim]["n_slices"] * crop_size * crop_size
            slice_infos[slice_dim]["voxels"] = voxels
            sum_counts = sum(slice_infos[slice_dim]["counts"].values())
            slice_infos[slice_dim]["counts"][0] = voxels - sum_counts

    return out_files, slice_infos


def transform_volume(subj: tio.Subject, crop_size: int, quantile: float = .99):
    x = subj.mri.tensor.float()
    clamp_val = nonzero_quantile(x, quantile)
    empty_val = x.min()

    transform = tio.Compose([
        tio.Clamp(out_max=clamp_val),
        # Empty voxels are also normalized (but they are left out of mean and std calc)
        tio.ZNormalization(masking_method=lambda x: x != empty_val),
        tio.RescaleIntensity((-1, 1))  # Note: empty voxels now have intensity -1
    ])
    return transform(subj)


def update_dim_info(slice_dim: int, dim_info: dict, x: np.ndarray, y: np.ndarray,
                    num_classes: int, ignore_empty: bool, consider_fg: bool):
    """
    Updates the dim_info dict with the information needed to independently sample slices including
    information needed to oversample foreground or undersample empty slices.

    Note: dim_info is a dict and thus passed by reference, so the global dict is updated.
    """
    assert len(x.shape) == 3, x.shape
    assert len(y.shape) == 3, y.shape

    empty_val = np.min(x)
    assert empty_val == -1.

    acc = dim_info["n_slices"]
    dim_info["lens"].append(dim_info["n_slices"])

    # calculate empty slice ids if we need them
    if ignore_empty:
        axis = list(range(len(x.shape)))
        axis.remove(slice_dim)
        empty_idx = np.argwhere(np.max(x, axis=tuple(axis)) == empty_val).ravel() + acc
        dim_info["empty_idx"] = dim_info["empty_idx"] + empty_idx.tolist()

    # count amount of voxels pr. group
    for k in dim_info["counts"].keys():
        dim_info["counts"][k] += np.count_nonzero(y == k)

    # get id of fg slices
    if consider_fg:
        label_fg_idx = np.unique(np.nonzero(y)[slice_dim]) + acc
        dim_info["fg_idx"] = dim_info["fg_idx"] + label_fg_idx.tolist()

    # update length
    dim_info["n_slices"] += y.shape[slice_dim]


def nonzero_quantile(x, q):
    xf = x.flatten()
    nz = xf.nonzero().squeeze()
    x_nz = torch.index_select(xf, torch.tensor(0), nz)
    quantile = x_nz.quantile(q)
    del x_nz
    return quantile


def load_pickle_from(path):
    file = open(path, 'rb')
    data = pickle.load(file)
    file.close()
    return data


def store_pickle_to(path, data):
    file = open(path, 'wb')
    data = pickle.dump(data, file)
    file.close()
