from pytest import approx
from datasets import MRIDataset, Oversampler

import pytest


def test_can_shuffle():
    source = "lpba40"
    train_path = f"data/{source}"

    dataset = MRIDataset(
        train_path,
        subfolder="preprocessed",
        out_dims=2,
        slice_dim=1,
        ignore_empty=False,
        consider_fg=False,
        crop_size=256,
    )

    num_samples = len(dataset)

    all_ids = list(range(num_samples))
    sampled_ids = Oversampler(dataset)

    assert len(all_ids) == len(list(sampled_ids))
    assert all_ids[0] != list(sampled_ids)[0]
    assert len(set(all_ids).difference(set(sampled_ids))) == 0


def test_can_oversample_with_replacement():
    source = "lpba40"
    train_path = f"data/{source}"

    epoch_size = 10000
    dataset = MRIDataset(
        train_path,
        subfolder="preprocessed",
        out_dims=2,
        slice_dim=1,
        ignore_empty=False,
        consider_fg=False,
        crop_size=256,
        epoch_size=epoch_size,
    )

    assert dataset.len < epoch_size

    sampled_ids = Oversampler(dataset)

    all_ids = list(range(dataset.len))

    assert len(list(sampled_ids)) == epoch_size
    assert max(sampled_ids) < dataset.len

    # bin counts of sampled ids will be normally distributed with a mean of 1 occurence, so a
    # siginifcant amount of ids will not be sampled in each epoch. We allow 33%, which is about
    # right.
    assert len(set(all_ids).difference(set(sampled_ids))) < (1/3 * dataset.len)


def test_can_ignore_empty():
    source = "lpba40"
    train_path = f"data/{source}"

    dataset = MRIDataset(
        train_path,
        subfolder="preprocessed",
        out_dims=2,
        slice_dim=1,
        ignore_empty=True,
        consider_fg=False,
        crop_size=256,
    )

    assert dataset.epoch_size < dataset.len
    assert dataset.empty_idx != []

    sampled_idx = Oversampler(dataset, ignore_empty=True)

    all_idx = set(range(dataset.len))
    empty_idx = set(dataset.empty_idx)
    non_empty_idx = all_idx.difference(empty_idx)

    assert len(empty_idx) > 0
    assert len(list(sampled_idx)) == len(non_empty_idx)
    assert len(set(sampled_idx).intersection(empty_idx)) == 0
    assert len(set(sampled_idx).intersection(non_empty_idx)) == len(non_empty_idx)


def test_can_oversample_foreground():
    source = "lpba40"
    epoch_size = 10000

    dataset = MRIDataset(
        f"data/{source}",
        subfolder="preprocessed",
        out_dims=2,
        slice_dim=1,
        ignore_empty=False,
        consider_fg=True,
        crop_size=256,
        epoch_size=epoch_size
    )

    assert dataset.epoch_size > dataset.len
    assert dataset.fg_idx != []

    p_fg = 0.2
    sampled_idx = Oversampler(dataset, p=p_fg)

    fg_idx = set(dataset.fg_idx)
    sampled_fg_idx = [i for i in list(sampled_idx) if i in fg_idx]

    assert len(list(sampled_idx)) == epoch_size
    assert len(sampled_fg_idx) / epoch_size == approx(p_fg, abs=0.02)  # so p_hat in [0.18, 0.22]
    assert len(set(sampled_fg_idx)) / len(fg_idx) > 0.8


@pytest.mark.parametrize("slice_dim", [0, 1, 2])
def test_all_ids_are_valid(slice_dim):
    dataset = MRIDataset(
        "data/lpba40",
        subfolder="preprocessed",
        out_dims=2,
        slice_dim=slice_dim,
        ignore_empty=True,
        consider_fg=True,
        crop_size=256,
        epoch_size=10000
    )

    idx = Oversampler(dataset, p=None, ignore_empty=True)

    for i in list(idx)[:1000]:
        x, y = dataset[i]
        assert x is not None
