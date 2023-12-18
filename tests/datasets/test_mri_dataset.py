from datasets import MRIDataset
import torchio as tio


def test_2d():
    source = "lpba40"
    train_path = f"data/{source}"

    dataset = MRIDataset(
        train_path,
        subfolder="preprocessed",
        out_dims=2,
        slice_dim=1,
        ignore_empty=True,
        consider_fg=True,
        crop_size=256,
        force_preprocess=True
    )

    assert len(dataset.fg_idx) == 1163
    assert len(dataset.empty_idx) == 1738
    assert dataset.counts == {0: 568531379, 1: 164817, 2: 156284}
    assert dataset.lens == _lpba40_lens()
    assert dataset.len == 8680
    assert dataset.epoch_size == dataset.len - len(dataset.empty_idx)

    x, y = dataset[0]

    assert x.shape == (1, 256, 256)
    assert y.shape == (256, 256)

    assert x.max() <= 1
    assert x.min() >= -1
    assert all([label in [0, 1, 2] for label in y.unique().ravel()])


def test_2d_aug():
    source = "lpba40"
    train_path = f"data/{source}"

    dataset = MRIDataset(
        train_path,
        subfolder="preprocessed",
        out_dims=2,
        slice_dim=1,
        ignore_empty=True,
        online_augmentation=tio.RandomFlip(),
        consider_fg=True,
        crop_size=256,
        force_preprocess=False
    )

    x, y = dataset[0]

    assert x.shape == (1, 256, 256)
    assert y.shape == (256, 256)

    assert x.max() <= 1
    assert x.min() >= -1
    assert all([label in [0, 1, 2] for label in y.unique().ravel()])


def test_3d():
    source = "hammers_tiny"
    train_path = f"data/{source}"

    dataset = MRIDataset(
        train_path,
        subfolder="preprocessed",
        out_dims=3,
        slice_dim=1,
        crop_size=256,
        force_preprocess=True
    )

    assert dataset.len == 4
    assert dataset.epoch_size == dataset.len

    x, y = dataset[0]

    assert x.shape == (256, 198, 256)
    assert y.shape == (256, 198, 256)

    assert x.max() == 1
    assert x.min() == -1
    assert x[0, 0, 0] == -1

    assert y.unique(sorted=True).tolist() == [0, 1, 2]
    assert y[0, 0, 0] == 0


# Test slice_dim not 1
def test_3d_slice_dim():
    source = "hammers_tiny"
    train_path = f"data/{source}"

    dataset = MRIDataset(
        train_path,
        subfolder="preprocessed",
        out_dims=3,
        slice_dim=2,
        crop_size=256,
        force_preprocess=True
    )

    assert dataset.len == 4
    assert dataset.epoch_size == dataset.len

    x, y = dataset[0]

    assert x.shape == (256, 256, 170)
    assert y.shape == (256, 256, 170)

    assert x.max() == 1
    assert x.min() == -1
    assert x[0, 0, 0] == -1

    assert y.unique(sorted=True).tolist() == [0, 1, 2]
    assert y[0, 0, 0] == 0


def _lpba40_lens():
    return [
        0, 217, 434, 651, 868, 1085, 1302, 1519, 1736, 1953, 2170, 2387, 2604, 2821, 3038, 3255,
        3472, 3689, 3906, 4123, 4340, 4557, 4774, 4991, 5208, 5425, 5642, 5859, 6076, 6293, 6510,
        6727, 6944, 7161, 7378, 7595, 7812, 8029, 8246, 8463
    ]
