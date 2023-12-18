import pytorch_lightning as pl
from . import MRIDataset, Oversampler
from torch.utils.data import DataLoader
import torchio as tio


class MRIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        source: str,
        val: str,
        targets: list[str],
        epoch_size: int = None,
        slice_dim: int = 1,
        crop_size: int = 256,
        prob_fg: float = None,
        ignore_empty: bool = False,
        batch_size: int = 10,
        use_augmentations: bool = False,
        p_blur_aug: float = 0.25,
        p_noise_aug: float = 0.25,
        p_gamma_aug: float = 0.25,
        p_rot_aug: float = 0.25,
        p_elastic_aug: float = .0,
        subfolder: str = "preprocessed",
        force_preprocess: bool = False
    ):
        """
        [train|val|test]path:  path to images and labels
        epoch_size: number of samples per epoch. Use `none` for dataset size.
        prob_fg: percentages of samples which should include foreground
        """
        super().__init__()

        self.train_path = f"data/{source}"
        self.val_path = f"data/{val}"
        self.test_paths = [f"data/{target}" for target in targets]
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.slice_dim = slice_dim
        self.crop_size = crop_size
        self.prob_fg = prob_fg
        self.ignore_empty = ignore_empty
        self.ce_weights = None

        self.use_augmentations = use_augmentations
        self.p_blur_aug = p_blur_aug
        self.p_noise_aug = p_noise_aug
        self.p_rot_aug = p_rot_aug
        self.p_gamma_aug = p_gamma_aug
        self.p_elastic = p_elastic_aug  # TODO: Use?

        self.subfolder = subfolder
        self.force_preprocess = force_preprocess

    def setup(self, stage: str):
        if stage == "fit":
            self.source_set = MRIDataset(
                self.train_path,
                subfolder=self.subfolder,
                out_dims=2,
                epoch_size=self.epoch_size,
                slice_dim=self.slice_dim,
                crop_size=self.crop_size,
                online_augmentation=self.augmentation() if self.use_augmentations else None,
                ignore_empty=self.ignore_empty,
                consider_fg=self.prob_fg is not None,
                force_preprocess=self.force_preprocess
            )

            self.target_set = MRIDataset(
                self.val_path,
                subfolder=self.subfolder,
                out_dims=2,
                epoch_size=self.epoch_size,
                slice_dim=self.slice_dim,
                crop_size=self.crop_size,
                ignore_empty=self.ignore_empty,
                consider_fg=self.prob_fg is not None,
                force_preprocess=self.force_preprocess
            )

            self.val_set = MRIDataset(
                self.val_path,
                subfolder=self.subfolder,
                out_dims=3,
                slice_dim=self.slice_dim,
                crop_size=self.crop_size,
                force_preprocess=self.force_preprocess
            )
            self.ce_weights = self.source_set.ce_weights

        if stage == "test":
            self.test_sets = [
                MRIDataset(
                    test_path,
                    out_dims=3,
                    subfolder=self.subfolder,
                    slice_dim=self.slice_dim,
                    crop_size=self.crop_size,
                    force_preprocess=self.force_preprocess
                )
                for test_path in self.test_paths
            ]

    def train_dataloader(self):
        source_loader = DataLoader(
            self.source_set,
            batch_size=self.batch_size,
            sampler=self.sampler(self.source_set),
            num_workers=4
        )

        target_loader = DataLoader(
            self.target_set,
            batch_size=self.batch_size,
            sampler=self.sampler(self.target_set),
            num_workers=4
        )

        return {'source': source_loader, 'target': target_loader}

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, num_workers=4)

    def test_dataloader(self):
        return [DataLoader(test_set, batch_size=1, num_workers=4) for test_set in self.test_sets]

    def augmentation(self):
        augmentations = {
            tio.RandomGamma(): self.p_gamma_aug,
            tio.RandomBlur(std=1): self.p_blur_aug,
            tio.RandomNoise(std=0.05): self.p_noise_aug,
            tio.RandomAffine(scales=0, degrees=(0, 0, -30, 30, 0, 0), translation=0): self.p_rot_aug
        }
        return tio.Compose([tio.OneOf(augmentations, p=0.8)])

    def sampler(self, set):
        return Oversampler(set, self.prob_fg, ignore_empty=self.ignore_empty)
