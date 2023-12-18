import torch
from typing import Iterator
from torch.utils.data import Sampler


class Oversampler(Sampler):
    def __init__(self, dataset, p=None, ignore_empty=False):
        """
        Oversample dataset with higher probability of getting foreground slices
        while ignoring empty slices.

        Params:
            p: probability of sampling fg id. If p is None, fg slices will not be prefered.
            ignore_empty: whether to ignore empty slices
        """
        self.num_samples = dataset.epoch_size
        self.replacement = True if dataset.epoch_size > dataset.len else False

        fg_idx = dataset.fg_idx
        empty_idx = dataset.empty_idx

        N = dataset.len  # amount of non-augmented elements
        n = N
        n -= len(empty_idx) if ignore_empty else 0

        n_fg = len(fg_idx)
        n_bg = n - n_fg

        if p is None:
            p = 1 / n
            self.weights = torch.ones(N) * p
        else:
            assert n_fg > 0, "We need fg slices to oversample them, sigh."

            p_fg = (p / n_fg)  # probability for each fg id to be sampled
            p_bg = ((1 - p) / n_bg)  # probability for each non-empty bg id to be sampled

            self.weights = torch.ones(N) * p_bg
            self.weights[fg_idx] = p_fg

        if ignore_empty:
            self.weights[empty_idx] = 0  # ignore empty slices

        # assert weights are probability distribution
        sum = torch.sum(self.weights)
        assert sum - 1 < 1e-5, f"Sum is not 1!: {sum}"

    def __iter__(self) -> Iterator[int]:
        yield from torch.multinomial(
            self.weights, replacement=self.replacement, num_samples=self.num_samples
        ).tolist()

    def __len__(self) -> int:
        return self.num_samples
