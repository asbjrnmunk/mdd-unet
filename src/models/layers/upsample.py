import torch.nn as nn


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x,
                                         size=self.size,
                                         scale_factor=self.scale_factor,
                                         mode=self.mode,
                                         align_corners=self.align_corners)
