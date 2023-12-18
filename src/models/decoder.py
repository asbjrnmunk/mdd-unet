import torch
import torch.nn as nn
import numpy as np
from .layers import StackedConvLayers, Upsample

from enum import Enum


class Type(Enum):
    normal = 1  # Normal UNet with one encoder and one decoder
    head = 2    # Adversarial UNet. This decoder is the head, that is will produce predictions
    bottom = 3  # Adversarial UNet. This decoder will take features and output features


class UNetDecoder(nn.Module):
    def __init__(self, encoder, num_classes, type=Type.normal, depth=None, deep_supervision=False):
        super().__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        """
        We assume the bottleneck is part of the encoder, so we can start with upsample->concat here
        """
        num_blocks_per_stage = encoder.num_blocks_per_stage[:-1][::-1]

        # in total, we have one less stages compared to the encoder the last stage of the encoder is
        # the bottleneck
        total_num_stages = len(encoder.stages) - 1
        assert depth <= total_num_stages

        if type == Type.normal:
            assert depth is None
            first_stage = 0
            last_stage = total_num_stages
            self.apply_segmentation = True
        elif type == Type.bottom:
            assert depth is not None
            first_stage = depth
            last_stage = total_num_stages
            self.apply_segmentation = False
        elif type == Type.head:
            assert depth is not None
            first_stage = 0
            last_stage = depth
            self.apply_segmentation = True
        else:
            raise ValueError("Argument `type` is not valid")

        self.tus = []
        self.stages = []
        self.deep_supervision_outputs = []

        cum_upsample = np.cumprod(np.vstack(encoder.stage_pool_kernel_size), axis=0).astype(int)
        stage_idx = list(reversed(range(first_stage, last_stage)))

        for i, s in enumerate(stage_idx):
            features_below = encoder.stage_output_features[s + 1]
            features_skip = encoder.stage_output_features[s]

            self.tus.append(nn.ConvTranspose2d(features_below, features_skip,
                                               encoder.stage_pool_kernel_size[s + 1],
                                               encoder.stage_pool_kernel_size[s + 1], bias=False))

            # after we tu we concat features so now we have 2xfeatures_skip
            self.stages.append(StackedConvLayers(2 * features_skip, features_skip,
                                                 encoder.stage_conv_op_kernel_size[s],
                                                 encoder.props, num_blocks_per_stage[i]))

            if deep_supervision and s != 0:
                upsample = Upsample(scale_factor=cum_upsample[s], mode="bilinear")
                layer_seg_out = nn.Conv2d(features_skip, num_classes, 1, 1, 0, 1, 1, bias=True)
                self.deep_supervision_outputs.append(nn.Sequential(layer_seg_out, upsample))

        if self.apply_segmentation:
            self.segmentation_output = nn.Conv2d(encoder.stage_output_features[0], num_classes,
                                                 1, 1, 0, 1, 1, bias=True)

        self.tus = nn.ModuleList(self.tus)
        self.stages = nn.ModuleList(self.stages)
        self.deep_supervision_outputs = nn.ModuleList(self.deep_supervision_outputs)

    def forward(self, x, skips, gt=None, loss=None):
        # skips are ordered from top to bottom. We need them reversed.
        skips = skips[::-1]
        seg_outputs = []

        for i in range(len(self.tus)):
            x = self.tus[i](x)
            x = torch.cat((x, skips[i]), dim=1)
            x = self.stages[i](x)
            if self.deep_supervision and (i != len(self.tus) - 1):
                tmp = self.deep_supervision_outputs[i](x)
                if gt is not None:
                    tmp = loss(tmp, gt)
                seg_outputs.append(tmp)

        if self.apply_segmentation:
            x = self.segmentation_output(x)

        if self.deep_supervision:
            tmp = x
            if gt is not None:
                tmp = loss(tmp, gt)
            seg_outputs.append(tmp)

            # seg_outputs are ordered so that the seg from the highest layer is first, the seg from
            return seg_outputs[::-1]
            # the bottleneck of the UNet last
        else:
            return x
