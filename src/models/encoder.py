import torch.nn as nn
from .layers import StackedConvLayers


class UNetEncoder(nn.Module):
    def __init__(self, input_channels, base_num_features, num_blocks_per_stage, feature_mult,
                 num_stages, props, max_num_features=480):
        """
        This encoder includes the bottleneck layer!
        """
        super().__init__()

        self.props = props
        assert 'nonlin' in props.keys()

        self.stages = []
        self.stage_output_features = []
        self.stage_pool_kernel_size = []
        self.stage_conv_op_kernel_size = []
        self.num_blocks_per_stage = [num_blocks_per_stage] * num_stages

        pool_op_kernel_sizes = [(1, 1)] + [(2, 2)] * (num_stages - 1)
        conv_kernel_sizes = [(3, 3)] * num_stages

        assert len(pool_op_kernel_sizes) == len(conv_kernel_sizes)

        input_features = input_channels
        for stage in range(num_stages):
            output_features = min(int(base_num_features * feature_mult ** stage), max_num_features)
            kernel_size = conv_kernel_sizes[stage]
            pool_kernel_size = pool_op_kernel_sizes[stage]

            stage = StackedConvLayers(input_features, output_features, kernel_size, props,
                                      self.num_blocks_per_stage[stage], pool_kernel_size)

            self.stages.append(stage)
            self.stage_output_features.append(output_features)
            self.stage_conv_op_kernel_size.append(kernel_size)
            self.stage_pool_kernel_size.append(pool_kernel_size)

            input_features = output_features

        self.stages = nn.ModuleList(self.stages)
        self.output_features = output_features

    def forward(self, x, return_skips=True):
        skips = []

        for s in self.stages:
            x = s(x)
            if return_skips:
                skips.append(x)

        if return_skips:
            return skips
        else:
            return x
