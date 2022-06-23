
import numpy as np
import torch

from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import (
    DynamicMBConvLayer,
)
from ofa.utils.layers import (
    ConvLayer,
    IdentityLayer,
    LinearLayer,
    MBConvLayer,
    ResidualBlock,
)
from ofa.imagenet_classification.networks import MobileNetV3
from ofa.utils import make_divisible, val2list, MyNetwork


class FirstBlock(torch.nn.Module):

    def __init__(self, input_channel, first_block_dim, stride, act_func, se):
        super().__init__()
        self.first_conv = ConvLayer(
            3, input_channel, kernel_size=3, stride=2, act_func="h_swish"
        )

        first_block_conv = MBConvLayer(
            in_channels=input_channel,
            out_channels=first_block_dim,
            kernel_size=3,
            stride=stride,
            expand_ratio=1,
            act_func=act_func,
            use_se=se,
        )
        self.first_block = ResidualBlock(
            first_block_conv,
            IdentityLayer(first_block_dim, first_block_dim)
            if input_channel == first_block_dim
            else None,
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.first_block(x)
        return x


class FinalBlock(torch.nn.Module):

    def __init__(self, feature_dim, final_expand_width, last_channel, n_classes, dropout_rate):
        super().__init__()

        self.final_expand_layer = ConvLayer(
            feature_dim, final_expand_width, kernel_size=1, act_func="h_swish"
        )
        self.feature_mix_layer = ConvLayer(
            final_expand_width,
            last_channel,
            kernel_size=1,
            bias=False,
            use_bn=False,
            act_func="h_swish",
        )

        self.classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

    def forward(self, x):
        x = self.final_expand_layer(x)
        x = x.mean(3, keepdim=True).mean(2, keepdim=True)
        x = self.feature_mix_layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class OFABlock(torch.nn.Module):

    def __init__(self, width, n_block, s, act_func, use_se, ks_list, expand_ratio_list, feature_dim):
        super().__init__()
        # the actual parameters of the OFABlock for every layer with #layers = self.depth
        self.max_channel = None
        self.depth = n_block
        self.blocks = []  # maybe there better name
        output_channel = width
        for i in range(n_block):
            if i == 0:
                stride = s
            else:
                stride = 1
            mobile_inverted_conv = DynamicMBConvLayer(
                in_channel_list=val2list(feature_dim),
                out_channel_list=val2list(output_channel),
                kernel_size_list=ks_list,
                expand_ratio_list=expand_ratio_list,
                stride=stride,
                act_func=act_func,
                use_se=use_se,
            )
            if stride == 1 and feature_dim == output_channel:
                shortcut = IdentityLayer(feature_dim, feature_dim)
            else:
                shortcut = None
            self.blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
            feature_dim = output_channel
        self.max_channel = feature_dim

    def forward(self, x):
        for i in range(self.depth):  # TODO is here +1 needed
            x = self.blocks[i](x)
        return x

    def random_state(self):
        #TODO self.kernel_list is not defined before
        #TODO self.expand_ratio is not defined before
        self.kernel_size_list = np.random.choice(self.kernel_list_options, size=4)
        self.depth = np.random.choice(self.depth_list_options)
        self.expand_ratio = np.random.choice(self.expand_ratio_list_options, size=4)

    def mutate(self):
        mutation = np.random.choice(["kernel_size", "depth", "expand_ratio"])
        #TODO self.kernel_list is not defined before
        #TODO self.expand_ratio is not defined before
        if mutation == "kernel_size":
            self.kernel_size_list = np.random.choice(self.kernel_list_options, size=4)
        elif mutation == "depth":
            self.depth = np.random.choice(self.depth_list_options)
        elif mutation == "expand_ratio":
            self.expand_ratio_list = np.random.choice(self.expand_ratio_list_options, size=4)
        else:
            raise ValueError(f"Mutation: {mutation} not found")

