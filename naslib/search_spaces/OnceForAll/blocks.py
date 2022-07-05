
import numpy as np
import torch
from collections import OrderedDict


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

from ofa.utils import MyNetwork, make_divisible, MyGlobalAvgPool2d
from ofa.utils import set_bn_param


from ..core.primitives import AbstractPrimitive


class FirstBlock(AbstractPrimitive):

    def __init__(self, input_channel, first_block_dim, stride, act_func, se):
        super().__init__(locals())
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

    def forward(self, x, edge_data):
        x = self.first_conv(x)
        x = self.first_block(x)
        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + "\n"
        _str += self.first_block.module_str + "\n"
        return _str

    def set_bn(self, momentum, eps):
        set_bn_param(self.first_block, momentum, eps)
        set_bn_param(self.first_conv, momentum, eps)

    def set_weights(self, conv_state_dict, block_state_dict):
        assert self.first_conv.state_dict().keys() == conv_state_dict.keys()
        assert len(self.first_conv.state_dict().keys()) == len(conv_state_dict.keys())
        self.first_conv.load_state_dict(conv_state_dict)

        assert self.first_block.state_dict().keys() == block_state_dict.keys()
        assert len(self.first_block.state_dict().keys()) == len(block_state_dict.keys())
        self.first_block.load_state_dict(block_state_dict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        d = super().state_dict()
        new_dict = OrderedDict([(k.replace("first_conv.", "block.0"), v) if "block.0" in k else (k, v) for k, v in d.items()])
        return new_dict

    def get_embedded_ops(self):
        return None


class FinalBlock(AbstractPrimitive):

    def __init__(self, feature_dim, final_expand_width, last_channel, n_classes, dropout_rate):
        super().__init__(locals())

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

    def forward(self, x, edge_data):
        x = self.final_expand_layer(x)
        x = x.mean(3, keepdim=True).mean(2, keepdim=True)
        x = self.feature_mix_layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.final_expand_layer.module_str + "\n"
        _str += self.feature_mix_layer.module_str + "\n"
        _str += self.classifier.module_str + "\n"
        return _str

    def set_bn(self, momentum, eps):
        set_bn_param(self.final_expand_layer, momentum, eps)
        set_bn_param(self.feature_mix_layer, momentum, eps)
        set_bn_param(self.classifier, momentum, eps)

    def set_weights(self, final_expand_dict, feature_mix_dict, classifier_dict):
        assert len(self.final_expand_layer.state_dict().keys()) == len(final_expand_dict.keys())
        assert self.final_expand_layer.state_dict().keys() == final_expand_dict.keys()
        self.final_expand_layer.load_state_dict(final_expand_dict)

        assert self.feature_mix_layer.state_dict().keys() == feature_mix_dict.keys()
        assert len(self.feature_mix_layer.state_dict().keys()) == len(feature_mix_dict.keys())
        self.feature_mix_layer.load_state_dict(feature_mix_dict)

        assert self.classifier.state_dict().keys() == classifier_dict.keys()
        assert len(self.classifier.state_dict().keys()) == len(classifier_dict.keys())
        self.classifier.load_state_dict(classifier_dict)

    def get_embedded_ops(self):
        return None


class OFABlock(AbstractPrimitive):

    def __init__(self, width, n_block, s, act_func, use_se, ks_list, expand_ratio_list, feature_dim, depth_list):
        super().__init__(locals())
        # the actual parameters of the OFABlock for every layer with #layers equal to self.depth
        self.depth_list = depth_list
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

    def forward(self, x, edge_data):
        for block in self.blocks[:self.depth]:
            x = block(x)
        return x

    @property
    def module_str(self):  # TODO care similiar named function in graph
        _str = ""
        for block in self.blocks[:self.depth]:
            _str += block.module_str + "\n"
        return _str

    def set_bn(self, momentum, eps):
        for b in self.blocks:
            set_bn_param(b, momentum, eps)

    def random_state(self):
        self.depth = np.random.choice(self.depth_list)
        for block in self.blocks:
            block.conv.active_kernel_size = int(np.random.choice(block.conv.kernel_size_list))
            block.conv.active_expand_ratio = int(np.random.choice(block.conv.expand_ratio_list))

    def mutate(self):
        mutation_type = np.random.choice(["depth", "kernel", "expand_ratio"])
        if mutation_type == "depth":
            choices = [d for d in self.depth_list if d != self.depth]
            self.depth = int(np.random.choice(choices))
        elif mutation_type == "kernel":
            block = np.random.choice(self.blocks[:self.depth])
            ks = block.conv.active_kernel_size
            choices = [k for k in block.conv.kernel_size_list if k != ks]
            block.conv.active_kernel_size = int(np.random.choice(choices))
        elif mutation_type == "expand_ratio":
            block = np.random.choice(self.blocks[:self.depth])
            er = block.conv.active_expand_ratio
            choices = [e for e in block.conv.expand_ratio_list if e != er]
            block.conv.active_expand_ratio = int(np.random.choice(choices))
        else:
            raise NotImplementedError(f"The mutation type {mutation_type} not supported")

    def set_weights(self, list_of_dicts):
        idx = 0
        for layer in self.blocks:
            assert len(layer.state_dict().keys()) == len(list_of_dicts[idx].keys())
            assert layer.state_dict().keys() == list_of_dicts[idx].keys()
            layer.load_state_dict(list_of_dicts[idx])
            idx += 1

    def get_embedded_ops(self):
        return None

