import numpy as np
import torch
from collections import OrderedDict
import copy

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
        new_dict = OrderedDict([(k.replace("first_conv.", "block.0"), v)
                                if "block.0" in k else (k, v) for k, v in d.items()])
        return new_dict

    def get_embedded_ops(self):
        return None

    def move_to(self, device):
        self.first_conv = self.first_conv.to(device)
        self.first_block = self.first_block.to(device)

    def size(self):
        param_size = 0.0
        for param in self.first_conv.parameters():
            param_size += param.nelement() * param.element_size()
        for param in self.first_block.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size / 1024 ** 2

    def params(self):
        return list(self.first_conv.parameters()) + list(self.first_block.parameters())


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

    def move_to(self, device):
        self.final_expand_layer = self.final_expand_layer.to(device)
        self.feature_mix_layer = self.feature_mix_layer.to(device)
        self.classifier = self.classifier.to(device)

    def size(self):
        param_size = 0.0
        for param in self.final_expand_layer.parameters():
            param_size += param.nelement() * param.element_size()
        for param in self.feature_mix_layer.parameters():
            param_size += param.nelement() * param.element_size()
        for param in self.classifier.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size / 1024 ** 2

    def params(self):
        return list(self.final_expand_layer.parameters()) + \
               list(self.feature_mix_layer.parameters()) + \
               list(self.classifier.parameters())


class OFAConv:

    def __init__(self, width, n_block, s, act_func, use_se, ks_list, expand_ratio_list, feature_dim):

        output_channel = width
        self.mobile_inverted_conv = DynamicMBConvLayer(
            in_channel_list=val2list(feature_dim),
            out_channel_list=val2list(output_channel),
            kernel_size_list=ks_list,
            expand_ratio_list=expand_ratio_list,
            stride=s,
            act_func=act_func,
            use_se=use_se,
        )
        if s == 1 and feature_dim == output_channel:
            shortcut = IdentityLayer(feature_dim, feature_dim)
        else:
            shortcut = None
        self.res_block = ResidualBlock(self.mobile_inverted_conv, shortcut)

    def move_to(self, device):
        self.res_block = self.res_block.to(device)


class OFALayer(AbstractPrimitive):

    def __init__(self, ofa_conv, active_kernel_size, active_expand_ratio):
        super().__init__(locals())
        self.ofa_conv = ofa_conv
        self.active_kernel_size = active_kernel_size
        self.active_expand_ratio = active_expand_ratio

    def forward(self, x, edge_data):
        # TODO this should probably be updated as soon as the operation is update
        self.ofa_conv.mobile_inverted_conv.active_kernel_size = self.active_kernel_size
        self.ofa_conv.mobile_inverted_conv.active_expand_ratio = self.active_expand_ratio
        x = self.ofa_conv.res_block(x)
        return x

    def get_embedded_ops(self):
        pass

    def set_weights(self, block_state_dict):
        assert self.ofa_conv.res_block.state_dict().keys() == block_state_dict.keys()
        assert len(self.ofa_conv.res_block.state_dict().keys()) == len(block_state_dict.keys())
        self.ofa_conv.res_block.load_state_dict(block_state_dict)

    def move_to(self, device):
        self.ofa_conv.move_to(device)

    def size(self):
        param_size = 0.0
        self.ofa_conv.mobile_inverted_conv.active_kernel_size = self.active_kernel_size
        self.ofa_conv.mobile_inverted_conv.active_expand_ratio = self.active_expand_ratio
        block = ResidualBlock(self.ofa_conv.res_block.conv.get_active_subnet(self.ofa_conv.res_block.conv.in_channels,
                                                                             preserve_weight=True),
                              copy.deepcopy(self.ofa_conv.res_block.shortcut))
        for param in block.parameters():
            param_size += param.nelement() * param.element_size()

        return param_size / 1024 ** 2

    def params(self):
        self.ofa_conv.mobile_inverted_conv.active_kernel_size = self.active_kernel_size
        self.ofa_conv.mobile_inverted_conv.active_expand_ratio = self.active_expand_ratio
        block = ResidualBlock(self.ofa_conv.res_block.conv.get_active_subnet(self.ofa_conv.res_block.conv.in_channels,
                                                                             preserve_weight=True),
                              copy.deepcopy(self.ofa_conv.res_block.shortcut))
        return self.ofa_conv.res_block.parameters()
