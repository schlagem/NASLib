import os
import pickle
import numpy as np
import random
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as f

from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.primitives import AbstractPrimitive
from naslib.search_spaces.core.query_metrics import Metric


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


class DynamicConv2d(torch.nn.Module):

    def __init__(self, max_in_channels, kernel_size_list, stride=1, dilation=1):
        super().__init__()
        self.max_in_channels = max_in_channels
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation

        self.max_kernel = max(self.max_kernel_list)
        self.active_kernel_size = self.max_kernel

        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_in_channels,
            max(self.kernel_size_list),
            self.stride,
            groups=self.max_in_channels,  # this acts to each channel having multiple convs with in/out channels
            bias=False,
        )

    # TODO maybe function outside of class
    def sub_filter_start_end(self):
        center = self.max_kernel // 2
        dev = self.active_kernel_size // 2
        start, end = center - dev, center + dev + 1
        assert end - start == self.active_kernel_size
        return start, end

    def get_active_filters(self, out_channels, in_channels):
        start, end = self.sub_filter_start_end()
        filters = self.conv.weight[:out_channels, :in_channels, start:end, start:end]
        # TODO transformation needed outside of cutting out
        return filters

    def forward(self, x):
        # TODO figure out channels
        in_channels = x.shape[1]
        filters = self.get_active_filters(in_channels, self.expand_ratio*in_channels)
        padding = get_same_padding(self.active_kernel_size)
        return f.conv2d(x, filters, None, self.stride, padding, self.dilation, c)

class OFABlock(torch.nn.Module):

    def __init__(self, in_channels, kernel_size=[3, 5, 7], depth=[2, 3, 4], expand_ratio=[3, 4, 6]):
        super().__init__()
        self.kernel_size = max(kernel_size)
        self.depth = max(depth)
        self.expand_ratio = max(expand_ratio)

        self.kernel_list = kernel_size
        self.depth_list = depth
        self.expand_ratio_list = expand_ratio

        # TODO change to custom conv layer with variable depth and width
        # TODO how does expand ratio influence channels
        self.layer1 = DynamicConv2d(in_channels, out_channels, kernel_size)
        self.layer2 = DynamicConv2d(in_channels, out_channels, kernel_size)
        self.layer3 = DynamicConv2d(in_channels, out_channels, kernel_size)
        self.layer4 = DynamicConv2d(in_channels, out_channels, kernel_size)

        # TODO does this data structure make sense (list of class variables)
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]

    def forward(self, x):
        # Forward throuth the number of layers that are active (depth)
        for i in range(self.depth):
            x = f.relu(self.layers[i].forward(x))
        return x

    def set_max(self):
        self.kernel_size = max(self.kernel_list)
        self.depth = max(self.depth_list)
        self.expand_ratio = max(self.expand_ratio_list)

    def random_state(self):
        # TODO kernel size and expand ratio should be per layer not per block?
        self.kernel_size = np.random.choice(self.kernel_list)
        self.depth = np.random.choice(self.depth_list)
        self.expand_ratio = np.random.choice(self.expand_ratio_list)

    def mutate(self):
        mutation = np.random.choice(["kernel_size", "depth", "expand_ratio"])
        if mutation == "kernel_size":
            self.kernel_size = np.random.choice(self.kernel_list)
        elif mutation == "depth":
            self.depth = np.random.choice(self.depth_list)
        elif mutation == "expand_ratio":
            self.expand_ratio = np.random.choice(self.expand_ratio_list)
        else:
            raise ValueError(f"Mutation: {mutation} not found")


class OnceForAllSearchSpace(Graph):
    """
    Implementation of the Once for All Search space.
    """

    # it is queryable using the pretrained model
    # TODO come back to this
    QUERYABLE = True

    def __init__(self):
        super().__init__()

        """
        5 units 7371 = (3x3)^2 + (3x3)^3 + (3x3)^4 choices per layer 
            -Each unit {2,3,4} layers with 9 (3x3) choices 
                - Each layer kernel size {3,5,7}
                - Each layer increases number of channels by {3,4,6}
        """
        # TODO do we need preprocessing and post proccessing

        # Graph definition
        self.number_of_units = 5
        self.add_nodes_from(range(1, self.number_of_units + 1))
        self.add_edges_from([(i, i + 1) for i in range(1, self.number_of_units)])

        # similiar to init of
        # https://github.com/mit-han-lab/once-for-all/blob/master/ofa/imagenet_classification/elastic_nn/networks/ofa_mbv3.py
        self.kernel_size_list = [3, 5, 7],
        self.expand_ratio_list = [3, 4, 6],
        self.depth_list = [2, 3, 4]

        self.stride_stages = [1, 2, 2, 2, 1, 2]

        # the idea is to have 5 blocks each block is the same class
        # it takes as input the kernelsize, expand ratio and depth
        # it has a function to randomize
        for i in range(1, self.number_of_units + 1):
            # TODO figure out in_channels
            self.edges[i, i + 1].set("op", OFABlock(in_channels=3))

    def query(self, metric: Metric, dataset: str, path: str) -> float:
        #TODO use pretrained OFA to query validation and test perfomance
        raise NotImplementedError

    def get_hash(self):
        raise NotImplementedError

    def get_arch_iterator(self):
        raise NotImplementedError

    def sample_random_architecture(self, dataset_api=None):
        for i in range(1, self.number_of_units + 1):
            block = self.edges[i, i + 1].op
            block.random_state()

    def get_nbhd(self, dataset_api=None):
        raise NotImplementedError

    def get_type(self):
        return "ofa"
