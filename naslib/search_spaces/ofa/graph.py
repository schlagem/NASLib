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


class OFABlock(torch.nn.Module):

    def __init__(self, in_channel, kernel_size=[3, 5, 7], depth=[2, 3, 4], expand_ratio=[3, 4, 6]):
        super().__init__()
        self.kernel_size = max(kernel_size)
        self.depth = max(depth)
        self.expand_ratio = max(expand_ratio)

        self.kernel_list = kernel_size
        self.depth_list = depth
        self.expand_ratio_list = expand_ratio

        # TODO change to custom conv layer with variable depth and width
        # TODO how does expand ratio influence channels
        self.layer1 = torch.nn.Conv2d(in_channel, out_channels, kernel_size)
        self.layer2 = torch.nn.Conv2d(in_channel, out_channels, kernel_size)
        self.layer3 = torch.nn.Conv2d(in_channel, out_channels, kernel_size)

    def forward(self, x):
        x = f.relu(self.layer1(x))
        x = f.relu(self.layer2(x))
        x = f.relu(self.layer3(x))

    def random_state(self):
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
        number_of_units = 5
        self.add_nodes_from(range(1, number_of_units + 1))
        self.add_edges_from([(i, i + 1) for i in range(1, number_of_units)])

        # similiar to init of
        # https://github.com/mit-han-lab/once-for-all/blob/master/ofa/imagenet_classification/elastic_nn/networks/ofa_mbv3.py
        self.kernel_size_list = [3, 5, 7],
        self.expand_ratio_list = [3, 4, 6],
        self.depth_list = [2, 3, 4]

        self.stride_stages = [1, 2, 2, 2, 1, 2]

        # TODO add operations
        # the idea is to have 5 blocks each block is the same class
        # it takes as input the kernelsize, expand ratio and depth
        # it has a function to randomize
        raise NotImplementedError


    def query(self, metric: Metric, dataset: str, path: str) -> float:
        #TODO use pretrained OFA to query validation and test perfomance
        raise NotImplementedError

    def get_hash(self):
        raise NotImplementedError

    def get_arch_iterator(self):
        raise NotImplementedError

    def sample_random_architecture(self, dataset_api=None):
        # TODO iterate thorugh edges call random state for the arch
        raise NotImplementedError

    def get_nbhd(self, dataset_api=None):
        raise NotImplementedError

    def get_type(self):
        return "ofa"
