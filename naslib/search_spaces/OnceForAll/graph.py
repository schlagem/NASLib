# naslib imports
from naslib.search_spaces.OnceForAll.blocks import FinalBlock, OFABlock, FirstBlock, OFAConv, OFALayer
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.core import primitives as ops

# Once for all imports
from ofa.utils import make_divisible, val2list, MyNetwork
from ofa.utils import download_url
from ofa.utils import MyNetwork, make_divisible, MyGlobalAvgPool2d

# other imports
import numpy as np
import torch
from itertools import product
from collections import OrderedDict


class OnceForAllSearchSpace(Graph):
    """
    Implementation of the Once for All Search space.
    """
    QUERYABLE = True

    def __init__(self):
        super().__init__()

        # we don't call the search space with params thus we define them here
        # for our application this should suffice
        n_classes = 1000  # ImageNet
        dropout_rate = 0.1  # default Paremeter TODO check if better value
        bn_param = (0.1, 1e-5)  # TODO check where this is used

        self.width_mult = 1.0
        self.ks_list = val2list(3, 1)
        self.expand_ratio_list = val2list(6, 1)
        self.depth_list = val2list(4, 1)

        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()

        base_stage_width = [16, 16, 24, 40, 80, 112, 160, 960, 1280]

        final_expand_width = make_divisible(
            base_stage_width[-2] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE
        )
        last_channel = make_divisible(
            base_stage_width[-1] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE
        )

        self.ks_list = [3, 5, 7]
        self.expand_ratio = [3, 4, 6]
        self.depth_list = [2, 3, 4]

        stride_stages = [1, 2, 2, 2, 1, 2]
        act_stages = ["relu", "relu", "relu", "h_swish", "h_swish", "h_swish"]
        se_stages = [False, False, True, False, True, True]
        n_block_list = [1] + [max(self.depth_list)] * 5
        width_list = []
        for base_width in base_stage_width[:-2]:
            width = make_divisible(
                base_width * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE
            )
            width_list.append(width)

        input_channel, first_block_dim = width_list[0], width_list[1]

        """
        Build the search space
        """
        # 1 + 5 * 6 + 1
        self.name = "ofa"

        # node types
        self.depth_nodes = [7,13,19,25,31]
        self.block_start_nodes = [2,8,14,20,26]

        #
        # nodes
        #

        self.add_nodes_from([i for i in range(1, 33)])

        #
        # edges
        #

        # edges preprocessing and postprocessing
        self.add_edges_from([(1, 2), (31, 32)])

        # edges between blocks, always identity
        self.add_edges_from([(i, i+1, EdgeData().finalize()) for i in self.depth_nodes[:-1]])

        # intermediate edges
        for i in self.block_start_nodes:
            self.add_edges_from([(i+k, i+k+1) for k in range(4)])

        # edges different depths
        self.add_edges_from([(i - 3, i) for i in self.depth_nodes])
        self.add_edges_from([(i - 2, i) for i in self.depth_nodes])
        self.add_edges_from([(i - 1, i) for i in self.depth_nodes])

        #
        # assign ops to edges
        #

        # preprocessing, first conv layer
        first_block = FirstBlock(input_channel, first_block_dim, stride_stages[0], act_stages[0], se_stages[0])
        self.edges([1, 2]).set("op", first_block)

        # intermediate nodes
        feature_dim = first_block_dim

        for start_node, width, n_block, s, act_func, use_se in zip(
            self.block_start_nodes,
            width_list[2:],
            n_block_list[1:],
            stride_stages[1:],
            act_stages[1:],
            se_stages[1:],
        ):
            # make an OFA layer object for all 9 edges cause of weight sharing?
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                ofa_conv = OFAConv(width, n_block, s, act_func, use_se, self.ks_list, self.expand_ratio, feature_dim)

                ofa_layer_list = []
                for ks, er in product(self.ks_list, self.expand_ratio):
                    ofa_layer_list.append(OFALayer(ofa_conv, ks, er))

                self.edges([start_node+i, start_node+i+1]).set("op", ofa_layer_list)

                feature_dim = output_channel

        # postprocessing
        final_block = FinalBlock(feature_dim, final_expand_width, last_channel, n_classes, dropout_rate)
        self.edges[(31, 32)].set("op", final_block)

        # depth layers: identity and zero
        for i in self.depth_nodes:
            for j in range(1, 4):
                self.edges([i - j, i]).set(
                    "op",
                    [
                        ops.Identity(),
                        ops.Zero(stride=1)  # we need to set stride to one
                    ])

        # edges between blocks
        for i in self.depth_nodes[:-1]:
            self.edges([i, i + 1]).set("op", ops.Identity())

        # set bn param
        # TODO this doesnt work iterate all modules and call this function
        # TODO IMPLEMENT THIS FUNCTION
        # self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth
        # doesn't work vanilla
        # TODO fix if needed dont know what it does
        # self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]


    def mutate(self):
        pass

    def sample_random_architecture(self, dataset_api):
        pass

