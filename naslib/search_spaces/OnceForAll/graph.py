# naslib imports
from naslib.search_spaces.OnceForAll.blocks import FinalBlock, OFABlock, FirstBlock
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.query_metrics import Metric

# Once for all imports
from ofa.utils import make_divisible, val2list, MyNetwork
from ofa.utils import download_url

# other imports
import numpy as np
import torch
from collections import OrderedDict


class OnceForAllSearchSpace(Graph):
    # TODO the original implemetnaion inherites form MobileNetV3 look into possible problems
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
        # Graph definition
        # similar to init only in graph version:
        # https://github.com/mit-han-lab/once-for-all/blob/master/ofa/imagenet_classification/elastic_nn/networks/ofa_mbv3.py

        # we don't call the search space with params thus we define them here
        # for our application this should suffice
        n_classes = 1000  # ImageNet
        dropout_rate = 0.1  # default Paremeter TODO check if better value
        bn_param = (0.1, 1e-5)  # TODO check where this is used
        # width_mult = 1.0  TODO check what this does
        self.width_mult = 1.0

        base_stage_width = [16, 16, 24, 40, 80, 112, 160, 960, 1280]

        final_expand_width = make_divisible(
            base_stage_width[-2] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE
        )
        last_channel = make_divisible(
            base_stage_width[-1] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE
        )

        """
                Default params but are in each call overwritten with see below
                ks_list = 3
                expand_ratio_list = 6
                depth_list = 4
        """
        self.ks_list = [3, 5, 7]
        self.expand_ratio = [3, 4, 6]
        self.depth_list = [2, 3, 4]

        self.stride_stages = [1, 2, 2, 2, 1, 2]
        act_stages = ["relu", "relu", "relu", "h_swish", "h_swish", "h_swish"]
        stride_stages = [1, 2, 2, 2, 1, 2]
        se_stages = [False, False, True, False, True, True]
        n_block_list = [1] + [max(self.depth_list)] * 5
        width_list = []
        for base_width in base_stage_width[:-2]:
            width = make_divisible(
                base_width * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE
            )
            width_list.append(width)

        input_channel, first_block_dim = width_list[0], width_list[1]

        # first block (kinda of like preprocessing)
        # Not variable in depht width or expand ratio
        self.add_nodes_from([1, 2])
        first_block = FirstBlock(input_channel, first_block_dim, stride_stages[0], act_stages[0], se_stages[0])
        self.add_edges_from([(1, 2)])
        self.edges[1, 2].set("op", first_block)

        # The next 5 blocks are the ones where the number of layers, number of channels and kernel size can be changed
        self.number_of_units = 5  # the number of dynamic units
        self.offset = 1  # refactor for loops
        self.add_nodes_from(range(2, 2 + self.number_of_units))
        self.add_edges_from([(i, i + 1) for i in range(2, 2 + self.number_of_units)])

        # dimension
        feature_dim = first_block_dim

        i = 2
        for width, n_block, s, act_func, use_se in zip(
                width_list[2:],
                n_block_list[1:],
                stride_stages[1:],
                act_stages[1:],
                se_stages[1:],
        ):
            unit = OFABlock(width, n_block, s, act_func, use_se, self.ks_list, self.expand_ratio, feature_dim,
                            self.depth_list)
            self.edges[i, i + 1].set("op", unit)
            feature_dim = unit.max_channel
            i += 1

        self.add_nodes_from([2 + self.number_of_units + 1])
        self.add_edges_from([(2 + self.number_of_units, 2 + self.number_of_units + 1)])

        final_block = FinalBlock(feature_dim, final_expand_width, last_channel, n_classes, dropout_rate)

        self.edges[2 + self.number_of_units, 2 + self.number_of_units + 1].set("op", final_block)

        # set bn param
        # TODO this doesnt work iterate all modules and call this function
        # self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth
        # doesn't work vanilla
        # TODO fix if needed dont know what it does
        # self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

    def _set_weights(self):
        net_id = "ofa_mbv3_d234_e346_k357_w1.0"
        url_base = "https://hanlab.mit.edu/files/OnceForAll/ofa_nets/"
        init = torch.load(
            download_url(url_base + net_id, model_dir=".pth/ofa_nets"),
            map_location="cpu",
            )["state_dict"]
        keys = init.keys()
        # TODO finish applying weights to OFA net space

        # First unit weights
        first_unit = self.edges[1, 2].op
        first_conv_state = OrderedDict((k.replace("first_conv.", ""), init[k]) for k in keys if "first_conv" in k)
        first_block_state = OrderedDict((k.replace("blocks.0.mobile_inverted_conv", "conv"), init[k]) for k in keys
                                        if "blocks.0" in k)
        first_unit.set_weights(first_conv_state, first_block_state)

        # TODO 5 Dynamics units
        raise NotImplementedError

        # TODO Last unit
        raise NotImplementedError

    def query(self, metric: Metric, dataset: str, path: str) -> float:
        # https://github.com/mit-han-lab/once-for-all/blob/master/ofa/tutorial/imagenet_eval_helper.py
        # TODO generate validation and test pipeline
        # TODO maybe there are already value somewhere (only found them for resnet space)
        raise NotImplementedError

    def get_hash(self):
        raise NotImplementedError

    def get_arch_iterator(self):
        raise NotImplementedError

    def sample_random_architecture(self, dataset_api=None):
        for i in range(1 + self.offset, self.number_of_units + self.offset + 1):
            block = self.edges[i, i + 1].op
            block.random_state()

    def mutate(self):
        # choose one unit
        block_idx = np.random.choice([i for i in range(1 + self.offset, self.number_of_units + self.offset + 1)])
        block = self.edges[block_idx, block_idx + 1].op
        # in that unit one property gets mutated
        block.mutate()

    def get_nbhd(self, dataset_api=None):
        raise NotImplementedError

    def get_type(self):
        return "OnceForAll"
