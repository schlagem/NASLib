# naslib imports
from naslib.search_spaces.OnceForAll.blocks import FinalBlock, FirstBlock, OFAConv, OFALayer
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.core import primitives as ops
from naslib.utils.utils import load_config

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

        base_stage_width = [16, 16, 24, 40, 80, 112, 160, 960, 1280]

        final_expand_width = make_divisible(
            base_stage_width[-2] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE
        )
        last_channel = make_divisible(
            base_stage_width[-1] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE
        )

        self.ks_list = [3, 5, 7]
        self.expand_ratio_list = [3, 4, 6]
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
        self.depth_nodes = [7, 13, 19, 25, 31]
        self.block_start_nodes = [2, 8, 14, 20, 26]

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
        self.add_edges_from([(i, i+1) for i in self.depth_nodes[:-1]])

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
        self.edges[1, 2].set("op", first_block)

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
            output_channel = width
            # make an OFA layer object for all 9 edges cause of weight sharing?
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                ofa_conv = OFAConv(width, n_block, stride, act_func, use_se, self.ks_list, max(self.expand_ratio_list),
                                   feature_dim)

                ofa_layer_list = []
                for ks, er in product(self.ks_list, self.expand_ratio_list):
                    ofa_layer_list.append(OFALayer(ofa_conv, ks, er))

                self.edges[start_node+i, start_node+i+1].set("op", ofa_layer_list)
                self._set_op_indice(self.edges[start_node+i, start_node+i+1], 8)

                feature_dim = output_channel

        # postprocessing
        final_block = FinalBlock(feature_dim, final_expand_width, last_channel, n_classes, dropout_rate)
        self.edges[(31, 32)].set("op", final_block)

        # depth layers: identity and zero
        for i in self.depth_nodes:
            for j in range(1, 4):
                self.edges[i - j, i].set(
                    "op",
                    [
                        ops.Identity(),
                        ops.Zero(stride=1)  # we need to set stride to one
                    ])
                if j == 1:  # TODO move this from init to sample architecutre or to set max function similiar to darts
                    self._set_op_indice(self.edges[i - j, i], 0)  # TOOD cahnge to max depth
                else:
                    self._set_op_indice(self.edges[i - j, i], 1)

        # edges between blocks
        for i in self.depth_nodes[:-1]:
            self.edges[i, i + 1].set("op", ops.Identity())

        # set bn param
        # TODO this doesnt work iterate all modules and call this function
        # TODO IMPLEMENT THIS FUNCTION
        # self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth
        # doesn't work vanilla
        # TODO fix if needed dont know what it does
        # self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

    def mutate(self):
        mutation = np.random.choice(["depth", "kernel", "expand"])
        if mutation == "depth":
            i = np.random.choice(self.depth_nodes)
            current = 0
            for j in range(1, 4):
                if not self.edges[i - j, i].op_index:
                    current = j
                self._set_op_indice(self.edges[i - j, i], 1)  # set all zero
            d = np.random.choice([n for n in [1, 2, 3] if n != current])
            self._set_op_indice(self.edges[i - d, i], 0)  # set one to identity
        elif mutation == "kernel" or mutation == "expand":
            ind = np.random.choice(np.arange(len(self.block_start_nodes)))
            start_block = self.block_start_nodes[ind]
            depth = 0
            for j in range(1, 4):
                if not self.edges[self.depth_nodes[ind] - j, self.depth_nodes[ind]].op_index:
                    depth = 5 - j
            mutate_ind = np.random.choice(np.arange(depth))
            layer = self.edges[start_block + mutate_ind, start_block + mutate_ind + 1]
            ks, er = layer.op.active_kernel_size, layer.op.active_expand_ratio
            if mutation == "kernel":
                ks = np.random.choice([k for k in self.ks_list if k != ks])
            elif mutation == "expand":
                er = np.random.choice([e for e in self.expand_ratio_list if e != er])
            op_index = self.ks_list.index(ks) * len(self.ks_list) + self.expand_ratio_list.index(er)
            self._set_op_indice(layer, op_index)
        else:
            raise ValueError(f"Mutation type: {mutation} not found")

    def load_op_from_config(self, config_file_path):
        # load operations from a config file
        config = load_config(path=config_file_path)
        print(f'config_kernel_size_list: {config.kernel_size_list} with size: {len(config.kernel_size_list)}')

        # set op indices
        op_indices = []

        # kernel size and expansion ratio
        for ks, er in zip(config.kernel_size_list, config.expand_ratio_list):
            if ks == 3 and er == 3:
                op_indices.append(0)
            elif ks == 3 and er == 4:
                op_indices.append(1)
            elif ks == 3 and er == 6:
                op_indices.append(2)
            elif ks == 5 and er == 3:
                op_indices.append(3)
            elif ks == 5 and er == 4:
                op_indices.append(4)
            elif ks == 5 and er == 6:
                op_indices.append(5)
            elif ks == 7 and er == 3:
                op_indices.append(6)
            elif ks == 7 and er == 4:
                op_indices.append(7)
            elif ks == 7 and er == 6:
                op_indices.append(8)
            else:
                raise ValueError(f"Combination of kernel size {ks} and expansion ratio {er} not allowed")

        # depth
        for i in config.depth_list:
            if i == 4:
                op_indices.append(1)
            elif i == 3:
                op_indices.append(2)
            elif i == 2:
                op_indices.append(3)
            else:
                raise ValueError(f"Depth {i} not allowed")

        self.set_op_indices(op_indices)

    def set_op_indices(self, op_indices):
        # This will update the edges in the OnceForAllSearchSpace object to op_indices
        # op_indices: [ 20 entries between 0 and 8 & 5 entries between 1 and 3]
        # [0]: ks=3 er=3, [1]: ks=3 er=4, [2]: ks=3 er=6, [3]: ks=5 er=3, ..., [8]: ks=7 er=6
        # [1]: depth=4, [2]: depth=3, [3]: depth=2
        for start_node, n_block, j in zip(
                self.block_start_nodes,
                [4] * 5,
                range(5),
        ):
            for i in range(n_block):
                index = op_indices[j * 4 + i]
                self._set_op_indice(self.edges[start_node + i, start_node + i + 1], index)

        for index, i in enumerate(self.depth_nodes):
            for j in range(1, 4):
                self._set_op_indice(self.edges[i - j, i], 1)  # set all zero
            d = op_indices[20 + index]
            self._set_op_indice(self.edges[i - d, i], 0)  # set one to identity

    def sample_random_architecture(self, dataset_api=None):
        # get random op_indices
        op_indices = np.concatenate((np.random.randint(9, size=20), np.random.randint(1, 4, size=5)))
        # set op indices
        self.set_op_indices(op_indices)

    def _set_op_indice(self, edge, index):
        # function that replaces list of ops or current ops with the index give!
        edge.set("op_index", index)
        if isinstance(edge.op, list):
            primitives = edge.op
        else:
            primitives = edge.primitives

        edge.set("op", primitives[edge.op_index])
        edge.set("primitives", primitives)  # store for later use
