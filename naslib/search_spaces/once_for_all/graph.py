# other imports
from collections import OrderedDict
from itertools import product
from typing import Iterator

import numpy as np
import torch
# Once-For-All imports
from ofa.utils import download_url, make_divisible, MyNetwork
from torch.nn.parameter import Parameter

# NASLib imports
from naslib.search_spaces.once_for_all.blocks import FinalBlock, FirstBlock, OFAConv, OFALayer
from naslib.search_spaces.once_for_all.ofa_utils import spec2feats, set_running_statistics
from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph
from naslib.search_spaces.core.query_metrics import Metric
from naslib.utils.utils import load_config


class OnceForAllSearchSpace(Graph):
    """
    Implementation of the Once for All Search space.
    """
    QUERYABLE = True

    def __init__(self):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_run_stats = False
        # we don't call the search space with params thus we define them here
        # for our application this should suffice
        n_classes = 1000  # ImageNet
        dropout_rate = 0  # default Parameter
        bn_param = (0.1, 1e-5)

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

        # nodes
        self.add_nodes_from([i for i in range(1, 33)])

        # edges preprocessing and postprocessing
        self.add_edges_from([(1, 2), (31, 32)])

        # edges between blocks, always identity
        self.add_edges_from([(i, i + 1) for i in self.depth_nodes[:-1]])

        # intermediate edges
        for i in self.block_start_nodes:
            self.add_edges_from([(i + k, i + k + 1) for k in range(4)])

        # edges different depths
        self.add_edges_from([(i - 3, i) for i in self.depth_nodes])
        self.add_edges_from([(i - 2, i) for i in self.depth_nodes])
        self.add_edges_from([(i - 1, i) for i in self.depth_nodes])

        # assign ops to edges
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
            # make an OFA layer object for all 9 edges
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

                self.edges[start_node + i, start_node + i + 1].set("op", ofa_layer_list)
                self._set_op_index(self.edges[start_node + i, start_node + i + 1], 8)

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
                if j == 1:
                    self._set_op_index(self.edges[i - j, i], 0)
                else:
                    self._set_op_index(self.edges[i - j, i], 1)

        # edges between blocks
        for i in self.depth_nodes[:-1]:
            self.edges[i, i + 1].set("op", ops.Identity())

    def mutate(self, parent=None, dataset_api=None):
        self.set_run_stats = False
        if parent:
            parent_conf = parent.get_op_indices()
            self.set_op_indices(parent_conf)
        mutation = np.random.choice(["depth", "kernel", "expand"])
        if mutation == "depth":
            i = np.random.choice(self.depth_nodes)
            current = 0
            for j in range(1, 4):
                if not self.edges[i - j, i].op_index:
                    current = j
                self._set_op_index(self.edges[i - j, i], 1)  # set all zero
            d = np.random.choice([n for n in [1, 2, 3] if n != current])
            self._set_op_index(self.edges[i - d, i], 0)  # set one to identity
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
            self._set_op_index(layer, op_index)
        else:
            raise ValueError(f"Mutation type: {mutation} not found")

    def load_op_from_config(self, config_file_path):
        """This load a configuration from a file"""
        config = load_config(path=config_file_path)

        # set op indices
        op_indices = []

        # kernel size and expansion ratio
        for ks, er in zip(config.kernel_size_list, config.expand_ratio_list):
            if ks not in self.ks_list:
                raise ValueError(f"Kernel size {ks} not in valid sizes {self.ks_list}!")
            if er not in self.expand_ratio_list:
                raise ValueError(f"Expand ratio {er} not in valid sizes {self.expand_ratio_list}!")
            op_indices.append(self.ks_list.index(ks) * len(self.ks_list) + self.expand_ratio_list.index(er))
        # depth
        for i in config.depth_list:
            if i not in self.depth_list:
                raise ValueError(f"Depth {i} not in valid sizes {self.depth_list}!")
            op_indices.append(5 - i)

        self.set_op_indices(op_indices)

    def set_op_indices(self, op_indices):
        """# This will update the edges in the OnceForAllSearchSpace object to op_indices
        op_indices: [ 20 entries between 0 and 8 & 5 entries between 1 and 3]
        [0]: ks=3 er=3, [1]: ks=3 er=4, [2]: ks=3 er=6, [3]: ks=5 er=3, ..., [8]: ks=7 er=6
        [1]: depth=4, [2]: depth=3, [3]: depth=2
        """
        for start_node, n_block, j in zip(
                self.block_start_nodes,
                [4] * 5,
                range(5),
        ):
            for i in range(n_block):
                index = op_indices[j * 4 + i]
                self._set_op_index(self.edges[start_node + i, start_node + i + 1], index)

        for index, i in enumerate(self.depth_nodes):
            for j in range(1, 4):
                self._set_op_index(self.edges[i - j, i], 1)  # set all zero
            d = op_indices[20 + index]
            self._set_op_index(self.edges[i - d, i], 0)  # set one to identity
        self.set_run_stats = False

    def get_op_indices(self):
        """This will return the operation indices of the complete graph."""
        op_indices = []
        for start_node, n_block, j in zip(
                self.block_start_nodes,
                [4] * 5,
                range(5),
        ):
            for i in range(n_block):
                op_indices.append(self.edges[start_node + i, start_node + i + 1].get('op_index', 0))

        for index, i in enumerate(self.depth_nodes):
            for j in range(1, 4):
                op = self.edges[i - j, i].get('op_index', 0)
                if op == 0:
                    op_indices.append(j)
        return op_indices

    def sample_random_architecture(self, dataset_api=None):
        """This will sample a random architecture and update the edges in the naslib object accordingly."""
        self.set_run_stats = False
        # get random op_indices
        op_indices = np.concatenate((np.random.randint(9, size=20), np.random.randint(1, 4, size=5)))
        # set op indices
        self.set_op_indices(op_indices)

    @staticmethod
    def _set_op_index(edge, index):
        """This will set the operation at a given index"""
        # function that replaces list of ops or current ops with the index give!
        edge.set("op_index", index)
        if isinstance(edge.op, list):
            primitives = edge.op
        else:
            primitives = edge.primitives

        edge.set("op", primitives[edge.op_index])
        edge.set("primitives", primitives)  # store for later use

    def set_weights(self, net_id="ofa_mbv3_d234_e346_k357_w1.0"):
        """Loads the weights of the given graph and sets them accordingly."""
        url_base = "https://hanlab.mit.edu/files/OnceForAll/ofa_nets/"
        init = torch.load(
            download_url(url_base + net_id, model_dir=".torch/ofa_nets"),
            map_location="cpu",
        )["state_dict"]
        keys = init.keys()

        first_unit = self.edges[1, 2].op
        first_conv_state = OrderedDict((k.replace("first_conv.", ""), init[k]) for k in keys if "first_conv" in k)
        first_block_state = OrderedDict((k.replace("blocks.0.mobile_inverted_conv", "conv"), init[k]) for k in keys
                                        if "blocks.0" in k)
        first_unit.set_weights(first_conv_state, first_block_state)

        block_idx = 1
        for e in self.edges:
            block = self.edges[e].op
            if not isinstance(block, OFALayer):
                continue
            block_dict = OrderedDict((k.replace("blocks." + str(block_idx) + ".mobile_inverted_conv", "conv"),
                                      init[k]) for k in keys if "blocks." + str(block_idx) + "." in k)
            block.set_weights(block_dict)
            block_idx += 1

        final = list(self.edges)[-1]
        final_unit = self.edges[final].op
        final_expand_dict = OrderedDict((k.replace("final_expand_layer.", ""), init[k]) for k in keys
                                        if "final_expand_layer." in k)
        feature_mix_dict = OrderedDict((k.replace("feature_mix_layer.", ""), init[k]) for k in keys
                                       if "feature_mix_layer." in k)
        classifier_dict = OrderedDict((k.replace("classifier.", ""), init[k]) for k in keys
                                      if "classifier." in k)
        final_unit.set_weights(final_expand_dict, feature_mix_dict, classifier_dict)
        self.set_run_stats = True

    def _state_dict(self):
        """Return the state of the graph as dictionary."""
        ord_dict = OrderedDict()
        block_idx = 1
        for e in self.edges:
            block = self.edges[e].op
            if type(block) in [ops.Identity, ops.Zero]:
                continue
            if isinstance(block, OFALayer):
                state = block.ofa_conv.res_block.state_dict()
                out = OrderedDict((k.replace("conv", "block" + str(block_idx)),
                                   state[k]) for k in state.keys())
                ord_dict.update(out)
                block_idx += 1
            else:
                ord_dict.update(block.state_dict())
        return ord_dict

    def query(
            self,
            metric=None,
            dataset=None,
            path=None,
            epoch=-1,
            full_lc=False,
            dataset_api=None,
    ):
        metric_to_ofa = {
            Metric.VAL_ACCURACY: "val_acc",
            Metric.TEST_ACCURACY: "test_acc",
        }
        metr = metric_to_ofa.get(metric)
        if metric == Metric.VAL_ACCURACY or metric == Metric.TEST_ACCURACY:
            if dataset_api:
                lut = dataset_api['lut']
                arch = self.encode_str()
                if lut.setdefault(arch, {}) and lut[arch].get(metr, 0):
                    return lut[arch][metr]
                else:
                    accuracy = self.evaluate(dataset_api, metric)
                    lut[arch][metr] = accuracy
                    return accuracy
        return -1

    @torch.no_grad()
    def evaluate(self, dataset_api=None, metric=None):
        if not self.set_run_stats:
            data_loader = dataset_api["dataloader_test"]
            set_running_statistics(self, data_loader, self.device)
            self.set_run_stats = True
        if metric == Metric.VAL_ACCURACY:
            data_loader = dataset_api["dataloader_val"]
        elif metric == Metric.TEST_ACCURACY:
            data_loader = dataset_api["dataloader_test"]
        self.eval()
        self.to(self.device)
        total = len(data_loader.dataset)
        correct = 0
        for images, labels in data_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            output = self(images)
            _, predicted = torch.max(output.detach(), 1)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total * 100
        self.to()
        return accuracy

    def get_active_config(self):
        d, k, e = [], [], []
        for d_node, start_node in zip(self.depth_nodes, self.block_start_nodes):
            depth = 0
            for j in range(1, 4):
                if not self.edges[d_node - j, d_node].op_index:
                    depth = 5 - j
            d.append(depth)
            for n in range(4):
                layer = self.edges[start_node + n, start_node + n + 1].op
                kernel, expand = layer.active_kernel_size, layer.active_expand_ratio
                k.append(kernel)
                e.append(expand)
        return d, k, e

    def get_active_conf_dict(self, resolution=224):
        d, ks, e = self.get_active_config()
        out = {
            "ks": ks,
            "e": e,
            "d": d,
            "r": [resolution]
        }
        return out

    def encode_str(self) -> str:
        d, k, e = self.get_active_config()
        r = 224
        one_hot = spec2feats(k, e, d, r).numpy().astype(int)
        out = ''
        for part in (one_hot[:60], one_hot[60:120], one_hot[120:]):
            out += ''.join(str(s) for s in part)
        return out

    def encode(self):
        d, k, e = self.get_active_config()
        r = 224
        enc = spec2feats(k, e, d, r)
        return enc

    def get_model_size(self):
        """Returns model size in mb."""
        size_all_mb = 0.0
        # First block
        size_all_mb += self.edges[1, 2].op.size()
        # Adaptive layers
        d, _, _ = self.get_active_config()
        for depth, start_node in zip(d, self.block_start_nodes):
            for n in range(depth):
                layer = self.edges[start_node + n, start_node + n + 1].op
                size_all_mb += layer.size()
        # Last block
        size_all_mb += self.edges[31, 32].op.size()
        return size_all_mb

    def to(self, device=torch.device('cpu')):
        """Helper function, that moves the graph to a specific device."""
        for e in self.edges:
            try:
                self.edges[e].op.move_to(device)
            except AttributeError:
                layer = self.edges[e].op.to(device)
                self.edges[e].set('op', layer)
        return self

    def eval(self):
        """Helper function, that activates eval mode for the whole graph."""
        for e in self.edges:
            layer = self.edges[e].op
            if isinstance(layer, OFALayer):
                layer = layer.ofa_conv.res_block
            layer.eval()

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Helper function, that returns the parameters of the active graph like a pytorch model."""
        edges = [(1, 2)]
        # Adaptive layers
        d, _, _ = self.get_active_config()
        for depth, start_node in zip(d, self.block_start_nodes):
            for n in range(depth):
                edges += [(start_node + n, start_node + n + 1)]
        # Last block
        edges += [(31, 32)]
        for e in edges:
            layer = self.edges[e].op
            for p in layer.params():
                yield p

    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        edges = [(1, 2)]
        # Adaptive layers
        d, _, _ = self.get_active_config()
        for start_node in self.block_start_nodes:
            for n in range(4):
                edges += [(start_node + n, start_node + n + 1)]
        # Last block
        edges += [(31, 32)]
        num = 0
        for e in edges:
            layer = self.edges[e].op
            submodule_prefix = str(num) + prefix
            for m in layer.named_modules(memo, submodule_prefix, remove_duplicate):
                # submodule_prefix = prefix + ('.' if prefix else '') + name
                for mod in m:
                    yield mod
            num += 1

    @staticmethod
    def get_type():
        return "ofa"
