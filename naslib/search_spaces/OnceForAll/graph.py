# naslib imports
import os.path

from naslib.search_spaces.OnceForAll.blocks import FinalBlock, FirstBlock, OFAConv, OFALayer
from naslib.search_spaces.core.graph import Graph
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.core import primitives as ops

# Once for all imports
from ofa.utils import download_url
from ofa.utils import MyNetwork, make_divisible

from ofa.model_zoo import ofa_net

# other imports
import numpy as np
import torch
from itertools import product
from collections import OrderedDict
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
from naslib.utils.utils import AverageMeter
from tqdm import tqdm
import math


class OnceForAllSearchSpace(Graph):
    """
    Implementation of the Once for All Search space.
    """
    QUERYABLE = True
    DEFAULT_IMAGENET_PATH = None

    def __init__(self):
        super().__init__()

        # we don't call the search space with params thus we define them here
        # for our application this should suffice
        n_classes = 1000  # ImageNet
        dropout_rate = 0  # default Parameter TODO check if better value
        # TODO dropout when using pretrained is zero
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
        self.add_edges_from([(i, i + 1) for i in self.depth_nodes[:-1]])

        # intermediate edges
        for i in self.block_start_nodes:
            self.add_edges_from([(i + k, i + k + 1) for k in range(4)])

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

                self.edges[start_node + i, start_node + i + 1].set("op", ofa_layer_list)
                self._set_op_indice(self.edges[start_node + i, start_node + i + 1], 8)

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
                if j == 1:  # TODO move this from init to sample architecture or to set max function similar to darts
                    self._set_op_indice(self.edges[i - j, i], 0)  # TODO change to max depth
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

    def sample_random_architecture(self, dataset_api=None):
        for start_node, n_block in zip(
                self.block_start_nodes,
                [4] * 5,
        ):
            for i in range(n_block):
                self._set_op_indice(self.edges[start_node + i, start_node + i + 1], np.random.choice(np.arange(9)))

        for i in self.depth_nodes:
            for j in range(1, 4):
                self._set_op_indice(self.edges[i - j, i], 1)  # set all zero
            d = np.random.choice([1, 2, 3])
            self._set_op_indice(self.edges[i - d, i], 0)  # set one to identity

    @staticmethod
    def _set_op_indice(edge, index):
        # function that replaces list of ops or current ops with the index give!
        edge.set("op_index", index)
        if isinstance(edge.op, list):
            primitives = edge.op
        else:
            primitives = edge.primitives

        edge.set("op", primitives[edge.op_index])
        edge.set("primitives", primitives)  # store for later use

    def _set_weights(self):
        net_id = "ofa_mbv3_d234_e346_k357_w1.0"
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

    def _state_dict(self):
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
            metric: Metric,
            dataset: str,
            path: str) -> float:

        assert metric in [Metric.VAL_ACCURACY, Metric.TEST_ACCURACY]

        d, ks, e = self.get_active_config()

        net_id = "ofa_mbv3_d234_e346_k357_w1.0"
        ofa_network = ofa_net(net_id, pretrained=True)
        ofa_network.set_active_subnet(ks=ks, e=e, d=d)
        manual_subnet = ofa_network.get_active_subnet(preserve_weight=True)
        return self._eval_pretrained_ofa(manual_subnet, metric)

    def _eval_pretrained_ofa(self, net, metric: Metric, path='~/dataset/imagenet/'):
        if self.DEFAULT_IMAGENET_PATH:
            path = self.DEFAULT_IMAGENET_PATH
        if metric == Metric.VAL_ACCURACY:
            metric = 'val'
        else:
            metric = 'test'
        data_path = os.path.join(path, metric)
        imagenet_data = datasets.ImageFolder(
            data_path,
            self._ofa_transform()
        )
        data_loader = torch.utils.data.DataLoader(
            imagenet_data,
            batch_size=256,  # ~5GB
            shuffle=False,
            pin_memory=True
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.eval()
        net.to(device)
        criterion = nn.CrossEntropyLoss()
        losses = AverageMeter()
        correct = 0
        total = len(data_loader.dataset)
        self.move_to_device()
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(data_loader, ascii=True)):
                images, labels = images.to(device), labels.to(device)
                output = self.forward(images)
                loss = criterion(output, labels)
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == labels).sum().item()
                losses.update(loss.item())
        accuracy = correct / total
        return accuracy

    @staticmethod
    def _ofa_transform(image_size=None):
        if image_size is None:
            image_size = 224
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        return transforms.Compose([
            transforms.Resize(int(math.ceil(image_size / 0.875))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize]
        )

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
