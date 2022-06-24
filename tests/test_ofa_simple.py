import unittest
import logging
import torch
import os

from naslib.search_spaces.OnceForAll.graph import *
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, DrNASOptimizer
from naslib.utils import utils, setup_logger

from ofa.model_zoo import ofa_net

logger = setup_logger(os.path.join(utils.get_project_root().parent, "tmp", "tests.log"))
logger.handlers[0].setLevel(logging.FATAL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = utils.AttrDict()
config.dataset = 'cifar10'
config.search = utils.AttrDict()
config.search.grad_clip = None
config.search.learning_rate = 0.01
config.search.momentum = 0.1
config.search.weight_decay = 0.1
config.search.arch_learning_rate = 0.01
config.search.arch_weight_decay = 0.1
config.search.tau_max = 10
config.search.tau_min = 1
config.search.epochs = 2


def calc_state_diff(state, new_state):
    changes = 0
    for i in range(5):
        if len(state[i]) != len(new_state[i]):
            # block depth
            changes += 1
        for old, new in zip(state[i], new_state[i]):
            # layers
            if old[0] != new[0] or old[1] != new[1]:
                changes += 1
            assert isinstance(new[0], int) or isinstance(new[0], tuple)  # requirement by ofa code
    return changes


class TestOFASearchSpace(unittest.TestCase):

    def setUp(self):
        utils.set_seed(1)
        self.search_space = OnceForAllSearchSpace()

    def generate_ds(self):
        ds = {}
        i = 0
        for j in range(1 + self.search_space.offset, self.search_space.number_of_units + self.search_space.offset + 1):
            block = self.search_space.edges[j, j + 1].op
            b_state = []
            for layer in block.blocks[:block.depth]:
                layer_state = [layer.conv.active_kernel_size, layer.conv.active_expand_ratio]
                b_state.append(layer_state)
            ds[i] = b_state
            i += 1
        return ds

    def test_mutate(self):
        """
        Test that mutation always changes exactly one property
        generate data structure to comapre can be maybe moved to search space (somewhat like identyfying hash)
        """
        for i in range(1000):
            state = self.generate_ds()
            self.search_space.mutate()
            new_state = self.generate_ds()
            num_of_changes = calc_state_diff(state, new_state)
            self.assertEqual(num_of_changes, 1)

    def test_random_sample(self):
        for i in range(1000):
            self.search_space.sample_random_architecture()
            for j in range(1 + self.search_space.offset,
                           self.search_space.number_of_units + self.search_space.offset + 1):
                block = self.search_space.edges[j, j + 1].op
                self.assertTrue(block.depth in [2, 3, 4])
                for layer in block.blocks:
                    self.assertTrue(layer.conv.active_kernel_size in [3, 5, 7])
                    self.assertTrue(layer.conv.active_expand_ratio in [3, 4, 6])

    def test_forward(self):
        self.search_space._set_weights()
        net_id = "ofa_mbv3_d234_e346_k357_w1.0"
        ofa_network = ofa_net(net_id, pretrained=True)
        print(ofa_network.module_str)
        print(self.search_space.module_str)
        for i in range(1):
            x = torch.rand((3, 3, 3, 3))
            y_graph = self.search_space.forward(x)
            y_ofa = ofa_network(x)
            # TODO they are almost equal ->> Error probadly because BatchNorm values not set correctly see init
            print(torch.max(y_ofa, dim=1))
            print(torch.max(y_graph, dim=1))
            self.assertTrue(torch.equal(y_graph, y_ofa))
            self.search_space.sample_random_architecture()

    def test_query(self):
        # TODO maybe test accuracy
        self.search_space._set_weights()
        raise NotImplementedError


if __name__ == '__main__':

    unittest.main()
