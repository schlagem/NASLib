import unittest
import logging
import torch
import os

from naslib.search_spaces.OnceForAll.graph import *
from naslib.search_spaces.darts.graph import *
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
        for d_node, start_node in zip(self.search_space.depth_nodes, self.search_space.block_start_nodes):
            depth = 0
            for j in range(1, 4):
                if not self.search_space.edges[d_node - j, d_node].op_index:
                    depth = 5 - j
            if depth not in [2, 3, 4]:
                raise ValueError(f"Depth {depth} not in valid range {[2, 3, 4]}")
            b_state = []
            for n in range(depth):
                layer = self.search_space.edges[start_node + n, start_node + n + 1].op
                layer_state = [layer.active_kernel_size, layer.active_expand_ratio]
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
            for start_node, n_block in zip(
                    self.search_space.block_start_nodes,
                    [4] * 5,
            ):
                for n in range(n_block):
                    # check active layer valid
                    layer = self.search_space.edges[start_node + n, start_node + n + 1].op
                    self.assertTrue(layer.active_kernel_size in [3, 5, 7])
                    self.assertTrue(layer.active_expand_ratio in [3, 4, 6])

            for d in self.search_space.depth_nodes:
                # check active depth valid
                active_op = []
                for j in range(1, 4):
                    active_op.append(self.search_space.edges[d - j, d].op_index)  # set all zero
                unique, counts = np.unique(active_op, return_counts=True)
                print(unique, counts)
                self.assertTrue((unique == [0, 1]).all())
                self.assertTrue((counts == [1, 2]).all())

    def test_forward_working(self):
        self.search_space.sample_random_architecture()
        for i in range(10):
            x = torch.rand((3, 3, 3, 3))
            y_graph = self.search_space.forward(x)
            self.search_space.sample_random_architecture()

    def test_forward_correctness(self):
        # requires loading weights
        self.search_space._set_weights()
        net_id = "ofa_mbv3_d234_e346_k357_w1.0"
        ofa_network = ofa_net(net_id, pretrained=True)
        with torch.no_grad():
            x = torch.rand((3, 3, 3, 3))
            y_graph = self.search_space.forward(x)
            y_ofa = ofa_network(x)
            self.assertTrue(torch.equal(y_graph, y_ofa))

    def test_weights(self):
        self.search_space._set_weights()
        ss_dict = self.search_space._state_dict()

        net_id = "ofa_mbv3_d234_e346_k357_w1.0"
        ofa_network = ofa_net(net_id, pretrained=True)
        ofa_dict = ofa_network.state_dict()

        for search_space_value, ofa_value in zip(ss_dict, ofa_dict):
            self.assertTrue(torch.equal(ss_dict[search_space_value], ofa_dict[ofa_value]))

    def test_query(self):
        # TODO maybe test accuracy
        raise NotImplementedError
        self.search_space._set_weights()


if __name__ == '__main__':
    unittest.main()
