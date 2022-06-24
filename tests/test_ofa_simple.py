import unittest
import logging
import torch
import os

from naslib.search_spaces.OnceForAll.graph import *
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, DrNASOptimizer
from naslib.utils import utils, setup_logger

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


class TestOFASearchSpace(unittest.TestCase):

    def setUp(self):
        utils.set_seed(1)
        self.search_space = OnceForAllSearchSpace()

    def test_mutate(self):
        # TODO implement fucntion and test
        raise NotImplementedError

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
        raise NotImplementedError

    def test_query(self):
        # TODO implemnt function and test
        raise NotImplementedError


if __name__ == '__main__':

    unittest.main()
