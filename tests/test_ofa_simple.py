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
        self.search_space = None
        """
        self.optimizer = DARTSOptimizer(config)
        self.optimizer.adapt_search_space(SimpleCellSearchSpace())
        self.optimizer.before_training()
        """

    def test_init(self):
        self.search_space = OnceForAllSearchSpace()

    def test_mutate(self):
        # TODO implement fucntion and test
        pass

    def test_random_sample(self):
        # TODO implement function and test
        pass
        """
        stats = self.optimizer.step(data_train, data_val)
        self.assertTrue(len(stats) == 4)
        self.assertAlmostEqual(stats[2].detach().cpu().numpy(), 2.4303, places=3)
        self.assertAlmostEqual(stats[3].detach().cpu().numpy(), 2.4303, places=3)
        """

    def test_query(self):
        # TODO implemnt function and test
        pass
        """
        final_arch = self.optimizer.get_final_architecture()
        logits = final_arch(data_train[0])
        self.assertTrue(logits.shape == (2, 10))
        self.assertAlmostEqual(logits[0, 0].detach().cpu().numpy(), 0.0921, places=3)
        """





if __name__ == '__main__':

    unittest.main()
