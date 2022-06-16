import os
import pickle
import numpy as np
import random
import itertools
import torch
import torch.nn as nn

from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.primitives import AbstractPrimitive
from naslib.search_spaces.core.query_metrics import Metric



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
        5 units 7371 choices per layer 
            -Each unit {2,3,4} layers with 9 (3x3) choices 
                - Each layer kernel size {3,5,7}
                - Each layer increases number of channels by {3,4,6}
        """
        # Graph definition
        # we have 5 units
        # maybe a smarter way to get the conditional layers
        number_of_units = 5
        self.add_nodes_from(range(1, number_of_units + 1))
        self.add_edges_from([(i, i + 1) for i in range(1, number_of_units)])

        # TODO add operations
        raise NotImplementedError


    def query(self, metric: Metric, dataset: str, path: str) -> float:
        #TODO use pretrained OFA to query validation and test perfomance
        raise NotImplementedError

    def get_hash(self):
        raise NotImplementedError

    def get_arch_iterator(self):
        raise NotImplementedError

    def set_spec(self):
        raise NotImplementedError

    def sample_random_architecture(self, dataset_api=None):
        raise NotImplementedError

    def get_nbhd(self, dataset_api=None):
        raise NotImplementedError

    def get_type(self):
        return "ofa"
