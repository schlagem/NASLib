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
    def __init__(self):
        super().__init__()

        # TODO define graph here

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
