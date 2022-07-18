import collections
import logging
import torch
import copy
import numpy as np

from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.optimizers.discrete.bananas.acquisition_functions import acquisition_function

from naslib.search_spaces.core.query_metrics import Metric

from naslib.predictors.ensemble import Ensemble

from naslib.utils import measure_net_latency
from naslib.utils.utils import AttrDict, count_parameters_in_MB
from naslib.utils.logging import log_every_n_seconds

logger = logging.getLogger(__name__)


class RegularizedEvolution(MetaOptimizer):
    # training the models is not implemented
    using_step_function = False

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.epochs = config.search.epochs
        self.sample_size = config.search.sample_size
        self.population_size = config.search.population_size

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset

        self.population = collections.deque(maxlen=self.population_size)
        self.history = torch.nn.ModuleList()

        self.constraint = config.search.constraint
        self.efficiency = config.search.efficiency

    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        assert (
            search_space.QUERYABLE
        ), "Regularized evolution is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api

    def new_epoch(self, epoch):
        # We sample as many architectures as we need
        if epoch < self.population_size:
            logger.info("Start sampling architectures to fill the population")
            # If there is no scope defined, let's use the search space default one

            model = (
                torch.nn.Module()
            )  # hacky way to get arch and accuracy checkpointable
            model.arch = self.search_space.clone()
            if self.constraint:
                self.get_valid_arch_under_constraint(model)
            else:
                model.arch.sample_random_architecture(dataset_api=self.dataset_api)
            model.accuracy = model.arch.query(
                self.performance_metric, self.dataset, dataset_api=self.dataset_api
            )

            self.population.append(model)
            self._update_history(model)
            log_every_n_seconds(
                logging.INFO, "Population size {}".format(len(self.population))
            )
        else:
            sample = []
            while len(sample) < self.sample_size:
                candidate = np.random.choice(list(self.population))
                sample.append(candidate)

            parent = max(sample, key=lambda x: x.accuracy)

            child = (
                torch.nn.Module()
            )  # hacky way to get arch and accuracy checkpointable
            child.arch = self.search_space.clone()
            if self.constraint:
                self.get_valid_arch_under_constraint(child, parent)
            else:
                child.arch.mutate(parent.arch, dataset_api=self.dataset_api)
            child.accuracy = child.arch.query(
                self.performance_metric, self.dataset, dataset_api=self.dataset_api
            )

            self.population.append(child)
            self._update_history(child)

    def get_valid_arch_under_constraint(self, model, parent=None):
        for i in range(100):
            if parent:
                model.arch.mutate(parent.arch)
            else:
                model.arch.sample_random_architecture()
            if self.constraint == 'latency':
                efficiency, _ = measure_net_latency(model.arch)
            else:
                efficiency = model.arch.get_model_size()
            if efficiency <= self.efficiency:
                break

    def _update_history(self, child):
        if len(self.history) < 100:
            self.history.append(child)
        else:
            for i, p in enumerate(self.history):
                if child.accuracy > p.accuracy:
                    self.history[i] = child
                    break

    def train_statistics(self, report_incumbent=True):
        if report_incumbent:
            best_arch = self.get_final_architecture()
        else:
            best_arch = self.population[-1].arch

        return (
            best_arch.query(
                Metric.TRAIN_ACCURACY, self.dataset, dataset_api=self.dataset_api
            ),
            best_arch.query(
                Metric.VAL_ACCURACY, self.dataset, dataset_api=self.dataset_api
            ),
            best_arch.query(
                Metric.TEST_ACCURACY, self.dataset, dataset_api=self.dataset_api
            ),
            best_arch.query(
                Metric.TRAIN_TIME, self.dataset, dataset_api=self.dataset_api
            ),
        )

    def test_statistics(self):
        best_arch = self.get_final_architecture()
        return best_arch.query(Metric.RAW, self.dataset, dataset_api=self.dataset_api)

    def get_final_architecture(self):
        return max(self.history, key=lambda x: x.accuracy).arch

    def get_op_optimizer(self):
        raise NotImplementedError()

    def get_checkpointables(self):
        return {"model": self.history}

    def get_model_size(self):
        return count_parameters_in_MB(self.history)


class RE(RegularizedEvolution):
    def __init__(self, config):
        super().__init__(config)
        self.num_ensemble = config.search.num_ensemble
        self.predictor_type = config.search.predictor_type
        self.acq_fn_type = config.search.acq_fn_type
        self.acq_fn_optimization = config.search.acq_fn_optimization

        self.train_data = []
        self.acq_fn = None

    def new_epoch(self, epoch):
        if epoch < self.population_size:
            model = torch.nn.Module()
            model.arch = self.search_space.clone()
            if self.constraint:
                self.get_valid_arch_under_constraint(model)
            else:
                model.arch.sample_random_architecture(dataset_api=self.dataset_api)
            model.accuracy = model.arch.query(self.performance_metric,
                                              self.dataset,
                                              dataset_api=self.dataset_api)
            self.population.append(model)
            self.train_data.append(model)
            self._update_history(model)

        # CREATE AND TRAIN PERFORMANCE PREDICTOR
        else:
            if epoch % 10 == 0:
                xtrain = [m.arch for m in self.train_data]
                ytrain = [m.arch.query(Metric.TEST_ACCURACY, dataset_api=self.dataset_api) for m in self.train_data]
                ensemble = Ensemble(num_ensemble=self.num_ensemble,
                                    ss_type=self.ss_type,
                                    predictor_type=self.predictor_type,
                                    config=self.config)
                ensemble.fit(xtrain, ytrain)
                self.acq_fn = acquisition_function(ensemble=ensemble,
                                                   ytrain=ytrain,
                                                   acq_fn_type=self.acq_fn_type)

            sample = []
            while len(sample) < self.sample_size:
                candidate = np.random.choice(list(self.population))
                sample.append(candidate)

            child = torch.nn.Module()
            child.arch = self.search_space.clone()
            if self.constraint:
                self.get_valid_arch_under_constraint(child)
            else:
                child.arch.sample_random_architecture(dataset_api=self.dataset_api)
            child.accuracy = child.arch.query(self.performance_metric,
                                              self.dataset,
                                              dataset_api=self.dataset_api)
            self.population.append(child)
            self.train_data.append(child)
            self._update_history(child)

    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        assert (
            search_space.QUERYABLE
        ), "Regularized evolution is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.ss_type = 'ofa'
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api
