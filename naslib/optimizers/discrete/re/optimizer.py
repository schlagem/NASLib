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

    def __init__(self, config, efficiency_predictor):
        super().__init__(config)
        self.train_data = []
        self.efficiency_predictor = efficiency_predictor
        self.ss_type = "ofa"
        self.pretrained_predictor = (config.search.predictor_type == 'pretrained')
        self.predictor = Ensemble(
            num_ensemble=self.config.search.num_ensemble,
            ss_type=self.ss_type,
            predictor_type=config.search.predictor_type,
            config=self.config,
        )

    def new_epoch(self, epoch):
        # This is the main method that you have to override in order to add the performance predictors
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

            self.train_data.append(model)
            self.population.append(model)
            self._update_history(model)
        else:
            if epoch - self.population_size == 0 and not self.pretrained_predictor:
                # we fit first time after population is filled
                # query whole population and fit predictor
                xtrain = [m.arch for m in self.train_data]
                ytrain = [m.accuracy for m in self.train_data]

                # train_error not needed here
                train_error = self.predictor.fit(xtrain, ytrain)

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
                self.get_valid_arch_under_constraint(child, parent=parent)
            else:
                child.arch.mutate(parent.arch, dataset_api=self.dataset_api)

            if self.pretrained_predictor:
                config = child.arch.get_active_conf_dict()
                child.accuracy = self.predictor.predict_accuracy([config]).item() * 100
            elif epoch - self.population_size != 0 and (epoch - self.population_size) % 10 == 0:
                child.accuracy = child.arch.query(
                    self.performance_metric, self.dataset, dataset_api=self.dataset_api
                )
                self.train_data.append(child)

                xtrain = [m.arch for m in self.train_data]
                ytrain = [m.accuracy for m in self.train_data]

                # train_error not needed here
                train_error = self.predictor.fit(xtrain, ytrain)
            else:
                child.accuracy = np.mean(self.predictor.query([child.arch]))

            self.population.append(child)
            self._update_history(child)

    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        assert (
            search_space.QUERYABLE
        ), "Regularized evolution is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.ss_type = 'ofa'
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api
        if self.pretrained_predictor:
            self.predictor = self.dataset_api["accuracy_predictor"]

    def get_op_optimizer(self):
        raise NotImplementedError()

    def get_valid_arch_under_constraint(self, model, parent=None):
        while True:
            if parent:
                model.arch.mutate(parent.arch)
            else:
                model.arch.sample_random_architecture()
            sample = model.arch.get_active_conf_dict()
            efficiency = self.efficiency_predictor.predict_efficiency(sample)
            if efficiency <= self.efficiency:
                break
