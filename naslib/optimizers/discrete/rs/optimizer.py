import collections
import numpy as np
import torch

from naslib.predictors.ensemble import Ensemble

from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.search_spaces.core.query_metrics import Metric
from naslib.utils import measure_net_latency


class RandomSearch(MetaOptimizer):
    """
    Random search in DARTS is done by randomly sampling `k` architectures
    and training them for `n` epochs, then selecting the best architecture.
    DARTS paper: `k=24` and `n=100` for cifar-10.
    """

    # training the models is not implemented
    using_step_function = False

    def __init__(
            self,
            config,
            weight_optimizer=torch.optim.SGD,
            loss_criteria=torch.nn.CrossEntropyLoss(),
            grad_clip=None,
    ):
        """
        Initialize a random search optimizer.

        Args:
            config
            weight_optimizer (torch.optim.Optimizer): The optimizer to
                train the (convolutional) weights.
            loss_criteria (): The loss
            grad_clip (float): Where to clip the gradients (default None).
        """
        super(RandomSearch, self).__init__()
        self.weight_optimizer = weight_optimizer
        self.loss = loss_criteria
        self.grad_clip = grad_clip

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset
        self.fidelity = config.search.fidelity
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sampled_archs = []
        self.history = torch.nn.ModuleList()

        self.constraint = config.search.constraint
        self.efficiency = config.search.efficiency

    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        assert (
            search_space.QUERYABLE
        ), "Random search is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api

    def new_epoch(self, e):
        """
        Sample a new architecture to train.
        """
        model = torch.nn.Module()  # hacky way to get arch and accuracy checkpointable
        model.arch = self.search_space.clone()
        if self.constraint:
            self.get_valid_arch_under_constraint(model)
        else:
            model.arch.sample_random_architecture(dataset_api=self.dataset_api)
        model.accuracy = model.arch.query(
            self.performance_metric,
            self.dataset,
            epoch=self.fidelity,
            dataset_api=self.dataset_api,
        )
        self.sampled_archs.append(model)
        self._update_history(model)

    def get_valid_arch_under_constraint(self, model):
        while True:
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

    def get_final_architecture(self):
        """
        Returns the sampled architecture with the lowest validation error.
        """
        return max(self.sampled_archs, key=lambda x: x.accuracy).arch

    def train_statistics(self, report_incumbent=True):
        if report_incumbent:
            best_arch = self.get_final_architecture()
        else:
            best_arch = self.sampled_archs[-1].arch

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

    def get_op_optimizer(self):
        return self.weight_optimizer

    def get_checkpointables(self):
        return {"model": self.history}


class RS(RandomSearch):
    def __init__(self, config, efficiency_predictor=None):
        super().__init__(config)
        self.config = config
        self.train_data = []
        self.ss_type = 'ofa'
        self.pretrained_predictor = (config.search.predictor_type == 'pretrained')
        self.predictor = Ensemble(
            num_ensemble=self.config.search.num_ensemble,
            ss_type=self.ss_type,
            predictor_type=config.search.predictor_type,
            config=self.config,
        )
        self.population_size = config.search.population_size
        self.efficiency_predictor = efficiency_predictor

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

            self.train_data.append(model)
            self.sampled_archs.append(model)
            self._update_history(model)

        # CREATE AND TRAIN PERFORMANCE PREDICTOR
        else:
            if epoch - self.population_size == 0 and not self.pretrained_predictor:
                # we fit first time after population is filled
                # query whole population and fit predictor
                xtrain = [m.arch for m in self.train_data]
                ytrain = [m.accuracy for m in self.train_data]

                # train_error not needed here
                train_error = self.predictor.fit(xtrain, ytrain)

            child = torch.nn.Module()
            child.arch = self.search_space.clone()
            if self.constraint:
                self.get_valid_arch_under_constraint(child)
            else:
                child.arch.sample_random_architecture(dataset_api=self.dataset_api)

            if self.pretrained_predictor:
                config = child.arch.get_active_conf_dict()
                child.accuracy = self.predictor.predict_accuracy([config]).item() * 100
            elif epoch - self.population_size != 0 and\
                    (epoch - self.population_size) % 10 == 0 and\
                    not self.pretrained_predictor:
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

            self.sampled_archs.append(child)
            self._update_history(child)

    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        assert (
            search_space.QUERYABLE
        ), "Regularized evolution is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api
        if self.pretrained_predictor:
            self.predictor = self.dataset_api["accuracy_predictor"]

    def get_valid_arch_under_constraint(self, model):
        while True:
            model.arch.sample_random_architecture()
            sample = model.arch.get_active_conf_dict()
            efficiency = self.efficiency_predictor.predict_efficiency(sample)
            if efficiency <= self.efficiency:
                break
