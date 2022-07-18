import collections
import numpy as np
import torch

from naslib.predictors.ensemble import Ensemble
from naslib.optimizers.discrete.bananas.acquisition_functions import acquisition_function

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
        for i in range(100):
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
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_ensemble = config.search.num_ensemble
        self.predictor_type = config.search.predictor_type
        self.acq_fn_type = config.search.acq_fn_type
        self.acq_fn_optimization = config.search.acq_fn_optimization
        self.sample_size = config.search.sample_size
        self.population_size = config.search.population_size
        self.acq_fn = None
        self.population = collections.deque(maxlen=self.population_size)

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
            self.sampled_archs.append(model)
            self._update_history(model)

        # CREATE AND TRAIN PERFORMANCE PREDICTOR
        else:
            if epoch % 10 == 0:
                xtrain = [m.arch for m in self.sampled_archs]
                ytrain = [m.arch.query(Metric.TEST_ACCURACY, dataset_api=self.dataset_api) for m in self.sampled_archs]
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

            # choose model with highest predicted performance as parent
            # pred_perf = [self.acq_fn(encoding.arch) for encoding in sample]
            # parent = sample[np.argmax(pred_perf)]

            child = torch.nn.Module()
            child.arch = self.search_space.clone()
            if self.constraint:
                self.get_valid_arch_under_constraint(child)
            else:
                child.arch.sample_random_architecture(dataset_api=self.dataset_api)
            child.accuracy = self.acq_fn(child.arch)
            self.population.append(child)
            self.sampled_archs.append(child)
            self._update_history(child)

    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        assert (
            search_space.QUERYABLE
        ), "Regularized evolution is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.ss_type = 'ofa'
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api