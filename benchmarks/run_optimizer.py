import logging

from naslib.optimizers import RegularizedEvolution as RS
from naslib.search_spaces import OnceForAllSearchSpace as OFA
from naslib.defaults.trainer import Trainer

from naslib.utils import utils, setup_logger, get_dataset_api


def run_optimizer(config_file="docs/re_run_0/imagenet/configs/nas_predictors/config_re_gp_0.yaml",
                  nas_optimizer=RE) -> None:
    # TODO: add all the utilities, such as config file reading, logging as before.
    # afterwards instantiate the search space, optimizer, trainer and run the search + evaluation
    # args = ["--config-file", config_file]
    config = utils.load_config(config_file)
    config.save = "{}/{}/{}/{}/{}/{}".format(
        config.out_dir, config.dataset, "nas_predictors", config.search_space, config.search.predictor_type, config.seed,
    )
    utils.set_seed(config.seed)
    utils.log_args(config)

    logger = setup_logger(config.save + "/log.log")
    logger.setLevel(logging.INFO)

    search_space = OFA()

    optimizer = nas_optimizer(config)

    # dataset_api = get_dataset_api(config.search_space, config.dataset)

    # adapt the search space to the optimizer type
    optimizer.adapt_search_space(search_space)

    trainer = Trainer(optimizer, config, lightweight_output=True)

    trainer.search()

    trainer.evaluate()


if __name__ == "__main__":
    run_optimizer()
