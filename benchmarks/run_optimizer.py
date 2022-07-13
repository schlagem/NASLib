import logging
from os import listdir
from os.path import isfile, join

from naslib.optimizers import Bananas, RegularizedEvolution as RE, RandomSearch as RS
from naslib.search_spaces import OnceForAllSearchSpace as OFA
from naslib.defaults.trainer import Trainer
from naslib.utils import utils, setup_logger, get_dataset_api


def run_optimizer(config_file, nas_optimizer):
    config = utils.load_config(config_file)
    config.save = "{}/{}/{}/{}/{}/{}".format(
        config.out_dir, config.dataset, "nas_predictors", config.search_space, config.search.predictor_type,
        config.search.seed,
    )
    utils.set_seed(config.search.seed)
    utils.log_args(config)

    logger = setup_logger(config.save + "/log.log")
    logger.setLevel(logging.INFO)

    search_space = OFA()

    optimizer = nas_optimizer(config)
    # dataset_api = get_dataset_api(config.search_space, config.dataset)
    optimizer.adapt_search_space(search_space)

    trainer = Trainer(optimizer, config, lightweight_output=True)
    trainer.search()
    trainer.evaluate()


if __name__ == "__main__":
    optim = RS
    path = 'docs/rs_run_0/imagenet/configs/nas_predictors'
    list_of_config_files = [f for f in listdir(path) if isfile(join(path, f))]

    for file in sorted(list_of_config_files):
        config_file_path = join(path, file)
        run_optimizer(config_file=config_file_path, nas_optimizer=optim)
