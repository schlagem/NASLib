import logging
import time
from os import listdir
from os.path import isfile, join

from naslib.optimizers import Bananas, RE, RS
from naslib.search_spaces.OnceForAll.efficiency_table import EfficiencyTable
from naslib.search_spaces import OnceForAllSearchSpace as OFA
from naslib.defaults.trainer import Trainer
from naslib.utils import utils, setup_logger, get_dataset_api


def run_optimizer(config_file, nas_optimizer):
    start = time.time()
    config = utils.load_config(config_file)
    quartile = config_file[-6:-5]
    constraint = config.search.constraint if config.search.constraint else 'none'
    config.save = "{}/{}/{}/{}/{}/{}".format(
        config.out_dir, config.dataset, "runs", config.search.predictor_type, constraint, quartile,
    )
    utils.set_seed(config.search.seed)
    utils.log_args(config)

    search_space = OFA()
    search_space.set_weights()

    efficiency_predictor = EfficiencyTable(pred_type=config.search.constraint, load_efficiency_table=True)\
        if config.search.constraint else None
    optimizer = nas_optimizer(config, efficiency_predictor)
    dataset_api = get_dataset_api(config.search_space, config.dataset)
    optimizer.adapt_search_space(search_space, None, dataset_api)

    trainer = Trainer(optimizer, config, lightweight_output=True)
    trainer.search()
    trainer.evaluate(dataset_api=dataset_api)

    dataset_api.close()
    logger.info(f"FINISHED RUN: {config.search.predictor_type}/{constraint}/{quartile}"
                f" in {'%dm %2ds' % (divmod(int(time.time()-start), 60))}")


if __name__ == "__main__":
    optimizer = RS
    path = 'docs/rs_run/imagenet/configs/nas_predictors'
    list_of_config_files = [f for f in listdir(path) if isfile(join(path, f))]

    logger = setup_logger('docs/rs_run/imagenet/runs' + "/log.log")
    logger.setLevel(logging.INFO)

    for file in sorted(list_of_config_files):
        config_file_path = join(path, file)
        run_optimizer(config_file=config_file_path, nas_optimizer=optimizer)

