import os
import sys
import torch
import logging
import traceback
import numpy as np
from pprint import pprint

from runner import *
from runner.aug_runner import AugRunner
from utils.logger import setup_logging
from utils.arg_helper import parse_arguments, get_config
from utils.data_helper import create_graphs

import networkx as nx

torch.set_printoptions(profile="full")


def main():

    torch.cuda.empty_cache()
    args = parse_arguments()
    config = get_config(args.config_file, is_test=args.test)
    config["org_config_path"] = args.config_file
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    config.use_gpu = config.use_gpu and torch.cuda.is_available()

    # log info
    log_file = os.path.join(config.save_dir, "log_exp_{}.txt".format(config.run_id))
    logger = setup_logging(args.log_level, log_file)
    logger.info("Writing log file to {}".format(log_file))
    logger.info("Exp instance id = {}".format(config.run_id))
    logger.info("Exp comment = {}".format(args.comment))
    logger.info("Config =")
    print(">" * 80)
    pprint(config)
    print("<" * 80)

    # Run the experiment
    try:
        runner = eval(config.runner)(config)
        # train_dataset = [nx.connected_watts_strogatz_graph(7, 2, 0.2) for _ in range(5)]
        # test_dataset = create_graphs("grid")[-20:]
        # runner = AugRunner(
        #     config, train_dataset, test_dataset, steps=10, epoch_per_step=5
        # )
        if not args.test:
            runner.train()
        else:
            runner.test()
    except:
        logger.error(traceback.format_exc())

    sys.exit(0)


if __name__ == "__main__":
    main()
