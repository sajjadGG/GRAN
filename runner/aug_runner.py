from __future__ import division, print_function
import os
import time
import networkx as nx
import numpy as np
import copy
import pickle
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch.utils.data.distributed as distributed

from model import *
from dataset import *
from utils.logger import get_logger
from utils.train_helper import data_to_gpu, snapshot, load_model, EarlyStopper
from utils.data_helper import *
from utils.eval_helper import *
from utils.dist_helper import compute_mmd, gaussian_emd, gaussian, emd, gaussian_tv
from utils.vis_helper import draw_graph_list, draw_graph_list_separate
from utils.data_parallel import DataParallel

from runner_utils import get_optimizer

logger = get_logger("exp_logger")


class AugRunner:
    def __init__(
        self,
        config: edict,
        train_graphs: list,
        test_graphs: list,
        steps=20,
        epoch_per_step=20,
        num_dev=10,
    ) -> None:
        self.config = config
        self.train_graphs = train_graphs
        self.test_graphs = test_graphs
        self.steps = steps
        self.epoch_per_step = epoch_per_step
        self.seed = config.seed
        self.dataset_conf = config.dataset
        self.model_conf = config.model
        self.train_conf = config.train
        self.test_conf = config.test
        self.use_gpu = config.use_gpu
        self.gpus = config.gpus
        self.device = config.device
        self.writer = SummaryWriter(config.save_dir)
        self.is_vis = config.test.is_vis
        self.better_vis = config.test.better_vis
        self.num_vis = config.test.num_vis
        self.vis_num_row = config.test.vis_num_row
        self.is_single_plot = config.test.is_single_plot
        self.num_gpus = len(self.gpus)
        self.is_shuffle = False
        self.use_gpu = True
        self.num_dev = num_dev

        if self.is_shuffle:
            self.npr = np.random.RandomState(self.seed)
            self.npr.shuffle(self.graphs)

    def train(self) -> None:

        train_dataset = GRANData(self.config, self.train_graphs, tag="train")

        train_loader = torch.utils.data.DataLoader(
            self.train_graphs,
            batch_size=self.train_conf.batch_size,
            shuffle=self.train_conf.shuffle,
            num_workers=self.train_conf.num_workers,
            collate_fn=self.train_graphs.collate_fn,
            drop_last=False,
        )

        self.graphs_train = self.train_graphs[self.num_dev :]
        self.graphs_dev = self.train_graphs[: self.num_dev]
        self.graphs_test = self.test_graphs

        model = eval(self.model_conf.name)(self.config)
        params = filter(lambda p: p.requires_grad, model.parameters())
        if self.use_gpu:
            model = DataParallel(model, device_ids=self.gpus).to(self.device)

        optimizer = get_optimizer(self.conf, params)
        early_stop = EarlyStopper([0.0], win_size=100, is_decrease=False)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.train_conf.lr_decay_epoch,
            gamma=self.train_conf.lr_decay,
        )

        optimizer.zero_grad()

        iter_count = 0

        results = defaultdict(list)

        for step_num in self.steps:
            # generate augmented samples

            # Training Loop
            for epoch in range(self.epoch_per_step):
                model.train()
                lr_scheduler.step()
                train_iterator = train_loader.__iter__()

                for inner_iter in range(len(train_loader) // self.num_gpus):
                    optimizer.zero_grad()

                    batch_data = []
                    if self.use_gpu:
                        for _ in self.gpus:
                            data = train_iterator.next()
                            batch_data.append(data)
                            iter_count += 1

                    avg_train_loss = 0.0
                    for ff in range(self.dataset_conf.num_fwd_pass):
                        batch_fwd = []

                        if self.use_gpu:
                            for dd, gpu_id in enumerate(self.gpus):
                                data = {}
                                data["adj"] = (
                                    batch_data[dd][ff]["adj"]
                                    .pin_memory()
                                    .to(gpu_id, non_blocking=True)
                                )
                                data["edges"] = (
                                    batch_data[dd][ff]["edges"]
                                    .pin_memory()
                                    .to(gpu_id, non_blocking=True)
                                )
                                data["node_idx_gnn"] = (
                                    batch_data[dd][ff]["node_idx_gnn"]
                                    .pin_memory()
                                    .to(gpu_id, non_blocking=True)
                                )
                                data["node_idx_feat"] = (
                                    batch_data[dd][ff]["node_idx_feat"]
                                    .pin_memory()
                                    .to(gpu_id, non_blocking=True)
                                )
                                data["label"] = (
                                    batch_data[dd][ff]["label"]
                                    .pin_memory()
                                    .to(gpu_id, non_blocking=True)
                                )
                                data["att_idx"] = (
                                    batch_data[dd][ff]["att_idx"]
                                    .pin_memory()
                                    .to(gpu_id, non_blocking=True)
                                )
                                data["subgraph_idx"] = (
                                    batch_data[dd][ff]["subgraph_idx"]
                                    .pin_memory()
                                    .to(gpu_id, non_blocking=True)
                                )
                                data["subgraph_idx_base"] = (
                                    batch_data[dd][ff]["subgraph_idx_base"]
                                    .pin_memory()
                                    .to(gpu_id, non_blocking=True)
                                )
                                batch_fwd.append((data,))

                        if batch_fwd:
                            train_loss = model(*batch_fwd).mean()
                            avg_train_loss += train_loss

                            # assign gradient
                            train_loss.backward()

                    # clip_grad_norm_(model.parameters(), 5.0e-0)
                    optimizer.step()
                    avg_train_loss /= float(self.dataset_conf.num_fwd_pass)

                    # reduce
                    train_loss = float(avg_train_loss.data.cpu().numpy())

                    self.writer.add_scalar("train_loss", train_loss, iter_count)
                    results["train_loss"] += [train_loss]
                    results["train_step"] += [iter_count]

                    if (
                        iter_count % self.train_conf.display_iter == 0
                        or iter_count == 1
                    ):
                        logger.info(
                            "NLL Loss @ epoch {:04d} iteration {:08d} = {}".format(
                                epoch + 1, iter_count, train_loss
                            )
                        )

                # snapshot model
                if (
                    step_num * self.epoch_per_step + epoch + 1
                ) % self.train_conf.snapshot_epoch == 0:
                    logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))

                    last_file_path, last_file_name = snapshot(
                        model.module if self.use_gpu else model,
                        optimizer,
                        self.config,
                        epoch + 1,
                        scheduler=lr_scheduler,
                    )

        return 1
