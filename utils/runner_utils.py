import networkx as nx
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from utils.train_helper import load_model
from model import *


def get_optimizer(conf, params):
    if conf.optimizer == "SGD":
        optimizer = optim.SGD(
            params,
            lr=conf.lr,
            momentum=conf.momentum,
            weight_decay=conf.wd,
        )
    elif conf.optimizer == "Adam":
        optimizer = optim.Adam(params, lr=conf.lr, weight_decay=conf.wd)
    else:
        raise ValueError("Non-supported optimizer!")
    return optimizer


def get_graph(adj):
    """get a graph from zero-padded adj"""
    # remove all zeros rows and columns
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj)
    return G


def generate_graphs(
    model_file, node_pmf, gpus, device, model_conf, config, num=10, batch_size=16
):
    model = eval(model_conf.name)(config)

    model = load_model(model, model_file, device)
    model = nn.DataParallel(model, device_ids=gpus).to(device)
    model.eval()  # model call

    ### Generate Graphs
    A_pred = []
    num_nodes_pred = []

    gen_run_time = []
    for ii in tqdm(range(num)):
        with torch.no_grad():
            input_dict = {}
            input_dict["is_sampling"] = True
            input_dict["batch_size"] = batch_size
            input_dict["num_nodes_pmf"] = node_pmf
            A_tmp = model(input_dict)
            A_pred += [aa.data.cpu().numpy() for aa in A_tmp]
            num_nodes_pred += [aa.shape[0] for aa in A_tmp]

    graphs_gen = [get_graph(aa) for aa in A_pred]

    return graphs_gen
