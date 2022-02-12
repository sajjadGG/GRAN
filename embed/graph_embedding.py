from embed.GraphEmbedder import FeatherEmbedder, GraphEmbedder
from embed.simMeasures import cosine_measure
from typing import AnyStr, Callable
import networkx as nx
import torch
import dgl
from embed.gin_evaluation import (
    load_feature_extractor,
    GINMetric,
)


# TODO: polymorphism , Strategy Pattern
def get_augmented_graphs(
    ref_graphs: list,
    generated_graphs: list,
    sim_measure: Callable = cosine_measure,
    n_aug: int = 10,
) -> list:
    n_ref = len(ref_graphs)
    n_gen = len(generated_graphs)
    assert n_gen > n_aug

    graph_embeddings = get_embedding(
        [
            nx.relabel_nodes(e, {v: k for k, v in enumerate(e.nodes)})
            for e in ref_graphs + generated_graphs
        ]
    )
    ref_embeddings = graph_embeddings[:n_ref, :]
    gen_embeddings = graph_embeddings[n_ref:, :]

    selected = sorted(
        [(i, generated_graphs[i], ge) for i, ge in enumerate(gen_embeddings)],
        key=lambda x: sim_measure(ref_embeddings, x[2]),
        reverse=True,
    )[:n_aug]

    return [x[1] for x in selected], [
        sim_measure(ref_embeddings, x[2]) for x in selected
    ]


def get_augmented_graphs_with_threshold(
    ref_graphs: list,
    generated_graphs: list,
    sim_measure: Callable = cosine_measure,
    sim_threshold: float = 0.6,
) -> list:
    n_ref = len(ref_graphs)

    # graph_embeddings = get_embedding(
    #     [
    #         nx.relabel_nodes(e, {v: k for k, v in enumerate(e.nodes)})
    #         for e in ref_graphs + generated_graphs
    #     ]
    # )

    graph_embeddings = get_embedding_gin(ref_graphs + generated_graphs)
    ref_embeddings = graph_embeddings[:n_ref, :]
    gen_embeddings = graph_embeddings[n_ref:, :]

    selected = [
        (i, generated_graphs[i], ge, sim_measure(ref_embeddings, ge))
        for i, ge in enumerate(gen_embeddings)
    ]

    return [x[1] for x in selected if x[3] > sim_threshold], [x[3] for x in selected]


def get_embedding(graphs: list):
    embedder = FeatherEmbedder()
    embedder.fit(graphs)
    return embedder.embed_train()


def get_embedding_gin(graphs: list):
    # TODO: ill practice :(

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    graphs_dgl = [
        dgl.DGLGraph(g).to(device) for g in graphs
    ]  # Convert graphs to DGL from NetworkX
    gin = load_feature_extractor(device=device)
    embedder = GINMetric(gin)

    # embedder.fit(graphs)
    return embedder.get_embed(graphs_dgl)
