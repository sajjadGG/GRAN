import networkx as nx
import numpy as np
from embed.feather import FEATHER, FEATHERG


class GraphEmbedder:
    def embed(self, g: nx.Graph) -> np.array:
        pass

    def fit(self, graphs: list) -> None:
        pass


class FeatherEmbedder(GraphEmbedder):
    def __init__(self) -> None:
        self.model = FEATHERG()

    def fit(self, graphs: list) -> None:
        self.model.fit(graphs)

    def embed_train(self) -> np.array:
        return self.model.get_embedding()
