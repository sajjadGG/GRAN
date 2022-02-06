import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cosine_measure(refs: list, candid: np.array) -> float:
    return np.mean(cosine_similarity(np.array(refs), candid.reshape(1, -1)))
