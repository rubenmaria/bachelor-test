import numpy as np
from numpy.typing import NDArray
from typing import Callable
from sklearn.metrics import ndcg_score
from sklearn.neighbors import NearestNeighbors
from pynndescent import NNDescent


def ranking_score(
    x: int,
    rank_x: int,
    other_ranking: list[int]
) -> float:
    assert rank_x < len(other_ranking)
    if other_ranking[rank_x] == x:
        return 1
    if x in other_ranking:
        return 0.5
    return 0
    

def ranking_diffrence(x: list[int], y: list[int]) -> float:
    assert len(x) == len(y)
    k = len(x)
    true_relevance_score = [1. for _ in range(k)]
    relevance_score = [ranking_score(x[i], i, y) for i in range(k)]
    score = (discounted_comulative_gain(relevance_score) 
            / discounted_comulative_gain(true_relevance_score))
    return score
       

def k_nearest_neighbor_slow(k: int, point: NDArray, space: NDArray) -> list[int]:
    neighbors = NearestNeighbors(n_neighbors=k, metric="cosine").fit(space)
    return neighbors.kneighbors([point], k, return_distance=False)[0].tolist()

def k_nearest_neighbor_fast(k: int, point: NDArray, space: NDArray) -> list[int]:
    nearest_neigbor = NearestNeighbor(space)
    assert isinstance(nearest_neigbor, NNDescent)
    return nearest_neigbor.query([point], k=k)[0][0].tolist()
   

def compare_embedding_spaces_k(
    k: int, 
    X: NDArray,
    Y: NDArray, 
    get_neighbors: Callable[[int,NDArray,NDArray], list[int]],
    aggregate: Callable[[list[float]], float]
    ) -> float:
    assert X.shape == Y.shape
    N = X.shape[0]
    all_neighbors = [
        (get_neighbors(k, X[i], X), get_neighbors(k, Y[i], Y)) 
        for i in range(N) 
    ]
    ranking_diffrences = [ranking_diffrence(x, y) for x,y in all_neighbors]
    return aggregate(ranking_diffrences)

def discounted_comulative_gain(scores: list[float]) -> float:
    k = len(scores)
    discounts = [np.log2(i+1) for i in range(1, k)]
    sum_elements = np.array(
        [relevance / discount for (relevance,discount) in zip(scores,discounts)]
    )
    return sum_elements.sum()


class NearestNeighbor:
    __algorithm = None

    def __new__(cls, space):
        if cls.__algorithm is None:
            cls.__algorithm = super(NearestNeighbor, cls).__new__(cls)
            cls.__algorithm = NNDescent(space, metric="cosine")
            cls.__algorithm.prepare()
        return cls.__algorithm
