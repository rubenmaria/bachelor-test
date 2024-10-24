from visual import load_cluster, load_embeddings
from metric import k_nearest_neighbor_slow
from typing import Any
import random
import csv
import numpy as np
from numpy.typing import NDArray


def get_named_embeddings_from_cluster(
    cluster_path: str,
    summary_path: str
) -> dict[str, NDArray]:
    cluster_data = load_cluster(cluster_path)
    cluster_data.pop("colors")
    cluster_names = [name for names in cluster_data.values() for name in names]
    named_embeddings = load_embeddings(summary_path).items()
    return {name : v for name, v in named_embeddings if name in cluster_names}


def generate_surveys_csv(
    csv_path: str,
    cluster_path: str,
    embeddings_paths: list[str],
    sample_size: int,
    neares_neigbor_count: int
) -> None:
    named_neighbors: list[list[str]] = []
    for embeddings_path in embeddings_paths:
        named_embeddings = get_named_embeddings_from_cluster(
            cluster_path,
            embeddings_path
        )
        print(f"Population size: {len(named_embeddings)}")
        sample = dict(random.sample(list(named_embeddings.items()), sample_size))
        current_named_neighbors = get_named_neigbors(
            neares_neigbor_count,
            named_embeddings,
            list(sample.values())
        )
        named_neighbors += [n[::-1] for n in current_named_neighbors]
        print(len(named_neighbors))
    write_dict_to_csv(named_neighbors, csv_path)


def generate_survey_csv(
    csv_path: str,
    cluster_path: str,
    embeddings_path: str,
    sample_size: int,
    neares_neigbor_count: int
) -> None:
    named_embeddings = get_named_embeddings_from_cluster(
        cluster_path,
        embeddings_path
    )
    print(f"Population size: {len(named_embeddings)}")
    sample = dict(random.sample(list(named_embeddings.items()), sample_size))
    named_neighbors = get_named_neigbors(
        neares_neigbor_count,
        named_embeddings,
        list(sample.values())
    )
    named_neighbors = [n[::-1] for n in named_neighbors]
    write_dict_to_csv(named_neighbors, csv_path)


def get_named_neigbors(
    k: int,
    named_embeddings: dict[str, NDArray],
    sample: list[NDArray]
) -> list[list[str]]:
    return [get_k_nearst_neigbor_names(k, s, named_embeddings) for s in sample]


def get_k_nearst_neigbor_names(
    k: int,
    point: NDArray,
    named_embeddings: dict[str, NDArray]
) -> list[str]:
    embeddings = np.array(list(named_embeddings.values()))
    names = list(named_embeddings.keys())
    neighbor_indecies = k_nearest_neighbor_slow(k, point, embeddings)
    return [names[index] for index in neighbor_indecies]


def write_dict_to_csv(table: list[list[str]], csv_path: str) -> None:
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, values in enumerate(table):
           writer.writerow([key] + [v for v in values])
