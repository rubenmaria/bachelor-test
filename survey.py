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

def get_data_range(keys: list[str]) -> tuple[int, int]:
    data_indecies = list(
        map( 
            lambda x: x[0],
            filter(
                lambda x: x[1].startswith("F") and x[1][-1].isdigit(),
                enumerate(keys)
            )
        )
    )
    return (min(data_indecies), max(data_indecies))

def evaluate_survey_results(csv_path: str) -> tuple[float, float, float]:
    META_DATA_ENTRIES_COUNT = 17
    (keys, results_one, results_two) = read_results_csv(csv_path)
    (min, max) = get_data_range(keys)
    data_one = results_one[min : max + 1]
    data_two = results_two[min : max + 1]
    sample_size = len(keys) - META_DATA_ENTRIES_COUNT
    code_llama_correct_count = 0
    function_names_correct_count = 0
    code2vec_corrcect_count = 0
    code_llama_sample_size = sample_size
    function_names_sample_size = sample_size
    code2vec_sample_size = sample_size

    
    assert len(data_one) == sample_size
    assert len(data_two) == sample_size

    for (index, result) in enumerate(data_one + data_two):
        if result == "":
            if index % 3 == 0:
                code_llama_sample_size -= 1
            elif index % 3 == 1:
                function_names_sample_size -= 1
            else:
                code2vec_sample_size -= 1
            continue

        is_result_correct_number = int(result) % 2

        if index % 3 == 0:
            code_llama_correct_count += is_result_correct_number
        elif index % 3 == 1:
            function_names_correct_count += is_result_correct_number
        else:
            code2vec_corrcect_count += is_result_correct_number
    
    return (
            code_llama_correct_count / code_llama_sample_size,
            function_names_correct_count / function_names_sample_size,
            code2vec_corrcect_count / code2vec_sample_size
    )


def read_results_csv(csv_path: str) -> tuple[list[str], list[str], list[str]]:
    with open(csv_path, 'r') as csv_file:
        results_reader = csv.reader(csv_file, delimiter=";")
        keys = next(results_reader)
        results_one = next(results_reader)
        results_two = next(results_reader)
        return (keys, results_one, results_two)
