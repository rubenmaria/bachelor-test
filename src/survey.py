from visual import load_cluster, load_embeddings
from embeddings import get_k_nearst_neigbor_names
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

def append_survey_to_csv(
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
    append_dict_to_csv(named_neighbors, csv_path)

def append_dict_to_csv(table: list[list[str]], csv_path: str) -> None:
    current_key = 0
    with open(csv_path, 'r') as csv_file:
        last_line = csv_file.readlines()[-1]
        current_key = int(last_line.split(",")[0])
    print(current_key)
    with open(csv_path, 'a') as csv_file:
        writer = csv.writer(csv_file)
        for key, values in enumerate(table):
           writer.writerow([current_key + key + 1] + [v for v in values])


def get_named_neigbors(
    k: int,
    named_embeddings: dict[str, NDArray],
    sample: list[NDArray]
) -> list[list[str]]:
    return [get_k_nearst_neigbor_names(k, s, named_embeddings) for s in sample]


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

def evaluate_survey_seven(
    data: list[str]
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    comment_correct_count = 0
    names_correct_count = 0
    code2vec_corrcect_count = 0
    invalid_comment_sample = 0
    invalid_name_sample = 0
    invalid_code2vec_sample = 0
    size = len(data)

    for _ in range(47):
        result = data.pop(0)
        if result == "": 
            invalid_name_sample += 1
            continue
        names_correct_count += int(result) % 2

    for _ in range(24):
        result = data.pop(0)
        if result == "": 
            invalid_code2vec_sample += 1
            continue
        code2vec_corrcect_count += int(result) % 2

    for _ in range(70):
        result = data.pop(0)
        if result == "": 
            invalid_comment_sample += 1
            continue
        comment_correct_count += int(result) % 2
    
    print("="*20)
    print("Survey Eight:")
    print(f"size={size}")
    print(
        f"llama_correct={0}",
        f"names_correct={names_correct_count}",
        f"comment_correct={comment_correct_count}",
        f"code2vec_corrcect={code2vec_corrcect_count}" 
    )
    print(
        f"llama_invalid={0}",
        f"names_invalid={invalid_name_sample}",
        f"comment_invalid={invalid_comment_sample}",
        f"code2vec_invalid={invalid_code2vec_sample}" 
    )
    print("="*20)

    return (
        (0, names_correct_count, comment_correct_count, code2vec_corrcect_count),
        (0, invalid_name_sample, invalid_comment_sample, invalid_code2vec_sample) 
    )


def evaluate_survey_eight(
    data: list[str]
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    comment_correct_count = 0
    llama_correct_count = 0
    code2vec_corrcect_count = 0
    invalid_comment_sample = 0
    invalid_llama_sample = 0
    invalid_code2vec_sample = 0
    size = len(data)

    for _ in range(47):
        result = data.pop(0)
        if result == "": 
            invalid_llama_sample += 1
            continue
        llama_correct_count += int(result) % 2

    for _ in range(24):
        result = data.pop(0)
        if result == "": 
            invalid_code2vec_sample += 1
            continue
        code2vec_corrcect_count += int(result) % 2

    for _ in range(70):
        result = data.pop(0)
        if result == "": 
            invalid_comment_sample += 1
            continue
        comment_correct_count += int(result) % 2

    print("="*20)
    print("Survey Eight:")
    print(f"size={size}")
    print(
        f"llama_correct={llama_correct_count}",
        f"names_correct={0}",
        f"comment_correct={comment_correct_count}",
        f"code2vec_corrcect={code2vec_corrcect_count}" 
    )
    print(
        f"llama_invalid={invalid_llama_sample}",
        f"names_invalid={0}",
        f"comment_invalid={invalid_comment_sample}",
        f"code2vec_invalid={invalid_code2vec_sample}" 
    )
    print("="*20)
    
    return (
        (llama_correct_count, 0, comment_correct_count, code2vec_corrcect_count),
        (invalid_llama_sample, 0, invalid_comment_sample, invalid_code2vec_sample)
    )


def evaluate_survey_three_two(
    data: list[str]
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    llama_correct_count = 0
    names_correct_count = 0
    code2vec_corrcect_count = 0
    invalid_llama_sample = 0
    invalid_name_sample = 0
    invalid_code2vec_sample = 0

    for (index, result) in enumerate(data):
        if result == "":
            if index % 3 == 0:
                invalid_llama_sample += 1
            elif index % 3 == 1:
                invalid_name_sample += 1
            else:
                invalid_code2vec_sample += 1
            continue

        is_result_correct_number = int(result) % 2

        if index % 3 == 0:
            llama_correct_count += is_result_correct_number
        elif index % 3 == 1:
            names_correct_count += is_result_correct_number
        else:
            code2vec_corrcect_count += is_result_correct_number
    print("="*20)
    print("Survey Three and Two:")
    print(f"size={len(data)}")
    print(
        f"llama_correct={llama_correct_count}",
        f"names_correct={names_correct_count}",
        f"comment_correct={0}",
        f"code2vec_corrcect={code2vec_corrcect_count}" 
    )
    print(
        f"llama_invalid={invalid_llama_sample}",
        f"names_invalid={invalid_name_sample}",
        f"comment_invalid={0}",
        f"code2vec_invalid={invalid_code2vec_sample}" 
    )
    print("="*20)

    return (
        (llama_correct_count, names_correct_count, 0, code2vec_corrcect_count),
        (invalid_llama_sample, invalid_name_sample, 0, invalid_code2vec_sample)
    )




def evaluate_survey_results(csv_path: str) -> tuple[float, float, float, float]:
    META_DATA_ENTRIES_COUNT = 17
    results = read_results_csv(csv_path)
    keys = results.pop(0)
    sample_size = len(keys) - META_DATA_ENTRIES_COUNT
    (min, max) = get_data_range(keys)

    assert (max + 1) - min == sample_size
    
    print(len(results))
    survey_three = results[0][min:max+1]
    survey_two = results[1][min:max+1]
    survey_seven = results[2][min:max+1]
    survey_eight = results[3][min:max+1]

    survey_results_two_three = evaluate_survey_three_two(survey_three + survey_two)
    (
        llama_correct_count, names_correct_count, comment_correct_count, code2vec_corrcect_count
    ) = survey_results_two_three[0]
    (
        invalid_llama_sample, invalid_name_sample, invalid_comment_sample, invalid_code2vec_sample
    ) = survey_results_two_three[1]
    
    survey_results_seven = evaluate_survey_seven(survey_seven)
    survey_results_eight = evaluate_survey_eight(survey_eight)

    llama_correct_count += survey_results_seven[0][0] + survey_results_eight[0][0]
    names_correct_count += survey_results_seven[0][1] + survey_results_eight[0][1]
    comment_correct_count += survey_results_seven[0][2] + survey_results_eight[0][2]
    code2vec_corrcect_count += survey_results_seven[0][3] + survey_results_eight[0][3]
    
    invalid_llama_sample += survey_results_seven[1][0] + survey_results_eight[1][0]
    invalid_name_sample += survey_results_seven[1][1] + survey_results_eight[1][1]
    invalid_comment_sample += survey_results_seven[1][2] + survey_results_eight[1][2]
    invalid_code2vec_sample += survey_results_seven[1][3] + survey_results_eight[1][3]
    return (
        llama_correct_count / (sample_size - invalid_llama_sample),
        names_correct_count / (sample_size - invalid_name_sample),
        comment_correct_count / (sample_size - invalid_comment_sample),
        code2vec_corrcect_count / (sample_size - invalid_code2vec_sample)
    )


def read_results_csv(csv_path: str) -> list[list[str]]:
    with open(csv_path, 'r') as csv_file:
        results_reader = csv.reader(csv_file, delimiter=";")
        results = []
        for result in results_reader:
            results += [result]
        return results
