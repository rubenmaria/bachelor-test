import json
import os
from typing import Callable
from random import sample
import numpy as np
from numpy._typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from text_transformer import TextTransformer
from transformers import AutoModel, AutoTokenizer
from llm import get_summaries, PROMPT_PATH, get_embbeding_from_llama
from metric import compare_embedding_spaces_k, k_nearest_neighbor
from collections import OrderedDict
import torch
import random

def compare_embeddings_simple(k: int, path_x:str, path_y: str) -> float:
    named_embeddings_x = load_embeddings(path_x)
    named_embeddings_y = load_embeddings(path_y)
    keys_in_both = named_embeddings_x.keys() & named_embeddings_y.keys()
    named_embeddings_x = {
        k:v for k,v in named_embeddings_x.items() if k in keys_in_both
    }
    named_embeddings_y = {
        k:v for k,v in named_embeddings_y.items() if k in keys_in_both
    }
    named_embeddings_x = OrderedDict(sorted(named_embeddings_x.items()))
    named_embeddings_y = OrderedDict(sorted(named_embeddings_y.items()))
    X = np.array([x for (k,x) in named_embeddings_x.items() if k in keys_in_both])
    Y = np.array([y for y in named_embeddings_y.values()])
    print(X.shape, Y.shape)
    return compare_embedding_spaces_k(k, X, Y, k_nearest_neighbor, np.mean)


def generate_llm_TSNE(
    definition_path: str,
    model: str,
    output_dir: str,
    output_name: str
) -> None:
    output_path = os.path.join(output_dir, output_name + ".json")
    definitions = load_text_data(definition_path)
    embeddings = [
        get_embbeding_from_llama(v, model, PROMPT_PATH)
        for v in definitions.values()
    ]
    embeddings = transformer.fit_transform(np.array(embeddings)).tolist()
    named_embeddings = {k:v for k,v in zip(definitions.keys(), embeddings)}
    dump_embeddings(output_path, named_embeddings)

def generate_embeddings_TSNE(
    input_path: str,
    output_dir: str,
    output_name: str,
    perplexity: int
) -> None:
    output_path = os.path.join(output_dir, output_name)
    generate_low_dimensional(
        f"{output_path}.json",
        input_path,
        TSNE(perplexity=perplexity)
    )

def dump_text_data(output_path: str, data: dict[str,str]) -> None:
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def dump_embeddings(output_path: str, embeddings: dict[str, NDArray]) -> None:
    with open(output_path, "w") as f:
        json.dump({k : v.tolist() for k,v in embeddings.items()}, f, indent=2)


def load_embeddings(path: str) -> dict[str, NDArray]:
    embeddings : dict[str, list[float]]= {}
    with open(path) as f:
        embeddings = json.load(f)
    return {key : np.array(value) for key, value in embeddings.items()}


def generate_embedding_sample(output_path: str, path: str, n: int) -> None:
    data: dict[str,list[float]] = {}
    with open(path) as f:
        data = json.load(f)
    sample = random.sample(list(data.items()), n)
    sample_embeddings = { k : v for (k,v) in sample }
    with open(output_path, "w") as f:
        json.dump(sample_embeddings, f, indent=2)


def generate_low_dimensional(
        output_path: str,
        path: str,
        transformer,
    ) -> None:
    data: dict[str,list[float]] = {}
    embeddings: list[NDArray] = []
    names: list[str] = []
    with open(path) as f:
        data = json.load(f)
    names = list(data.keys())
    embeddings = [np.array(embedding) for embedding in data.values()]
    embeddings = transformer.fit_transform(np.array(embeddings)).tolist()
    named_embeddings = {
        names[i] : embeddings[i] for i in range(len(embeddings))
    }
    with open(output_path, "w") as f:
        json.dump(named_embeddings, f, indent=2)

def calculate_standard_deviation_llm(
    model_name: str,
    output_dir: str,
    output_name: str,
    functions_path: str,
    text_count: int,
    n: int
) -> None: 
    generate_summaries_n(
        model_name,
        output_dir,
        output_name,
        functions_path,
        text_count,
        n
    )
    embed_summaries_n(output_dir, output_name, n)
    (mean,dev) = get_standard_deviation_from_embeddings_n(output_dir, output_name, n)
    dump_standard_deviation(dev, output_dir, output_name)
    dump_mean(mean, output_dir, output_name)

def generate_summaries_n(
    model_name: str,
    output_dir: str,
    output_name: str,
    definition_path: str,
    defintion_count: int,
    n: int
    ) -> None:
    definitions = dict(sample(list(
        load_text_data(definition_path).items()), defintion_count)
    )
    for i in range(n):
        output_path = get_summaries_path_n(output_dir, output_name, i)
        print(f"generating: {output_path}...")
        dump_text_data(
            output_path,
            get_summaries(definitions, PROMPT_PATH, model_name)
        )

def embed_summaries_n(
    input_dir: str,
    input_name: str,
    n: int
    ) -> None:
    summaries = load_summaries_n(input_dir, input_name, n)
    for i, summary in enumerate(summaries):
        embedding_path = get_embeddings_path_n(input_dir, input_name, i)
        print(f"generating: {embedding_path}...")
        embedding = {
            k : text_to_embedding(v) for (k,v) in summary.items()
        }
        dump_embeddings(embedding_path, embedding)


def calculate_standard_deviation_sentence_transfomer(
    output_dir: str,
    output_name: str,
    text_data_path: str,
    text_count: int,
    n: int
) -> None: 
    text_sample = dict(sample(list(load_text_data(text_data_path).items()), text_count))
    generate_high_dimensional_n(output_dir, output_name, text_sample, n)
    (mean,dev) = get_standard_deviation_from_embeddings_n(output_dir, output_name, n)
    dump_standard_deviation(dev, output_dir, output_name)
    dump_mean(mean, output_dir, output_name)

    
def calculate_standard_deviation_from_embeddings(output_dir: str, output_name: str, n: int):
    (mean, dev) =  get_standard_deviation_from_embeddings_n(output_dir, output_name, n)
    dump_standard_deviation(dev, output_dir, output_name)
    dump_mean(mean, output_dir, output_name)


def dump_standard_deviation(
    standard_deviation: NDArray,
    output_dir: str,
    output_name: str
) -> None:
    path = os.path.join(output_dir, f"{output_name}-deviation.json")
    with open(path, "w") as f:
        json.dump({"standard deviation" : standard_deviation.tolist()}, f, indent=2)


def dump_mean(
    standard_deviation: NDArray,
    output_dir: str,
    output_name: str
) -> None:
    path = os.path.join(output_dir, f"{output_name}-mean.json")
    with open(path, "w") as f:
        json.dump({"mean" : standard_deviation.tolist()}, f, indent=2)


def generate_high_dimensional_n(
    output_dir: str,
    output_name: str,
    text_to_embed: dict[str, str],
    n: int
) -> None:
    for i in range(n):
        output_path = get_embeddings_path_n(output_dir, output_name, i)
        print(f"generating: {output_path}...")
        dump_embeddings(
            output_path, 
            {k : text_to_embedding(v) for k,v in text_to_embed.items()}
        )
    

def load_text_data(path: str) -> dict[str,str]:
    with open(path, "r") as f:
        return json.load(f)

def get_embeddings_path_n(dir_path: str, name: str, i: int) -> str:
    return os.path.join(dir_path, f"{name}-{i}.json")

def get_summaries_path_n(dir_path: str, name: str, i: int) -> str:
    return os.path.join(dir_path, f"{name}-summary-{i}.json")

def get_standard_deviation_from_embeddings_n(
    input_dir: str,
    input_name: str,
    n: int
) -> tuple[NDArray,NDArray]:
    embeddings = load_embeddings_n(input_dir, input_name, n)
    distances = get_pairwise_distance_n(embeddings)
    standard_deviations = get_standard_deviation(distances)
    mean = get_mean(distances)
    return (mean, standard_deviations)

def get_mean(distances: list[NDArray]) -> NDArray:
    distance_matrix = np.array(distances)
    return np.mean(distance_matrix, axis=1)

def get_pairwise_distance_n(embeddings: list[list[NDArray]]) -> list[NDArray]:
    distance_vectors: list[NDArray] = []
    for vectors in embeddings:
        distances = cosine_distances(vectors)
        distance_vectors.append(distances)
    return distance_vectors

def get_standard_deviation(distances: list[NDArray]) -> NDArray:
    distance_matrix = np.array(distances)
    return np.std(distance_matrix, axis=0)
 
def norm_vector(vector: NDArray) -> NDArray:
    return 1 / np.linalg.norm(vector) * vector

def load_embeddings_n(
    input_dir: str,
    input_name: str,
    n: int
) -> list[list[NDArray]]:
    embeddings: list[list[NDArray]] = []
    for i in range(n):
        embeddings_path = get_embeddings_path_n(input_dir, input_name, i)
        named_embeddings = load_embeddings(embeddings_path)
        embeddings.append([v for v in named_embeddings.values()])
    return embeddings

def load_summaries_n(
    input_dir: str,
    input_name: str,
    n: int
) -> list[dict[str,str]]:
    summaries: list[dict[str,str]] = []
    for i in range(n):
        summary_path = get_summaries_path_n(input_dir, input_name, i)
        named_summaries = load_text_data(summary_path)
        summaries.append(named_summaries)
    return summaries

def generate_high_dimensional(
        output_path: str,
        path: str,
        text_encoder: Callable[[str], NDArray] = lambda x : text_to_embedding(x),
    ) -> None:
    data: dict[str,str] = {}
    values: list[str] = []
    keys: list[str] = []
    with open(path) as f:
        data = json.load(f)
    values = list(data.values())
    keys   = list(data.keys())
    embeddings = make_same_dimensions([text_encoder(t) for t in values])
    named_embeddings = {
        keys[i] : embeddings[i].tolist() for i in range(len(values))
    }
    with open(output_path, "w") as f:
        json.dump(named_embeddings, f, indent=1)

def dump_summary_embeddings_clap(transformer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_tokenizer = AutoTokenizer.from_pretrained(
        "hustcw/clap-text",
        trust_remote_code=True
    )
    text_encoder = AutoModel.from_pretrained(
        "hustcw/clap-text",
        trust_remote_code=True
    ).to(device)

    summaries: dict[str,str] = {}
    with open("data/function-summaries.json") as f:
        summaries = json.load(f)
    text_embeddings = []
    for chunk in chunks(list(summaries.values()), 100):
        print(len(chunk))
        text_input = text_tokenizer(
            chunk,
            return_tensors='pt', padding=True
        ).to(device)
        text_embeddings += text_encoder(**text_input).tolist()
    embeddings = transformer.fit_transform(np.array(text_embeddings))
    summary_embeddings = {
        list(summaries.keys())[i] : embeddings[i].tolist()
        for i in range(len(summaries))
    }
    with open("summary-embeddings-raw-clap.json", "w") as f:
        json.dump(summary_embeddings, f)
    print(embeddings)

def make_same_dimensions(embeddings: list[NDArray]) -> list[NDArray]:
    max_dimension = max([e.shape[0] for e in embeddings])
    result = list()
    for e in embeddings:
        delta_dimension = max_dimension - e.shape[0]
        if delta_dimension > 0:
            result.append(np.concatenate((e,np.zeros(delta_dimension))))
        else:
            result.append(e)
    return result

def text_to_embedding(function: str) -> NDArray:
    model = TextTransformer()
    assert type(model) == SentenceTransformer
    embedding = np.array(model.encode(function, convert_to_numpy=True))
    return embedding

def sentences_to_embedding(function: str) -> NDArray:
    model = TextTransformer()
    assert type(model) == SentenceTransformer
    lines = function.split("\n")
    embedding_raw = np.array(model.encode(lines, convert_to_numpy=True))
    embedding = np.array(list())
    for emb in embedding_raw:
        embedding = np.concatenate((embedding, emb))
    print(len(embedding))
    return embedding

def chunks(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
