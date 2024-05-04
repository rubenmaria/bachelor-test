import json
import os
from typing import Callable
from random import sample
import numpy as np
from numpy._typing import NDArray
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch
import random


def generate_embeddings_TSNE(
    output_path: str,
    input_path: str,
    perplexity: int
) -> None:
    generate_low_dimensional(
        f"{output_path}_low.json",
        input_path,
        TSNE(perplexity=perplexity)
    )


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


def calculate_standard_deviation_sentence_transfomer(
    output_dir: str,
    output_name: str,
    path: str,
    text_count: int,
    n: int
) -> None: 
    generate_high_dimensional_n(output_dir, output_name, path, text_count, n)
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
    path: str,
    text_count: int,
    n: int
) -> None:
    embeddings = dict(sample(list(load_text_data(path).items()), text_count))
    for i in range(n):
        output_path = get_embeddings_path_n(output_dir, output_name, i)
        print(f"generating: {output_path}...")
        dump_embeddings(
            output_path, 
            {k : text_to_embedding(v) for k,v in embeddings.items()}
        )
    
def load_text_data(path: str) -> dict[str,str]:
    with open(path, "r") as f:
        return json.load(f)

def get_embeddings_path_n(dir_path: str, name: str, i: int) -> str:
    return os.path.join(dir_path, f"{name}-{i}.json")

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
    for i, vectors in enumerate(embeddings):
        distances = euclidean_distances(vectors)
        distance_vectors.append(distances)
        #print(f"distances from {i}: \n", distances)
    return distance_vectors

def get_standard_deviation(distances: list[NDArray]) -> NDArray:
    distance_matrix = np.array(distances)
    return np.std(distance_matrix, axis=1)
 

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
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = np.array(model.encode(function, convert_to_numpy=True))
    return embedding

def sentences_to_embedding(function: str) -> NDArray:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    lines = function.split("\n")
    embedding_raw = np.array(model.encode(lines, convert_to_numpy=True))
    embedding = np.array(list())
    for emb in embedding_raw:
        embedding = np.concatenate((embedding, emb))
    print(len(embedding))
    return embedding

def chunks(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
