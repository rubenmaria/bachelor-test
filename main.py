import json
from typing import Callable
import numpy as np
from numpy._typing import NDArray
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import plotly.express as px
import torch
import random

FUNCTION_COMMENTS: str = "data/comments-with-deduction.json"
FUNCTION_SUMMARIES: str = "data/summaries.json"
FUNCTION_NAMES: str = "data/names.json"

FUNCTION_COMMENT_EMBEDDINGS_HIGH: str = "data/comment-embeddings-high.json"
FUNCTION_SUMMARY_EMBEDDINGS_HIGH: str = "data/summary-embeddings-high.json"
FUNCTION_NAME_EMBEDDINGS_HIGH: str = "data/name-embeddings-high.json"

FUNCTION_COMMENT_EMBEDDINGS_LOW: str = "data/comment-embeddings-low.json"
FUNCTION_SUMMARY_EMBEDDINGS_LOW: str = "data/summary-embeddings-low.json"
FUNCTION_NAME_EMBEDDINGS_LOW: str = "data/name-embeddings-low.json"

FUNCTION_SUMMARY_CLAP_EMBEDDINGS: str = "data/summary-embeddings-clap.json"

def main():
    plot_clusters_from_path(FUNCTION_NAME_EMBEDDINGS_LOW, "data/cluster.json", "hhaha")

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

def plot_clusters_from_path(path: str,  cluster_path: str, title: str):
    cluster_data = load_cluster(cluster_path)
    colors = cluster_data.pop("colors")
    cluster_names = [name for names in cluster_data.values() for name in names]
    embeddings = {k:v for k, v in load_embeddings(path).items() if k in cluster_names}
    
    labels: list[str] = []
    for label_name in embeddings.keys():
        for color_index, names in enumerate(cluster_data.values()):
            if label_name in names:
                labels.append(colors[color_index])
                break

    x_coordinates = np.array(list(embeddings.values()))[:,0]
    y_coordinates = np.array(list(embeddings.values()))[:,1]
    fig = px.scatter(
        x=x_coordinates,
        y=y_coordinates,
        color=labels,
        hover_name=list(embeddings.keys()),
        title=title
    )
    fig.show()
    


def load_cluster(path: str) -> dict[str,list[str]]:
    with open(path) as f:
        return json.load(f)


def plot_clusters(
    embeddings: dict[str, NDArray],
    labels: list[int],
    title: str
):
    x_coordinates = np.array(list(embeddings.values()))[:,0]
    y_coordinates = np.array(list(embeddings.values()))[:,1]
    fig = px.scatter(
        x=x_coordinates,
        y=y_coordinates,
        color=labels,
        hover_name=list(embeddings.keys()),
        title=title
    )
    fig.show()

def cluster_embeddings(
    embeddings: NDArray,
    eps: int = 3,
    min_samples: int = 5
    ) -> list[int]:
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    return clustering.labels_.tolist()

def generate_cluster_from_path(output_path: str , input_path: str) -> None:
    embeddings = load_embeddings(input_path)
    names = list(embeddings.keys())
    cluster = cluster_embeddings(np.array(list(embeddings.values())))
    
    named_cluster = []
    for element in set(cluster):
        named_cluster.append(
            (
                f"{element}", 
                [names[i] for i, x in enumerate(cluster) if element == x]
            )
        )
    with open(output_path, "w") as f:
        json.dump(dict(named_cluster), f, indent=2)

def plot_embeddings_from_path(
        path: str,
        title: str = "title",
        highlight: list[str] = [],
        text_position: str = "top center",
        color: str = "#FF7F50",
        highlight_color: str = "#6495ED"
    ) -> None:
    embeddings = load_embeddings(path)
    highlight_colors: list[str] = [
        highlight_color if x in highlight else color  for x in embeddings.keys()
    ]
    highlight_text: list[str] = [
            x if x in highlight else "" for x in embeddings.keys()
    ]
    scatter_plot_named_embeddings(
        embeddings,
        highlight_colors,
        highlight_text,
        title,
        text_position
    )

def scatter_plot_named_embeddings(
        embeddings: dict[str, NDArray],
        highlight_color: list[str],
        highlight_text: list[str],
        title: str,
        text_position: str = "center top"
) -> None:
    x_coordinates = np.array(list(embeddings.values()))[:,0]
    y_coordinates = np.array(list(embeddings.values()))[:,1]
    fig = px.scatter(
        x=x_coordinates,
        y=y_coordinates,
        color=highlight_color,
        hover_name=list(embeddings.keys()),
        text=highlight_text,
        opacity=0.7,
        title=title
    )
    fig.update_traces(textposition=text_position, textfont_size=18)
    fig.show()

def load_embeddings(path: str) -> dict[str, NDArray]:
    embeddings : dict[str, list[float]]= {}
    with open(path) as f:
        embeddings = json.load(f)
    return {key : np.array(value) for key, value in embeddings.items()}

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
    print(len(embedding))
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

if __name__ == '__main__':
    main()
