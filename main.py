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
import copy
import random
import umap

FUNCTION_COMMENT_EMBEDDINGS: str = "data/function-comment-embeddings.json"
FUNCTION_SUMMARY_EMBEDDINGS: str = "data/function-summary-embeddings.json"
FUNCTION_SUMMARY_CLAP_EMBEDDINGS: str = "data/summary-embeddings-clap.json"
FUNCTION_NAME_EMBEDDINGS: str = "data/name-embeddings.json"

FUNCTION_COMMENTS: str = "data/function-comments.json"
FUCNTION_SUMMARIES: str = "data/function-summaries.json"
FUNCTION_NAMES: str = "data/function-names.json"


def main():
    generate_low_dimensional(
        "test-dimension-raw_",
        FUNCTION_COMMENTS,
        n=3,
        sample_size=1000,
        transformer=TSNE()
    )
    for i in range(3):
        plot_embeddings_from_path(f"test-dimension-raw_{i}.json")

def generate_low_dimensional(
        output_path: str,
        path: str,
        text_encoder: Callable[[str], NDArray] = lambda x : text_to_embedding(x),
        transformer: object = TSNE(),
        sample_size: int = None,
        n: int = 1
    ) -> None:
    data: dict[str,str] = {}
    values: list[str] = []
    with open(path) as f:
        data = json.load(f)
    values = list(data.values())
    values = random.sample(values, sample_size or len(values))
    for i in range(n):
        embeddings = make_same_dimensions([text_encoder(t) for t in values])
        embeddings = transformer.fit_transform(np.array(embeddings))
        named_embeddings = {
            list(data.keys())[i] : embeddings[i].tolist() for i in range(len(values))
        }
        with open(f"{output_path}{i+1}", "w") as f:
            json.dump(named_embeddings, f, indent=2)

def scatter_plot_clusters_from_path(path: str):
    embeddings = load_embeddings(path)
    labels = cluster_embeddings(np.array(list(embeddings.values())))
    scatter_plot_clusters(embeddings, labels)

def scatter_plot_clusters(embeddings: dict[str, NDArray], labels: list[int]):
    x_coordinates = np.array(list(embeddings.values()))[:,0]
    y_coordinates = np.array(list(embeddings.values()))[:,1]
    fig = px.scatter(
        x=x_coordinates,
        y=y_coordinates,
        color=labels,
        hover_name=list(embeddings.keys()),
    )
    fig.show()

def cluster_embeddings(embeddings: NDArray, eps: int = 3, min_samples: int = 5) -> list[int]:
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    return clustering.labels_.tolist()

def plot_embeddings_from_path(
        path: str,
        highlight: list[str] = [],
        text_position: str = "center top",
        color: str = "#FF7F50",
        highlight_color: str = "#6495ED"
):
    embeddings = load_embeddings(path)
    highlight_color: list[str] = [
        highlight_color if x in highlight else color  for x in embeddings.keys()
    ]
    highlight_text: list[str] = [
            x if x in highlight else "" for x in embeddings.keys()
    ]
    scatter_plot_named_embeddings(
        embeddings,
        highlight_color,
        highlight_text,
        text_position
    )

def scatter_plot_named_embeddings(
        embeddings: dict[str, NDArray],
        highlight_color: list[str],
        highlight_text: list[str],
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
        opacity=0.7
    )
    fig.update_traces(textposition=text_position, textfont_size=18)
    fig.show()

def load_embeddings(path: str) -> dict[str, NDArray]:
    embeddings : dict[str, list[float]]= {}
    with open(path) as f:
        embeddings = json.load(f)
    return {key : np.array(value) for key, value in embeddings.items()}

def dump_summary_embeddings_clap(transformer: object):
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
            return_tensors='pt',
            padding=True
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
    lines = function.split("\n")
    embedding_raw = np.array(model.encode(lines, convert_to_numpy=True))
    embedding = np.array(list())
    for emb in embedding_raw:
        embedding = np.concatenate((embedding, emb))
    return embedding

def chunks(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

if __name__ == '__main__':
    main()
