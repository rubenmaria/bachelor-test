import json
import numpy as np
from numpy._typing import NDArray
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
import plotly.express as px
import torch
from transformers import AutoModel, AutoTokenizer

FUNCTION_COMMENT_EMBEDDINGS: str = "data/function-comment-embeddings.json"
FUNCTION_SUMMARY_EMBEDDINGS: str = "data/function-summary-embeddings.json"
FUNCTION_SUMMARY_CLAP_EMBEDDINGS: str = "data/summary-embeddings-clap.json"
FUNCTION_NAME_EMBEDDINGS: str = "data/name-embeddings.json"


def main():
    plot_embeddings_from_path(
        FUNCTION_COMMENT_EMBEDDINGS,
        ["lchmod", "seed48", "nrand48"],
        "top right"
    )
    #scatter_plot_clusters_from_path(FUNCTION_SUMMARY_CLAP_EMBEDDINGS)

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


def cluster_embeddings(embeddings: NDArray) -> list[int]:
    clustering = DBSCAN(eps=3, min_samples=5).fit(embeddings)
    return clustering.labels_.tolist()

def plot_embeddings_from_path(
        path: str,
        highlight: list[str],
        text_position: str
):
    embeddings = load_embeddings(path)
    highlight_color: list[str] = [
        "#6495ED" if x in highlight else "#FF7F50" for x in embeddings.keys()
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
        text_position: str
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

def dump_name_embeddings():
    names: list[str] = []
    with open("data/function-names.json") as f:
        names = json.load(f)["functions"]
    print(names)
    embeddings = get_low_dimension_embeddings(names)
    summary_embeddings = {
        names[i] : embeddings[i].tolist() for i in range(len(names))
    }
    with open("name-embeddings-raw.json", "w") as f:
        json.dump(summary_embeddings, f)

def dump_summary_embeddings():
    summaries: dict[str,str] = {}
    with open("data/function-summaries.json") as f:
        summaries = json.load(f)
    print(summaries)
    embeddings = get_low_dimension_embeddings(list(summaries.values()))
    summary_embeddings = {
        list(summaries.keys())[i] : embeddings[i].tolist()
        for i in range(len(summaries))
    }
    with open("summary-embeddings-raw.json", "w") as f:
        json.dump(summary_embeddings, f)

def dump_summary_embeddings_clap():
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
    embeddings = embedding_to_low_dimension(np.array(text_embeddings))
    summary_embeddings = {
        list(summaries.keys())[i] : embeddings[i].tolist()
        for i in range(len(summaries))
    }
    with open("summary-embeddings-raw-clap.json", "w") as f:
        json.dump(summary_embeddings, f)
    print(embeddings)

def dump_comment_embeddings():
    comments: dict[str,str] = {}
    with open("data/function-comments.json") as f:
        comments = json.load(f)
    print(comments)
    embeddings = get_low_dimension_embeddings(list(comments.values()))
    comments_embeddings = {
        list(comments.keys())[i] : embeddings[i].tolist()
        for i in range(len(comments))
    }
    with open("comment-embeddings.json", "w") as f:
        json.dump(comments_embeddings, f)

def get_low_dimension_embeddings(text: list[str]) -> NDArray:
    embeddings = make_same_dimensions([function_to_embedding(t) for t in text])
    return embedding_to_low_dimension(np.array(embeddings))

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

def function_to_embedding(function: str) -> NDArray:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    lines = function.split("\n")
    embedding_raw = np.array(model.encode(lines, convert_to_numpy=True))
    embedding = np.array(list())
    for emb in embedding_raw:
        embedding = np.concatenate((embedding, emb))
    return embedding

def embedding_to_low_dimension(embedding: NDArray) -> NDArray:
    return TSNE(perplexity=5).fit_transform(embedding)

def chunks(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

if __name__ == '__main__':
    main()
