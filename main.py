import json
import numpy as np
from numpy._typing import NDArray
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from matplotlib import pyplot as plt
import plotly.express as px
import torch
from transformers import AutoModel, AutoTokenizer

FUNCTION_COMMENT_EMBEDDINGS: str = "data/function-comment-embeddings.json"
FUNCTION_SUMMARY_EMBEDDINGS: str = "data/function-summary-embeddings.json"
FUNCTION_SUMMARY_CLAP_EMBEDDINGS: str = "data/summary-embeddings-clap.json"


def main():
    scatter_plot_embeddings(FUNCTION_SUMMARY_CLAP_EBEDDINGS)

def scatter_plot_embbedings(path: str):
    embeddings = load_embeddings(path)
    scatter_plot_named_embeddings(embeddings)

def scatter_plot_named_embeddings(embeddings: dict[str, NDArray]) -> None:
    x_coordinate = np.array(list(embeddings.values()))[:,0]
    y_coordinate = np.array(list(embeddings.values()))[:,1]
    fig = px.scatter(
        x=x_coordinate,
        y=y_coordinate,
        hover_name=list(embeddings.keys())
    )
    fig.show()


def load_embeddings(path: str) -> dict[str, NDArray]:
    embeddings : dict[str, list[float]]= {}
    with open(path) as f:
        embeddings = json.load(f)
    return {key : np.array(value) for key, value in embeddings.items()}

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

def get_function_names(functions: list[str]) -> list[str]:
    names = [x.split("\n")[0] + " " + x.split("\n")[1] for x in functions]
    return [x.strip("{") for x in names]

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

def draw_2d_embeddings(embedding: NDArray, names: list[str]):
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'grey', 'orange', 'purple'
    colors = colors + colors
    names  = names + names
    for i, c, n in zip(range(embedding.shape[0]), colors, names):
        if i < 10:
            plt.scatter(embedding[i, 0], embedding[i, 1], c=c, label=n)
        else:
            plt.scatter(embedding[i, 0], embedding[i, 1], c=c)
    plt.legend()
    plt.show()

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
