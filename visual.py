import json
import numpy as np
from numpy._typing import NDArray
from sklearn.cluster import DBSCAN
import plotly.express as px


def load_embeddings(path: str) -> dict[str, NDArray]:
    embeddings : dict[str, list[float]]= {}
    with open(path) as f:
        embeddings = json.load(f)
    return {key : np.array(value) for key, value in embeddings.items()}


def plot_clusters_from_path(
    path: str,
    cluster_path: str,
    title: str,
    show_category: bool = True,
    save_file: bool = False
) -> None:
    cluster_data = load_cluster(cluster_path)
    colors = cluster_data.pop("colors")
    cluster_names = [name for names in cluster_data.values() for name in names]
    embeddings = {k:v for k, v in load_embeddings(path).items() if k in cluster_names}
    
    labels: list[str] = []
    symbols: list[str] = []
    for label_name in embeddings.keys():
        for color_index, (category, names) in enumerate(cluster_data.items()):
            if label_name in names:
                labels.append(colors[color_index])
                symbols.append(category)
                break

    x_coordinates = np.array(list(embeddings.values()))[:,0]
    y_coordinates = np.array(list(embeddings.values()))[:,1]

    assert len(labels) == x_coordinates.shape[0] 
    assert len(labels) == y_coordinates.shape[0]
    assert len(symbols) == x_coordinates.shape[0]
    assert len(symbols) == y_coordinates.shape[0]

    fig = px.scatter(
        x=x_coordinates,
        y=y_coordinates,
        color_discrete_sequence=colors,
        color=labels,
        symbol=symbols if show_category else None,
        hover_name=list(embeddings.keys()),
        title=title
    )
    if save_file:
        output_path = path.removesuffix(".json").replace("/", "-") + ".html"
        fig.write_html(output_path)
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
        text_position: str = "top center"
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


def cluster_missing(path: str, output_path: str) -> None:
    cluster_data = load_cluster(path)
    cluster_data.pop("colors")
    names_in_cluster = set([name for names in cluster_data.values() for name in names])
    all_names = set(load_embeddings(FUNCTION_NAME_EMBEDDINGS_LOW).keys())
    diffrence = list(all_names - names_in_cluster)
    with open (output_path, "w+") as f:
        json.dump({"functions: ": diffrence}, f, indent=2)


def plot_triplet_from_path(
    path: str,
    triple_names: tuple[str,str,str],
    text_position: str = "top center"
) -> None:
    named_embeddings = load_embeddings(path)
    triplet = {k : v for k,v in named_embeddings.items() if k in triple_names}
    size = 30
    x_coordinates = np.array(list(triplet.values()))[:,0]
    y_coordinates = np.array(list(triplet.values()))[:,1]
    fig = px.scatter(
        x=x_coordinates,
        y=y_coordinates,
        hover_name=(names := list(triplet.keys())),
        text=names,
        title=f"Triple: {names[0]}, {names[1]}, {names[2]}",
        size=[size for _ in range(3)]
    )
    fig.update_traces(textposition=text_position, textfont_size=40)
    fig.show()

