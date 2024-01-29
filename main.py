import os
import json
import numpy as np
from numpy._typing import NDArray
from sklearn.manifold import TSNE
import clang.cindex
from clang.cindex import Cursor
from clang.cindex import SourceRange
from sentence_transformers import SentenceTransformer
from matplotlib import pyplot as plt
GLIBC_PATH = "glibc"

def main():
    src_files = get_src_files(GLIBC_PATH)
    functions = get_n_functions(src_files, 10)
    functions_and_chatgpt = functions + get_chaptgpt_text()
    draw_2d_embeddings(
            get_low_dimension_embeddings(functions_and_chatgpt),
            get_function_names(functions)
    )

def get_chaptgpt_text() -> list[str]:
    with open("chatgpt.json", "r") as file:
        functions = json.load(file)["functions"]
    return [x["value"] for x in functions]



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



def get_src_files(glibc_path: str) -> list[str]:
    src_files = list()
    for root, _, files in os.walk(glibc_path):
        for file in files:
            current_file_path = os.path.join(root, file)
            if current_file_path.endswith(".c"):
                src_files.append(current_file_path)
    return src_files

def get_functions(file_path: str) -> list[str]:
    index = clang.cindex.Index.create()
    translation_unit = index.parse(file_path)
    functions = []
    for cursor in translation_unit.cursor.walk_preorder():
        if cursor.kind != clang.cindex.CursorKind.FUNCTION_DECL:
            continue
        if not cursor.is_definition():
            continue
        functions.append(get_function_definition(cursor.extent))
    return functions


def get_function_definition(location: SourceRange) -> str:
    filename = location.start.file.name
    with open(filename, 'r') as fh:
        contents = fh.read()
    return contents[location.start.offset: location.end.offset]


def get_n_functions(src_files: list[str], n: int) -> list[str]:
    functions = []
    src_files_index = 0
    while len(functions) < n:
        functions += get_functions(src_files[src_files_index])
        src_files_index += 1
    return functions[:n]


if __name__ == '__main__':
    main()
