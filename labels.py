from collections import namedtuple
import pickle
import os

from llm import get_summary, PROMPT_PATH
from embeddings import text_to_embedding
from visual import progress_bar

FuncData  = namedtuple(
    'FuncData', 
    ['func_id', 'srcabspath', 'basepath', 'srcline', 'name', 'srccode', 'pkg']
)
LabelData = namedtuple('LabelData', ['func_id', 'embedding', 'summary'])

def generate_labels(
    output_directory: str,
    output_name: str,
    function_data_path: str,
    model_name: str
    ) -> None:
    functions = load_function_data(function_data_path)
    for function in progress_bar(functions, prefix="Generating embeddings", decimals=6):
        if function.srccode is None:
            continue
        label = get_label_from_function(function, model_name)
        append_label(output_directory, output_name, label)


def load_function_data(path: str) -> list[FuncData]:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_label_from_function(func: FuncData, model_name: str) -> LabelData:
    summary = get_summary(func.srccode, model_name, PROMPT_PATH)
    embedding = text_to_embedding(summary)
    return LabelData(func.func_id, embedding, summary)


def dump_labels(directory: str, name: str, labels: list[LabelData]) -> None:
    file_path = os.path.join(directory, name + ".pkl")
    with open(file_path, "wb") as file:
        pickle.dump(labels, file)


def append_label(directory: str, name: str, label: LabelData) -> None:
    file_path = os.path.join(directory, name + ".pkl")
    with open(file_path, "ab") as file:
        pickle.dump(label, file)
