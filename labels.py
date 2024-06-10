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
    functions = get_missing_functions(
        function_data_path,
        output_directory,
        output_name
    )
    for function in progress_bar(functions, prefix="Generating embeddings", decimals=6):
        if function.srccode is None:
            continue
        label = get_label_from_function(function, model_name)
        append_label(output_directory, output_name, label)


def get_missing_functions(
    functions_path: str,
    output_directory: str,
    output_name: str
    ) -> list[FuncData]:
    functions = load_function_data(functions_path)
    labels = load_label_data(
        get_output_file_path(output_directory, output_name)
    )
    processed_indecies = [l.index for l in labels]
    return [f for f in functions if f.index not in processed_indecies]


def load_function_data(path: str) -> list[FuncData]:
    with open(path, "rb") as f:
        return pickle.load(f)

def load_label_data(path: str) -> list[LabelData]:
    with open(path, "rb") as f:
        label_data = []
        while True:
            try:
                label_data.append(pickle.load(f))
            except EOFError:
                break
    return label_data

def get_label_from_function(func: FuncData, model_name: str) -> LabelData:
    summary = get_summary(func.srccode, model_name, PROMPT_PATH)
    embedding = text_to_embedding(summary)
    return LabelData(func.func_id, embedding, summary)


def get_output_file_path(directory: str, name: str) -> str:
    return os.path.join(directory, name + ".pkl")


def dump_labels(directory: str, name: str, labels: list[LabelData]) -> None:
    file_path = get_output_file_path(directory, name)
    with open(file_path, "wb") as file:
        pickle.dump(labels, file)


def append_label(directory: str, name: str, label: LabelData) -> None:
    file_path = os.path.join(directory, name + ".pkl")
    with open(file_path, "ab") as file:
        pickle.dump(label, file)
