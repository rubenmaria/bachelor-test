import fire
import os
import subprocess
import sys
import shutil
import numpy
from numpy.typing import NDArray
from embeddings import norm_vector, dump_embeddings
from distutils.dir_util import copy_tree


CODE2VEC_RELATIVE_TO_C2VEC = "code2vec"
C2VEC_RELATIVE_TO_CODE2VEC = ".."
MODEL_PATH = "models/c2vec-lite/saved_model_iter6.release"
TARGET_HISTOGRAM_FILE = "temp-target.histo.tgt.c2v"
ORIGIN_HISTOGRAM_FILE = "temp-origin.histo.ori.c2v"
PATH_HISTOGRAM_FILE = "temp-path.path.c2v"
GENERATE_VECTORS_SCRIPT = "generate-vectors-c.sh"


def main() -> None:
    fire.Fire({
        "vectors": generate_vectors
    })


def generate_vectors(input_dir: str, output_dir: str, output_name: str) -> None:
    copy_input_dir_to_code2vec(input_dir)
    os.chdir(CODE2VEC_RELATIVE_TO_C2VEC)
    run_generate_vector_script(input_dir, output_name)
    os.chdir(C2VEC_RELATIVE_TO_CODE2VEC)
    remove_input_dir_in_code2vec(input_dir)
    named_vectors = retrieve_vectors_from_files(output_name)
    dump_code2vec_embeddings(output_dir, output_name, named_vectors)
    remove_vector_files(output_name)

def dump_code2vec_embeddings(
    output_dir: str,
    output_name: str,
    named_embeddings: dict[str, NDArray]
) -> None:
    output_file_name = f"{output_name}-high.json"
    output_file_path = os.path.join(output_dir, output_file_name)
    dump_embeddings(output_file_path, named_embeddings)


def remove_vector_files(output_name: str) -> None:
    name_file = os.path.join(
        os.getcwd(),
        CODE2VEC_RELATIVE_TO_C2VEC,
        f"{output_name}.names.txt"
    )
    vector_file = os.path.join(
        os.getcwd(),
        CODE2VEC_RELATIVE_TO_C2VEC,
        f"{output_name}.test.c2v.vectors"
    )
    os.remove(name_file)
    os.remove(vector_file)


def retrieve_vectors_from_files(output_name: str) -> dict[str,NDArray]:
    name_file = os.path.join(
        CODE2VEC_RELATIVE_TO_C2VEC,
        f"{output_name}.names.txt"
    )
    vector_file = os.path.join(
        CODE2VEC_RELATIVE_TO_C2VEC,
        f"{output_name}.test.c2v.vectors"
    )
    names = parse_name_file(name_file)
    vectors = parse_vector_file(vector_file)
    return {name : vector for name, vector in zip(names, vectors) }


def parse_vector_file(path: str) -> list[NDArray]:
    vectors: list[NDArray] = []
    with open(path) as file:
        for line in file:
            vector = numpy.array(list(map(lambda x: float(x), line.split())))
            vectors.append(norm_vector(vector))
    return vectors


def parse_name_file(path: str) -> list[str]:
    names: list[str] = []
    with open(path) as file:
        for line in file:
            names.append(line.strip())
    return names

def run_generate_vector_script(input_dir: str, output_name: str) -> None:
    subprocess.run(["sh", GENERATE_VECTORS_SCRIPT, input_dir, output_name])

def copy_input_dir_to_code2vec(input_dir: str) -> None:
    dest = os.path.join(os.getcwd(), CODE2VEC_RELATIVE_TO_C2VEC, input_dir)
    print(f"copying '{input_dir}' to '{dest}'")
    copy_tree(input_dir, dest)

def remove_input_dir_in_code2vec(input_dir: str) -> None:
    root_input_dir = get_directory_one_above(
        os.path.join(os.getcwd(), CODE2VEC_RELATIVE_TO_C2VEC),
        os.path.join(os.getcwd(), CODE2VEC_RELATIVE_TO_C2VEC, input_dir)
    )
    shutil.rmtree(root_input_dir)


def get_directory_one_above(root: str, dir: str) -> str:
    current_dir = dir
    above_name = ""
    while root != current_dir:
        current_dir, above_name = os.path.split(current_dir)
        if current_dir == "/":
            raise RuntimeError("Invalid path combination")
    return os.path.join(current_dir, above_name)

if '__main__' == __name__:
    main()

