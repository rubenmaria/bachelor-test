#!/usr/bin/python
from argparse import ArgumentParser
import yaml
import subprocess
import pickle
import os
import shutil
import re
from distutils.dir_util import copy_tree

ASTMINER_TEMP_OUTPUT_DIR = "TEMP"
TOKENS_PATH = "TEMP/c/tokens.csv"
CONTEXTS_PATH = "TEMP/c/data/path_contexts.c2s"
ASTMINER_CONFIG_NAME = "astminer-config.yaml"
ASTMINER_CONFIG_PATH = "CExtractor/astminer/astminer-config.yaml"
ASTMINER_SCRIPT_NAME = "cli.sh"
CODE2VEC_RELATIVE_TO_ASTMINER = "../../"
ASTMINER_RELATIVE_TO_CODE2VEC = "CExtractor/astminer"


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "-maxlen",
        "--max_path_length",
        dest="max_path_length",
        required=False,
        default=8
    )
    parser.add_argument(
        "-maxwidth",
        "--max_path_width",
        dest="max_path_width",
        required=False,
        default=2
    )
    parser.add_argument(
        "-ofile_name",
        "--ofile_name",
        dest="ofile_name",
        required=True
    )
    parser.add_argument(
        "-ofile_name_names",
        "--ofile_name_names",
        dest="ofile_name_names",
        required=True
    )
    parser.add_argument("-dir", "--dir", dest="dir", required=True)
    args = parser.parse_args()
    generate_context_paths(
        args.dir,
        int(args.max_path_length),
        int(args.max_path_width),
        args.ofile_name,
        args.ofile_name_names
    )


def generate_context_paths(
    input_dir: str,
    max_path_length: int,
    max_path_width: int,
    output_file: str,
    output_file_names: str
) -> None:
    generate_astminer_config(input_dir, max_path_length, max_path_width)
    generate_context_paths_with_astminer(input_dir)
    generate_code2vec_context_paths(output_file, output_file_names)
    remove_tempory_dir()


def remove_tempory_dir() -> None:
    temp_dir = os.path.join(os.getcwd(), ASTMINER_TEMP_OUTPUT_DIR)
    print(f"Removing temporary directory: '{temp_dir}'")
    shutil.rmtree(temp_dir)


def generate_context_paths_with_astminer(input_dir: str) -> None:
    setup_astminer(input_dir)
    run_astminer()
    cleanup_astminer(input_dir)


def run_astminer() -> None:
    print(f"starting script: {ASTMINER_SCRIPT_NAME} {ASTMINER_CONFIG_NAME}")
    result = subprocess.run(
        ["sh", ASTMINER_SCRIPT_NAME, ASTMINER_CONFIG_NAME],
        stdout=subprocess.PIPE
    )
    print(result.stdout.decode("utf-8"))


def setup_astminer(input_dir: str) -> None:
    trainings_dir = os.path.join(os.getcwd(), input_dir)
    astminer_dir = os.path.join(
        os.getcwd(),
        ASTMINER_RELATIVE_TO_CODE2VEC,
        input_dir
    )
    print(f"copying '{trainings_dir}' to '{astminer_dir}'")
    copy_tree(trainings_dir, astminer_dir)
    print(f"changing directory to '{ASTMINER_RELATIVE_TO_CODE2VEC}'")
    os.chdir(ASTMINER_RELATIVE_TO_CODE2VEC)


def cleanup_astminer(input_dir: str):
    trainings_dir_full = os.path.join(os.getcwd(), input_dir)
    trainings_dir = get_directory_one_above(os.getcwd(), trainings_dir_full)
    print(f"remove temporary trainings data: {trainings_dir}")
    shutil.rmtree(trainings_dir)
    print(f"changing directory to '{CODE2VEC_RELATIVE_TO_ASTMINER}'")
    os.chdir(CODE2VEC_RELATIVE_TO_ASTMINER)


def get_directory_one_above(root: str, dir: str) -> str:
    current_dir = dir
    above_name = ""
    while root != current_dir:
        print(f"root[{root}] == current_dir[{current_dir}] ")
        current_dir, above_name = os.path.split(current_dir)
    return os.path.join(current_dir, above_name)


def generate_astminer_config(
    input_dir: str,
    max_path_len: int,
    max_path_width: int
) -> None:
    output_dir = os.path.join(
        CODE2VEC_RELATIVE_TO_ASTMINER,
        ASTMINER_TEMP_OUTPUT_DIR
    )
    astminer_config = {
        'inputDir': input_dir,
        'outputDir': output_dir,
        'parser': {'name': 'fuzzy', 'languages': ['c']},
        'filters': [{'name': 'by tree size', 'maxTreeSize': 1000}],
        'label': {'name': 'function name'},
        'storage': {
            'name': 'code2vec',
            'maxPathLength': max_path_len,
            'maxPathWidth': max_path_width
        }
    }
    print(f"Generating astminer config: '{ASTMINER_CONFIG_PATH}' ...")
    with open(ASTMINER_CONFIG_PATH, 'w') as file:
        yaml.dump(
            astminer_config,
            file,
            default_flow_style=False,
            sort_keys=False
        )


def print_data_dict(path: str) -> None:
    with open(path, 'rb') as file:
        word_to_count = pickle.load(file)
        path_to_count = pickle.load(file)
        target_to_count = pickle.load(file)
        num_training_examples = pickle.load(file)
    print(f"token counts: {word_to_count}")
    print(f"path context counts: {path_to_count}")
    print(f"target counts: {target_to_count}")
    print(f"training examples: {num_training_examples}")


def get_token_table(token_file_path: str) -> dict[int, str]:
    tokens: dict[int, str] = {}
    with open(token_file_path, "r") as f:
        lines = f.readlines()
        lines.pop(0)
        for line in lines:
            token_tuple = line.split(",")
            current_token = token_tuple[1].strip("\n").split()[0]
            current_token_id = int(token_tuple[0])
            tokens[current_token_id] = current_token
    return tokens


def parse_astimer_path_contexts(
    path: str
) -> tuple[dict[str, list[tuple[int, int, int]]],list[str]]:
    context_paths: dict[str, list[tuple[int, int, int]]] = {}
    original_names: list[str] = []
    with open(path, "r") as f:
        for context_path_raw in f:
            context_path = context_path_raw.split(" ")
            normalized_name = context_path.pop(0)
            original_name = context_path.pop(0)
            if not normalized_name in context_paths.keys():
                original_names.append(original_name)
                context_paths[normalized_name] = [
                    (int((pth := pths.split(","))[0]), int(pth[1]), int(pth[2]))
                    for pths in context_path
                ]
    return (context_paths, original_names)


def remove_orignal_name_path_contexts(
        context_paths: dict[str, list[tuple[int, int, int]]]
) -> tuple[dict[str, list[tuple[int, int, int]]],list[str]]:
    new_context_paths: dict[str, list[tuple[int, int, int]]] = {}
    original_names: list[str] = []
    for name, value in context_paths.items():
        if not (match := re.search("\\[(.+)\\]", name)):
            continue
        original_name = match.group(1)
        new_name = re.sub("\\[.+\\]", "", name)
        if new_name not in new_context_paths.keys():
            new_context_paths[new_name] = value
            original_names.append(original_name)
    return (new_context_paths, original_names)


def replace_token_id_with_token(
    astminer_paths: dict[str, list[tuple[int, int, int]]],
    token_table: dict[int, str],
) -> dict[str, list[tuple[str, int, str]]]:
    context_paths: dict[str, list[tuple[str, int, str]]] = {}
    for (name, paths) in astminer_paths.items():
        context_paths[name] = [
            (token_table[pth[0]], pth[1], token_table[pth[2]]) for pth in paths
        ]
    return context_paths


def dump_raw_code2vec_context_paths(
    output_path: str,
    context_paths: dict[str, list[tuple[str, int, str]]]
) -> None:
    context_paths_str = ""
    for name, context_path in context_paths.items():
        context_paths_str += ' '.join(
            [name] + [f'{p[0]},{p[1]},{p[2]}' for p in context_path]
        )
        context_paths_str += "\n"
    context_paths_str = context_paths_str.strip("\n")
    with open(output_path, "w") as file:
        file.write(context_paths_str)

def dump_original_name_list(
    output_path: str,
    names: list[str]
) -> None:
    with open(output_path, "w") as file:
        for index,name in enumerate(names):
            if index < len(names) - 1:
                file.write(name + "\n")
            else:
                file.write(name)


def generate_code2vec_context_paths(
    output_path_paths: str,
    output_path_names: str,
) -> None:
    print("converting astminer output to code2vec format...")
    token_table = get_token_table(TOKENS_PATH)
    astminer_context_paths, names = parse_astimer_path_contexts(CONTEXTS_PATH)
    code2vec_context_paths = replace_token_id_with_token(
        astminer_context_paths,
        token_table
    )
    dump_raw_code2vec_context_paths(output_path_paths, code2vec_context_paths)
    dump_original_name_list(output_path_names, names)


if __name__ == '__main__':
    main()
