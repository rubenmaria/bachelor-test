import os
import subprocess
from code2vec.CExtractor.extract import generate_context_paths
from code2vec.preprocess import process_file
from code2vec.common import common


CODE2VEC_RELATIVE_TO_C2VEC = "code2vec/"
MODEL_PATH = "models/c2vec-lite/saved_model_iter6.release"
TARGET_HISTOGRAM_FILE = "temp-target.histo.tgt.c2v"
ORIGIN_HISTOGRAM_FILE = "temp-origin.histo.ori.c2v"
PATH_HISTOGRAM_FILE = "temp-path.path.c2v"


def generate_prediction_file(
    output_file_name: str,
    prediction_dir: str,
    max_path_length: int = 8,
    max_path_width: int = 2
) -> None:
    os.chdir(CODE2VEC_RELATIVE_TO_C2VEC)
    tmp_raw_file = get_temporary_name(prediction_dir)
    generate_context_paths(
        prediction_dir,
        max_path_length,
        max_path_width,
        tmp_raw_file
    )

    generate_path_histogram(output_file_name, PATH_HISTOGRAM_FILE)
    generate_origin_histogram(output_file_name, ORIGIN_HISTOGRAM_FILE)
    generate_target_histogram(output_file_name, TARGET_HISTOGRAM_FILE)

    word_to_count = common.load_vocab_from_histogram()
    path_to_count = common.load_vocab_from_histogram()

    process_file(
        tmp_raw_file,
        "test",
        output_file_name,
    )


def generate_target_histogram(input_path: str, output_path: str) -> None:
    subprocess.run(
        ["sh", f"cat {input_path} " + "| cut -d' ' -f1 | awk '{n[$0]++} END  \
            {for (i in n) print i,n[i]}" + f"> {output_path}"]
    )


def generate_origin_histogram(input_path: str, output_path: str) -> None:
    subprocess.run(
        [
            "sh", f"cat {input_path} | cut -d' ' -f2- | tr ' ' '\n'\
            | cut -d',' -f1,3 | tr ',' '\n' | " +
            "awk '{n[$0]++} END {for (i in n) print i,n[i]}' " +
            f"> {output_path}"
        ]
    )


def generate_path_histogram(input_path: str, output_path: str) -> None: 
    subprocess.run(
        [
         "sh", f"cat {input_path}" +
         "| cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f2 | awk '{n[$0]++} END \
         {for (i in n) print i,n[i]}'"
         + f"> {output_path}"
        ]
    )


def get_temporary_name(prediction_dir: str) -> str:
    return prediction_dir + "-tmp"

