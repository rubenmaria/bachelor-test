import fire
import os
import subprocess
import code2vec


CODE2VEC_RELATIVE_TO_C2VEC = "code2vec/"
MODEL_PATH = "models/c2vec-lite/saved_model_iter6.release"
TARGET_HISTOGRAM_FILE = "temp-target.histo.tgt.c2v"
ORIGIN_HISTOGRAM_FILE = "temp-origin.histo.ori.c2v"
PATH_HISTOGRAM_FILE = "temp-path.path.c2v"

def main() -> None:
    fire.Fire({
        "prediction-file": generate_prediction_file
    })


def generate_prediction_file(
    output_file_name: str,
    prediction_dir: str,
    max_path_length: int = 8,
    max_path_width: int = 2,
    word_vocab_size: int = 1301136,
    path_vocab_size: int = 911417,
    max_contexts: int = 200
) -> None:
    os.chdir(CODE2VEC_RELATIVE_TO_C2VEC)
    tmp_raw_file = get_temporary_name(prediction_dir)
    code2vec.generate_context_paths(
        prediction_dir,
        max_path_length,
        max_path_width,
        tmp_raw_file
    )

    generate_path_histogram(output_file_name, PATH_HISTOGRAM_FILE)
    generate_origin_histogram(output_file_name, ORIGIN_HISTOGRAM_FILE)
    generate_target_histogram(output_file_name, TARGET_HISTOGRAM_FILE)

    word_to_count = code2vec.common.load_vocab_from_histogram(
        ORIGIN_HISTOGRAM_FILE,
        start_from=1,
        max_size=word_vocab_size,
        return_counts=True   
    )
    path_to_count = code2vec.common.load_vocab_from_histogram(
        PATH_HISTOGRAM_FILE,
        start_from=1,
        max_size=path_vocab_size,
        return_counts=True
    )

    code2vec.process_file(
        tmp_raw_file,
        "test",
        output_file_name,
        word_to_count,
        path_to_count,
        max_contexts
    )

    os.remove(PATH_HISTOGRAM_FILE)
    os.remove(ORIGIN_HISTOGRAM_FILE)
    os.remove(TARGET_HISTOGRAM_FILE)


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

if '__main__' == __name__:
    main()

