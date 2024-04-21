import fire
import os
import subprocess
import sys
import shutil
from distutils.dir_util import copy_tree

CODE2VEC_RELATIVE_TO_C2VEC = "code2vec/"
MODEL_PATH = "models/c2vec-lite/saved_model_iter6.release"
TARGET_HISTOGRAM_FILE = "temp-target.histo.tgt.c2v"
ORIGIN_HISTOGRAM_FILE = "temp-origin.histo.ori.c2v"
PATH_HISTOGRAM_FILE = "temp-path.path.c2v"
GENERATE_VECTORS_SCRIPT = "generate-vectors-c.sh"

sys.path.append(os.path.join(os.getcwd(), CODE2VEC_RELATIVE_TO_C2VEC))


def main() -> None:
    fire.Fire({
        "vectors": generate_vectors
    })


def generate_vectors(input_dir: str, output_name: str) -> None:
    dest = os.path.join(os.getcwd(), CODE2VEC_RELATIVE_TO_C2VEC, input_dir)
    print(f"copying '{input_dir}' to '{dest}'")
    copy_tree(input_dir, dest)
    os.chdir(CODE2VEC_RELATIVE_TO_C2VEC)
    subprocess.run(["sh", GENERATE_VECTORS_SCRIPT, input_dir, output_name])
    shutil.rmtree(dest)



if '__main__' == __name__:
    main()

