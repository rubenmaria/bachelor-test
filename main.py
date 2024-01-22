import os
import clang.cindex
from clang.cindex import Cursor
from clang.cindex import SourceRange
from sentence_transformers import SentenceTransformer

GLIBC_PATH = "glibc"

def main():
    src_files = get_src_files(GLIBC_PATH)
    functions = get_n_functions(src_files, 100)
    print(function_to_embedding(functions[0]))
    

def function_to_embedding(function: str):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    lines = function.split("\n")
    return model.encode(lines, convert_to_numpy=True)


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
