import os
import json
import numpy as np
from numpy._typing import NDArray
from sklearn.manifold import TSNE
import clang.cindex
from clang.cindex import Cursor
from clang.cindex import SourceRange
from sentence_transformers import SentenceTransformer
from itertools import islice

GLIBC_PATH = "glibc"

def main():
    #dump_function_definitions(GLIBC_PATH)
    #dump_function_names(GLIBC_PATH)
    dump_function_comments(GLIBC_PATH)
    

def dump_function_definitions(glibc_path: str) -> None:
    src_files = get_src_files(GLIBC_PATH)
    symbols   = get_symbols("libm-libc-symbols.json")
    functions = get_all_function_definitions(src_files, symbols)
    with open('function-defintion-table.json', 'w') as f:
        json.dump(functions, f)

def dump_function_names(glibc_path: str) -> None:
    src_files = get_src_files(GLIBC_PATH)
    symbols   = get_symbols("libm-libc-symbols.json")
    functions = get_all_function_names(src_files, symbols)
    with open('function-name-table.json', 'w') as f:
        json.dump({"functions": functions}, f)

def dump_function_comments(glibc_path: str) -> None:
    src_files = get_src_files(glibc_path)
    symbols   = get_symbols("libm-libc-symbols.json")
    function_comments = get_all_function_comments(src_files, symbols)
    with open('function-comment-table.json', 'w') as f:
        json.dump(function_comments, f)
    

def get_symbols(path: str) -> list[str]:
    with open(path, 'r') as fh:
        symbols = json.load(fh)
        return symbols['symbols']

def get_src_files(glibc_path: str) -> list[str]:
    src_files = list()
    for root, _, files in os.walk(glibc_path):
        for file in files:
            current_file_path = os.path.join(root, file)
            if current_file_path.endswith(".c"):
                src_files.append(current_file_path)
    return src_files

def get_function_definitions(file_path: str, valid_symbols: list[str]) -> dict[str,str]:
    index = clang.cindex.Index.create()
    translation_unit = index.parse(file_path)
    functions = {}
    for cursor in translation_unit.cursor.walk_preorder():
        if cursor.kind != clang.cindex.CursorKind.FUNCTION_DECL:
            continue
        if not cursor.is_definition():
            continue
        symbol_name = cursor.displayname.split('(')[0]
        if symbol_name in valid_symbols:
            functions[cursor.displayname] = get_function_definition(cursor.extent)
    return functions

def get_function_names(file_path: str, valid_symbols: list[str]) -> list[str]:
    index = clang.cindex.Index.create()
    translation_unit = index.parse(file_path)
    functions = []
    for cursor in translation_unit.cursor.walk_preorder():
        if cursor.kind != clang.cindex.CursorKind.FUNCTION_DECL:
            continue
        if not cursor.is_definition():
            continue
        symbol_name = cursor.displayname.split('(')[0]
        if symbol_name in valid_symbols:
            functions.append(cursor.displayname)
    return functions


def get_function_comment(file_path: str, valid_symbols: list[str]) -> dict[str,str]:
    index = clang.cindex.Index.create()
    translation_unit = index.parse(file_path)
    functions = {}
    for cursor in translation_unit.cursor.walk_preorder():
        if cursor.kind != clang.cindex.CursorKind.FUNCTION_DECL:
            continue
        if not cursor.is_definition():
            continue
        symbol_name = cursor.displayname.split('(')[0]
        if symbol_name in valid_symbols:
            comments = retrieve_comments(cursor)
            functions[cursor.displayname] = comments
    return functions

def retrieve_comments(cursor: clang.cindex.Cursor) -> list[str]:
    comments = []
    for token_raw in cursor.get_tokens():
        token: str = token_raw.spelling
        if token.startswith("//") or token.startswith("/*"):
            comments.append(token)
    return comments

def get_all_function_comments(src_files: list[str], symbols: list[str]) -> dict[str,list[str]]:
    comments = {}
    for src_file in src_files:
        comments = dict(comments, **get_function_comment(src_file, symbols))
    return comments

def get_all_function_names(src_files: list[str], symbols: list[str]) -> list[str]:
    functions = []
    for src_file in src_files:
        functions += get_function_names(src_file, symbols)
    return list(set(functions))

def get_function_definition(location: SourceRange) -> str:
    filename = location.start.file.name
    with open(filename, 'r') as fh:
        contents = fh.read()
    return contents[location.start.offset: location.end.offset]

def get_all_function_definitions(src_files: list[str], symbols: list[str]) -> dict[str,str]:
    functions = {}
    for src_file in src_files:
        functions = dict(
            functions,
            **get_function_definitions(src_file, symbols)
        )
    return functions
                                               

def get_n_function_definitions(src_files: list[str], n: int, symbols: list[str]) -> dict[str,str]:
    functions = {}
    src_files_index = 0
    while len(functions) < n:
        functions = dict(
            functions,
            **get_function_definitions(src_files[src_files_index], symbols)
        )
        src_files_index += 1
    return dict(take(n,functions.items()))

def take(n, iterable):
    return list(islice(iterable, n))

if __name__ == '__main__':
    main()
