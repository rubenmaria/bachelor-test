from tree_sitter import Language
from tree_sitter import Parser
from tree_sitter import Node
from parser import get_root_node_from_path, get_function_comments
from parser import get_function_comments_deduction, get_function_definition_names
from parser import get_function_definitions

C_LANGUAGE = Language("grammar/c-grammar.so", "c")


def get_root_node_from_path(path: str) -> Node:
    parser = Parser()
    parser.set_language(C_LANGUAGE)
    src = str()
    with open(path) as f:
        src = f.read()
    tree = parser.parse(src.encode("utf-8"))
    return tree.root_node


def debug_print(path: str) -> None:
    root_node = get_root_node_from_path(path)
    print("="*10, " comments + deduction ", "="*10)
    print(get_function_comments_deduction(root_node, root_node))
    print("="*30)
    print("="*10, " comments ", "="*10)
    print(get_function_comments(root_node))
    print("="*30)
    print("="*10, " names ", "="*10)
    print(get_function_definition_names(root_node))
    print("="*30)
    print("="*10, " defintions ", "="*10)
    print(get_function_definitions(root_node))
    print("="*30)

debug_print("glibc/io/read.c")
debug_print("glibc/stdlib/random.c")
debug_print("glibc/stdlib/exit.c")
debug_print("glibc/stdlib/qsort.c")
