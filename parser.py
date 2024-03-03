import os
import json
import fire
from tree_sitter import Parser, Language, Node

C_LANGUAGE = Language("grammar/c-grammar.so", "c")
SYMBOL_PATH = "data/libm-libc-symbols.json"

def main():
    fire.Fire({
        "definition" : dump_function_definitions,
        "name": dump_function_names,
        "comment": dump_function_comments
    })
    
def dump_function_definitions(glibc_path: str) -> None:
    print("Generating function-defintion-table.json...")
    src_files = get_src_files(glibc_path)
    symbols   = get_symbols(SYMBOL_PATH)
    functions = get_all_function_definitions(src_files, symbols)
    with open('function-defintion-table.json', 'w') as f:
        json.dump(functions, f)

def dump_function_names(glibc_path: str) -> None:
    print("Generating function-name-table.json...")
    src_files = get_src_files(glibc_path)
    symbols   = get_symbols(SYMBOL_PATH)
    functions = get_all_function_names(src_files, symbols)
    with open('function-name-table.json', 'w') as f:
        json.dump({"functions": functions}, f)

def dump_function_comments(glibc_path: str, deduction: bool) -> None:
    src_files = get_src_files(glibc_path)
    symbols   = get_symbols(SYMBOL_PATH)
    dump_name = ""
    if deduction:
        function_comments = get_all_function_comments_deduction(src_files, symbols)
        dump_name = "function-comment-deduction-table.json"
    else:
        function_comments = get_all_function_comments(src_files, symbols)
        dump_name = "function-comment-table.json"
    print(f"Generating {dump_name}...")
    with open(dump_name, 'w') as f:
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

def get_all_function_comments(src_files: list[str], symbols: list[str]) -> dict[str,list[str]]:
    comments = {}
    parser = Parser()
    parser.set_language(C_LANGUAGE)
    for src_file in src_files:
        try:
            root_node = parse_src_file(parser, src_file)
            comments = dict(
                comments,
                **get_function_comments(root_node)
            )
        except RuntimeError as msg:
            print(msg)
    return {key: value for key, value in comments.items() if key in symbols}

def get_all_function_comments_deduction(src_files: list[str], symbols: list[str]) -> dict[str,list[str]]:
    comments = {}
    parser = Parser()
    parser.set_language(C_LANGUAGE)
    for src_file in src_files:
        try: 
            root_node = parse_src_file(parser, src_file)
            comments = dict(
                comments,
                **get_function_comments_deduction(root_node, root_node)
            )
        except RuntimeError as msg:
            print(msg)
    return {key: value for key, value in comments.items() if key in symbols}

def get_all_function_names(src_files: list[str], symbols: list[str]) -> list[str]:
    functions = []
    parser = Parser()
    parser.set_language(C_LANGUAGE)
    for src_file in src_files:
        root_node = parse_src_file(parser, src_file)
        name = get_function_definition_names(root_node)
        functions += list(set(name).intersection(set(symbols)))
    return list(set(functions))

def get_all_function_definitions(src_files: list[str], symbols: list[str]) -> dict[str,str]:
    functions = {}
    parser = Parser()
    parser.set_language(C_LANGUAGE)
    for src_file in src_files:
        root_node = parse_src_file(parser, src_file)
        functions = dict(
            functions,
            **get_function_definitions(root_node)
        )
    return {key: value for key, value in functions.items() if key in symbols}


def get_function_definition_names(node: Node) -> list[str]:
    names: list[str] = []
    current_name: str = ""

    if node.type == "function_definition":
        try:
            current_name = get_name_from_definition(node)
            names.append(current_name)
        except RuntimeError as msg:
            print(msg)

    for child in node.named_children:
        names += get_function_definition_names(child)

    return names

def get_function_definitions(node: Node) -> dict[str,str]:
    definitions: dict[str,str] = {}
    current_definition: str = ""
    current_name: str = ""

    if node.type == "function_definition":
        try: 
            current_definition = get_definition_text_from_definition(node)
            current_name = get_name_from_definition(node)
            definitions[current_name] = current_definition
        except RuntimeError as msg:
            print(msg)

    for child in node.named_children:
        definitions = dict(definitions, **get_function_definitions(child))

    return definitions

def get_function_comments_deduction(node: Node, root: Node) -> dict[str,str]:
    comments: dict[str,str] = {}
    current_function_name: str = ""
    current_comment: str = ""

    if node.type == "function_definition":
        current_function_name = get_name_from_definition(node)
        comment_node = node.prev_named_sibling

        if comment_node is not None and comment_node.type == "comment":
            current_comment = comment_node.text.decode("utf-8")
        elif comment_node is not None and marco_blocking_comment(comment_node) and node_contains(comment_node, "comment"):
            current_comment = get_contained_text(comment_node, "comment")
        else:
            current_comment = get_comment_from_called_functions(node, root)
        
        comments[current_function_name] = strip_comment(current_comment)
        
    for child in node.named_children:
        comments = dict(
            comments, 
            **get_function_comments_deduction(child, root)
        )

    return comments

def get_function_comments(node: Node) -> dict[str,str]:
    comments: dict[str,str] = {}
    current_function_name: str = ""
    current_comment: str = ""

    if node.type == "function_definition":
        current_function_name = get_name_from_definition(node)
        comment_node = node.prev_named_sibling
        current_comment = ""
        
        if comment_node is not None:
            if comment_node.type == "comment":
                current_comment = comment_node.text.decode("utf-8")
            elif marco_blocking_comment(comment_node) and node_contains(comment_node, "comment"):
                current_comment = get_contained_text(comment_node, "comment")

        comments[current_function_name] = strip_comment(current_comment)

    for child in node.named_children:
        comments = dict(comments, **get_function_comments(child))

    return comments

def get_name_from_definition(node: Node) -> str:
    for child in node.named_children:
        body_node = child.next_named_sibling
        if body_node is not None and body_node.type == "compound_statement":
            return function_node_to_name(child)
    raise RuntimeError("No name in defintion found!")

def get_definition_text_from_definition(node: Node) -> str:
    for child in node.named_children:
        body_node = child.next_named_sibling
        if body_node is not None and body_node.type == "compound_statement":
            return (child.text + body_node.text).decode("utf-8")
    raise RuntimeError("No defintion text in defintion found!")

def get_comment_from_called_functions(node: Node, root_node: Node) -> str:
    comments : str = ""
    functions_called: list[str] = get_function_called_names(node)
    functions_comments = get_function_comments(root_node)
    for function in functions_called:
        comments += str(functions_comments.get(function) or "")
    return comments

def get_function_called_names(node: Node) -> list[str]:
    names: list[str] = []
    current_name: str = ""
    if node.type == "call_expression":
        current_name = function_node_to_name(node)
        names.append(current_name)
    for child in node.named_children:
        names += get_function_called_names(child)
    return names

def function_node_to_name(node: Node) -> str:
    name = node.text.decode("utf-8").split("(")[0].strip()
    if "\n" not in name:
        return name
    return name.split("\n").pop()

def node_contains(node: Node, node_type: str) -> bool:
    contains: bool = False
    if node.type == node_type:
        return True
    for child in node.named_children:
        if contains:
            break
        contains = contains or node_contains(child, node_type)
    return contains

def get_contained_text(node: Node, node_type: str) -> str:
    txt: str = ""
    if node.type == node_type:
        return node.text.decode("utf-8")
    for child in node.named_children:
        if txt != "":
            break
        txt += get_contained_text(child, node_type)
    return txt

def strip_comment(comment: str) -> str:
    return comment.strip().strip("/*").strip()

def marco_blocking_comment(node: Node) -> bool:
    return node.has_error and node.type != "function_definition"

def get_root_node_from_path(path: str) -> Node:
    parser = Parser()
    parser.set_language(C_LANGUAGE)
    src = str()
    with open(path) as f:
        src = f.read()
    tree = parser.parse(src.encode("utf-8"))
    return tree.root_node

def parse_src_file(parser: Parser, src_file: str) -> Node:
    with open(src_file) as f:
        try:
            src = f.read()
        except UnicodeDecodeError:
            src = ""
            print(f"Invalid chars in {src_file}")
    tree = parser.parse(src.encode("utf-8"))
    return tree.root_node

if __name__ == '__main__':
    main()
