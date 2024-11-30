import subprocess
import json
import os
import angr

def generate_valid_symbols(binary_dir: str, output_path: str) -> None:
    print(f"Generating valid symbol table: '{output_path}'...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(
            get_functions_from_dir(binary_dir),
            f,
            ensure_ascii=False,
            indent=4
        )

def get_functions_from_binary(binary_path: str) -> dict[str,str]:
    project = angr.Project(binary_path, load_options={'auto_load_libs': False})
    cfg = project.analyses.CFGFast()
    function_table: dict[str,str] = {}
    for function in cfg.kb.functions.values():
        if function.name.startswith("sub_"):
            continue
        function_id = function.project.filename + "/" + str(function.addr)
        function_table[function_id] = function.name.split("@")[0]
    return function_table


def get_functions_from_dir(binary_dir: str) -> dict[str,str]:
    binary_directory = os.fsencode(binary_dir)
    functions: dict[str, str] = {}
    for file in os.listdir(binary_directory):
        filename = os.fsdecode(file)
        directoy = os.fsdecode(binary_directory)
        if filename.endswith(".so"):
            binary_path = os.path.join(directoy, filename)
            functions = dict(functions, **get_functions_from_binary(binary_path))
            functions.update()
            print(os.path.join(directoy, filename))
    return functions


def parse_symbols(so_str: str) -> list[str]:
    lines = list(filter(lambda x: x != "", so_str.splitlines()))
    symbols = []
    for line in lines:
        tokens = line.split()
        if len(tokens) < 7:
            print(tokens)
            continue
        symbols.append(line.split()[6])
    return symbols


def get_objdump_ouput_from_dir(binary_dir: str) -> str:
    objdump_output = ""
    binary_directory = os.fsencode(binary_dir)
    for file in os.listdir(binary_directory):
        filename = os.fsdecode(file)
        directoy = os.fsdecode(binary_directory)
        if filename.endswith(".so"):
            objdump_output += subprocess.run(
                ['objdump', '-T', os.path.join(directoy, filename)],
                stdout=subprocess.PIPE
            ).stdout.decode('utf-8') + "\n"
            print(os.path.join(directoy, filename))
    return objdump_output
