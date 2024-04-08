import fire

from binary import generate_valid_symbols, get_functions_from_binary
from c_parser import generate_function_names, generate_function_comments, generate_function_definitions
from llama_prompt import generate_summaries
from embeddings import generate_embeddings_TSNE


def main():
    fire.Fire({
        "symbols" : generate_valid_symbols,
        "name": generate_function_names,
        "comment": generate_function_comments,
        "definition": generate_function_definitions,
        "summaries": generate_summaries,
        "embeddings": generate_embeddings_TSNE
    })


if __name__ == '__main__':
    main()
