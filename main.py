import fire

from binary import generate_valid_symbols, get_functions_from_binary
from c_parser import generate_function_names, generate_function_comments, generate_function_definitions
#from llama_prompt import generate_summaries
from embeddings import calculate_standard_deviation_from_embeddings, generate_embeddings_TSNE, \
    calculate_standard_deviation_sentence_transfomer, generate_high_dimensional, \
    calculate_standard_deviation_from_embeddings
from code2vec_api import generate_vectors
from visual import plot_clusters_from_path


def main():
    fire.Fire({
        "symbols" : generate_valid_symbols,
        "name": generate_function_names,
        "comment": generate_function_comments,
        "definition": generate_function_definitions,
        #"summaries": generate_summaries,
        "plot-clusters": plot_clusters_from_path,
        "generate_c2vec": generate_vectors,
        "embeddings": generate_high_dimensional,
        "embeddings-low": generate_embeddings_TSNE,
        "deviation-st": calculate_standard_deviation_sentence_transfomer,
        "deviation-embeddings": calculate_standard_deviation_from_embeddings
    })


if __name__ == '__main__':
    main()
