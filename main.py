import fire

from binary import generate_valid_symbols, get_functions_from_binary
from c_parser import generate_function_names, generate_function_comments, generate_function_definitions
#from llama_prompt import generate_summaries
from embeddings import (calculate_standard_deviation_from_embeddings, calculate_standard_deviation_llm, generate_embeddings_TSNE, 
    calculate_standard_deviation_sentence_transfomer, generate_high_dimensional, 
    calculate_standard_deviation_from_embeddings, calculate_standard_deviation_llm, 
    generate_llm_TSNE, compare_embeddings_simple, calculate_max_deviation_sentence_transfomer)
from survey import generate_survey_csv, generate_surveys_csv, evaluate_survey_results
from code2vec_api import generate_vectors
from visual import (plot_clusters_from_path, plot_compare_random,
    plot_compare_from_file)
from labels import generate_labels, FuncData, LabelData


def main():
    fire.Fire({
        "symbols": generate_valid_symbols,
        "name": generate_function_names,
        "comment": generate_function_comments,
        "definition": generate_function_definitions,
        "plot-clusters": plot_clusters_from_path,
        "generate_c2vec": generate_vectors,
        "embeddings": generate_high_dimensional,
        "embeddings-low": generate_embeddings_TSNE,
        "deviation-st": calculate_standard_deviation_sentence_transfomer,
        "deviation-llm": calculate_standard_deviation_llm,
        "generate-labels": generate_labels,
        "deviation-embeddings": calculate_standard_deviation_from_embeddings,
        "generate-llm-tsne": generate_llm_TSNE,
        "simple-compare":  compare_embeddings_simple,
        "plot-compare-file": plot_compare_from_file,
        "plot-compare-random": plot_compare_random,
        "generate-survey": generate_survey_csv,
        "generate-surveys": generate_surveys_csv,
        "max-deviation-st": calculate_max_deviation_sentence_transfomer,
        "eval-survey": evaluate_survey_results
    })


if __name__ == '__main__':
    main()
