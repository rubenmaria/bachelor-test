import json
from c_parser import get_definition_at
from typing import Optional
import ollama
import os
#from llama import Llama

CHECKPOINT_DIR: str = "/mnt/ambrym1/llama-model/llama/llama-2-13b-chat/"
TOKENIZER_PATH: str = "/mnt/ambrym1/llama-model/llama/tokenizer.model"
MAX_SEQUENCE_LENGTH: int = 8192 * 2
MAX_BATCH_SIZE: int = 6
TEMPERATURE: float = 0.6
TOP_P: float = 0.9
MAX_GEN_LENGTH: Optional[int] = None

PROMPT_PATH = "data/prompt.txt"


def main():
    generate_summaries(
        "data/function-defintions.json",
        "data/prompt.txt",
        "function-definition-explanation.json"
    )


def prompt_llm(llm_name: str, prompt: str) -> str:
    result = ollama.generate(model=llm_name, prompt=prompt)
    assert type(result) == dict
    return result["response"]

def get_summary(
    definition: str,
    model_name: str,
    prompt_path: str
) -> str:
    llama_prompt = load_prompt(prompt_path)
    try:
        return prompt_llm(model_name, llama_prompt + "\n" + definition)
    except Exception as err:
        raise RuntimeError(err)

def get_summary_from_file(
    path: str,
    name: str,
    model_name: str,
    prompt_path: str
) -> str:
    definition = get_definition_at(path, name)
    llama_prompt = load_prompt(prompt_path)
    try:
        return prompt_llm(model_name, llama_prompt + definition)
    except Exception as err:
        raise RuntimeError(err)


def get_summaries_from_file(
    functions_path: str,
    prompt_path: str,
    model_name: str
    ) -> dict[str,str]:
    definitions = load_function_definitions(functions_path)
    llama_prompt = load_prompt(prompt_path)
    llama_out = {}
    for definition_name in definitions.keys():
        try:
            response = prompt_llm(
                model_name,
                llama_prompt + definitions[definition_name]
            )
            print(response)
        except Exception as err:
            response = ""
            print(f"Error at {definition_name}: {err}")
        llama_out[definition_name] = response
    return llama_out


def get_summaries(
    functions: dict[str,str],
    prompt_path: str,
    model_name: str
    ) -> dict[str,str]:
    llama_prompt = load_prompt(prompt_path)
    llama_out = {}
    for definition_name in functions.keys():
        try:
            response = prompt_llm(
                model_name,
                llama_prompt + functions[definition_name]
            )
            print(response)
        except Exception as err:
            response = ""
            print(f"Error at {definition_name}: {err}")
        llama_out[definition_name] = response
    return llama_out


def generate_summaries_from_file(
    functions_path: str,
    prompt_path: str,
    output_directory: str,
    output_name: str,
    model_name: str
    ) -> None:
    llama_out = get_summaries_from_file(
        functions_path,
        prompt_path,
        model_name
    )
    output_path = os.path.join(output_directory, output_name) + ".json"
    with open(output_path, "w") as f:
        json.dump(llama_out, f)


def get_summary_at(path: str, name: str, prompt_path: str) -> str:
    llama_generator = build_llama_generator()
    definition = get_definition_at(path, name)
    llama_prompt = load_prompt(prompt_path)
    try:
        return prompt_llama(llama_generator, llama_prompt + definition)
    except Exception as err:
        raise RuntimeError(err)


def generate_summaries(
    def_path: str,
    prompt_path: str,
    output_path: str
) -> None:
    llama_generator = build_llama_generator()
    definitions = load_function_definitions(def_path)
    llama_prompt = load_prompt(prompt_path)
    llama_out = {}

    for definition_name in definitions.keys():
        try:
            response = prompt_llama(
                llama_generator,
                llama_prompt + definitions[definition_name]
            )
            print(response)
        except Exception as err:
            response = ""
            print(f"Error at {definition_name}: {err}")
        llama_out[definition_name] = response

    with open(output_path, "w") as f:
        json.dump(llama_out, f)


def load_function_definitions(path: str) -> dict[str, str]:
    with open(path) as f:
        return json.load(f)


def load_prompt(path: str) -> str:
    with open(path) as f:
        return f.readline().strip("\n")


def build_llama_generator():
    generator = Llama.build(
        ckpt_dir=CHECKPOINT_DIR,
        tokenizer_path=TOKENIZER_PATH,
        max_seq_len=MAX_SEQUENCE_LENGTH,
        max_batch_size=MAX_BATCH_SIZE,
    )
    return generator


def prompt_llama(generator, msg: str) -> str:

    dialogs = [[
        {"role": "system", "content": "Just give the summary nothing else!"},
        {"role": "user", "content": msg}
        ]]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=MAX_GEN_LENGTH,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    anwser: str = str()
    for _, result in zip(dialogs, results):
        anwser += f"{result['generation']['content']}".strip() + "\n"
    return anwser


if __name__ == "__main__":
    main()
