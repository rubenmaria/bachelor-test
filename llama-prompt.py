import json
import time
from typing import Optional
from llama import Llama

CHECKPOINT_DIR: str = "/mnt/ambrym1/llama-model/llama/llama-2-7b-chat/"
TOKENIZER_PATH: str = "/mnt/ambrym1/llama-model/llama/tokenizer.model"
MAX_SEQUENCE_LENGTH: int = 2048
MAX_BATCH_SIZE: int  = 6
TEMPERATURE: float  = 0.6
TOP_P: float = 0.9
MAX_GEN_LENGTH: Optional[int] = None

def main():
    with open("data/function-definition-table.json") as file:
        definitions = json.load(file)
        llama_prompt = "Kannst du mir die Funktion erklären?\n"
        llama_generator = build_llama_generator()
        for definition_name in definitions.keys():
            print(definition_name + ": " +  prompt_llama(
                llama_generator,
                llama_prompt + definitions[definition_name]
            ))

def dump_llama_function_explanation() -> None:
    with open("data/function-definition-table.json") as file:
        definitions = json.load(file)
        llama_out = {}
        llama_prompt = "Kannst du mir die Funktion erklären?\n"
        for definition_name in definitions.keys():
            llama_out[definition_name] = prompt_llama(
                llama_prompt + definitions[definition_name]
            )
        with open("function-definition-explanation.json") as f:
            json.dump(llama_out, f)


def build_llama_generator():
    generator = Llama.build(
        ckpt_dir=CHECKPOINT_DIR,
        tokenizer_path=TOKENIZER_PATH,
        max_seq_len=MAX_SEQUENCE_LENGTH,
        max_batch_size=MAX_BATCH_SIZE,
    )
    return generator

def prompt_llama(generator, msg: str) -> str:

    dialogs = [[{"role": "user", "content": msg}]]
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
