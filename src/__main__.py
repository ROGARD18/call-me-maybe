from llm_sdk import Small_LLM_Model
from typing import Any
from src.vocab_dict import make_vocab_dict
from src.prompt import make_prompt
import numpy as np


def check_token(token: str) -> str:
    token = token.replace('Ġ', ' ')
    token = token.replace('ĉ', '\t')
    token = token.replace('Ċ', '\n')
    return (token)

def main() -> None:
    text: str = "How much do 40 + 2"

    prompt: str = make_prompt(text)
    generate_text: str = ""
    key_temp: str = ""

    llm: Small_LLM_Model = Small_LLM_Model()
    vocab: dict = make_vocab_dict(llm.get_path_to_vocab_file())

    torch_tensor = llm.encode(prompt)
    ids: Any = torch_tensor[0].tolist()
    log_list: list[float] = llm.get_logits_from_input_ids(ids)

    print(f"- prompt: {prompt}")
    print(f"- torch tensor: {torch_tensor}")
    print(f"- IDs: {ids}")

    index_max: int = np.argmax(log_list)
    print("- index max =", index_max)
    print(f"- log max: {log_list[index_max]}")
    for key, value in vocab.items():
        if int(value) == index_max:
            next_token = key
    print(f"- next token = -{key_temp}-")
    print(f"- decode IDs: {llm.decode(ids)}")

    

    for _ in range(200):
        log_list: list[float] = llm.get_logits_from_input_ids(ids)
        index_max = np.argmax(log_list)
        for key, value in vocab.items():
            if int(value) == index_max:
                next_token = key
        next_token = check_token(next_token)
        prompt += next_token
        generate_text += next_token
        if "JSON VALID" in generate_text:
            break
        torch_tensor = llm.encode(prompt)
        ids: Any = torch_tensor[0].tolist()

    print("\n\nNEW prompt GENERATE:\n")
    print(prompt)


if __name__ == "__main__":
    main()
    print("FINSH !!")
