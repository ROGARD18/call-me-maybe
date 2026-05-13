from llm_sdk import Small_LLM_Model
from typing import Any
from src.vocab_dict import make_vocab_dict
from src.prompt import make_prompt
import numpy as np
from src.next_token import GrammarConstrainedSampler, step_json
from src.class_ import JSONState, FunctionsClass
from src.validate_json import validate_json as valid_js
from src.get_function import get_functions


def format_text(text: str) -> str:
    text = text.replace("Ġ", " ")
    text = text.replace("Ċ", "\n")
    text = text.replace("ĉ", "\t")
    return text


def new_step(
    text: str, token: str, step: int, js: JSONState, Functions: FunctionsClass
) -> int:
    if step == 0:
        js.JSON_START = js.JSON_START[len(token) :]
        if not (js.JSON_START):
            return 1
    elif step == 1:
        js.KEY_PROMPT = js.KEY_PROMPT[len(token) :]
        if not (js.KEY_PROMPT):
            return 2
    elif step == 2:
        js.PROMPT_VALUE = js.PROMPT_VALUE[len(token) :]
        if not (js.PROMPT_VALUE):
            return 3
    elif step == 4:
        js.KEY_NAME = js.KEY_NAME[len(token) :]
        if not (js.KEY_NAME):
            return 5
    elif step == 5:
        clean_token = token.replace("Ġ", "")
        Functions.list = [f for f in Functions.list if f.startswith(clean_token)]
        for i in range(len(Functions.list)):
            Functions.list[i] = Functions.list[i][len(clean_token) :]
        if any(len(f) == 0 for f in Functions.list):
            return 6
    elif step == 7:
        js.KEY_PARA = js.KEY_PARA[len(token) :]
        if not (js.KEY_PARA):
            return 8
    elif step == 8:
        if "}" in token:
            return 9
    elif step == 9:
        js.JSON_END = js.JSON_END[len(token) :]
        if not (js.JSON_END):
            return 10
    elif step == 3 or step == 6:
        js.LINE_END = js.LINE_END[len(token) :]
        if not (js.LINE_END):
            js.LINE_END = ",Ċĉĉ"
            return step + 1
    return step


def check_token(token: str) -> str:
    token = token.replace("Ġ", " ")
    token = token.replace("ĉ", "\t")
    token = token.replace("Ċ", "\n")
    return token


def main() -> None:
    text: str = "what is the reverse of the string 'Hello'"
    names, defs = get_functions()
    Functions: FunctionsClass = FunctionsClass(names, defs)

    js: JSONState = JSONState(text, len(max(Functions.list, key=len)))
    prompt: str = make_prompt(text, Functions.list)
    generate_text: str = ""

    llm: Small_LLM_Model = Small_LLM_Model()
    vocab: dict = make_vocab_dict(llm.get_path_to_vocab_file())

    cons_sampler = GrammarConstrainedSampler(grammar_valid_fn=step_json)
    step: int = 0
    while True:
        if step == 10:
            break
        torch_tensor = llm.encode(prompt)
        ids: Any = torch_tensor[0].tolist()
        logits: list[float] = llm.get_logits_from_input_ids(ids)
        logits = np.array(logits)
        index_max = cons_sampler.constrained_sample(
            js, logits, text, prompt, Functions, step, vocab
        )
        for key, value in vocab.items():
            if int(value) == index_max:
                next_token = key
        print("token ->", next_token)
        generate_text += next_token
        prompt += next_token
        next_step: int = new_step(text, next_token, step, js, Functions)
        if next_step == 6 and step == 5:
            js.FUNCTION = (
                format_text(generate_text)
                .split('"name": "')[-1]
                .replace('"', "")
                .strip()
            )

            generate_text += '"'
        elif next_step == 9 and step == 8:
            generate_text += ","
        step = next_step

    print("\n\nNEW prompt GENERATE:\n")
    generate_text = format_text(generate_text)
    print(generate_text)

    print(valid_js(generate_text))


if __name__ == "__main__":
    main()
    print("FINSH !!")
