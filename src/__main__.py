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
    print(f"step = {step}")
    if step == 0:
        js.JSON_START = js.JSON_START[len(token):]
        return 1 if not js.JSON_START else 0
    elif step == 1:
        js.KEY_PROMPT = js.KEY_PROMPT[len(token):]
        return 2 if not js.KEY_PROMPT else 1
    elif step == 2:
        js.PROMPT_VALUE = js.PROMPT_VALUE[len(token):]
        return 3 if not js.PROMPT_VALUE else 2
    elif step == 4:
        js.KEY_NAME = js.KEY_NAME[len(token):]
        return 5 if not js.KEY_NAME else 4
    elif step == 5:
        clean_token = token.replace("Ġ", "")
        Functions.list = [f for f in Functions.list
                          if f.startswith(clean_token)]
        for i in range(len(Functions.list)):
            Functions.list[i] = Functions.list[i][len(clean_token):]
        return 6 if any(len(f) == 0 for f in Functions.list) else 5
    elif step == 7:
        js.KEY_PARA = js.KEY_PARA[len(token):]
        return 8 if not js.KEY_PARA else 7
    elif step == 8:
        print(f"sub_STEP = {js.sub_step}")
        if js.sub_step == 0 and "{" in token:
            js.sub_step = 1
        elif js.sub_step == 1 and '"' in token:
            js.sub_step = 2
        elif js.sub_step == 2:
            js.sub_step = 3
        elif js.sub_step == 3 and '"' in token:
            js.sub_step = 4
        elif js.sub_step == 4 and ":" in token:
            js.sub_step = 5
        elif js.sub_step == 5 and "Ġ" in token:
            js.sub_step = 6
        elif js.sub_step == 6:
            if "," in token:
                js.param_order += 1
                if not js.param_order == len(js.TYPES):
                    js.sub_step = 1
            elif "}" in token:
                return 9
        return 8
    elif step == 9:
        js.JSON_END = js.JSON_END[len(token):]
        return 10 if not js.JSON_END else 9
    elif step in [3, 6]:
        js.LINE_END = js.LINE_END[len(token):]
        if not js.LINE_END:
            js.LINE_END = ",Ċĉĉ"
            return step + 1
        return step
    return step


def check_token(token: str) -> str:
    token = token.replace("Ġ", " ")
    token = token.replace("ĉ", "\t")
    token = token.replace("Ċ", "\n")
    return token


def main() -> None:
    text: str = "What is the sum of 15.63 and 78.1?"
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
            js.sub_step = 0
            js.param_order = 0
            js.FUNCTION = (
                format_text(generate_text)
                .split('"name": "')[-1]
                .replace('"', "")
                .strip()
            )
            parameters: dict = Functions.definitions.get(js.FUNCTION)
            print("parameters ==", parameters)
            types: list = []
            for param in parameters:
                print(f"test ____ = {parameters.get(param).get("type")}")
                types.append(parameters.get(param).get("type"))
            js.TYPES = types
            print("---------------->>>>>>>", js.TYPES)
            generate_text += '"'
        step = next_step

    print("\n\nNEW prompt GENERATE:\n")
    generate_text = format_text(generate_text)
    print(generate_text)

    print(valid_js(generate_text))


if __name__ == "__main__":
    main()
    print("FINSH !!")
