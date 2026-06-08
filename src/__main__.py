import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict
import numpy as np

from llm_sdk import Small_LLM_Model     # type: ignore
from src.models import FunctionsClass
from src.vocab_dict import make_vocab_dict


def generate_constrained_call(
    prompt_text: str,
    functions_class: FunctionsClass,
    llm: Small_LLM_Model,
    inverse_vocab: Dict[int, str],
) -> Dict[str, Any]:

    schema_lines = []
    for name, f in functions_class.definitions.items():
        desc = f.get("description", "")
        params = f.get("parameters", {})
        param_str = ", ".join(
            f'"{k}": {v.get("type", "unknown")}' for k, v in params.items()
        )
        schema_lines.append(f"- {name}: {desc}\n  Parameters: {{{param_str}}}")

    context = (
        "You are an exact parameter extractor. Select the correct "
        "function and extract the literal arguments.\n"
        "Copy values exactly as they appear in the request, preserving "
        "full file paths, URLs, and special characters.\n"
        f"Available functions:\n{chr(10).join(schema_lines)}\n\n"
    )

    examples = [
        {
            "prompt": "Read the file at /home/user/data.json with "
            "utf-8 encoding",
            "name": "fn_read_file",
            "parameters": {"path": "/home/user/data.json",
                           "encoding": "utf-8"},
        },
        {
            "prompt": 'Format template: Say "hello" to {name}',
            "name": "fn_format_template",
            "parameters": {"template": 'Say "hello" to {name}'},
        },
        {
            "prompt": "Calculate compound interest on 1234567.89 at "
            "0.0375 rate for 23 years",
            "name": "fn_calculate_compound_interest",
            "parameters": {"principal": 1234567.89, "rate": 0.0375,
                           "years": 23},
        },
    ]

    for ex in examples:
        ex_str = json.dumps(ex, indent=2)
        context += f"Request: {ex['prompt']}\n{ex_str}\n\n"

    safe_prompt = (
        prompt_text.replace("\\", "\\\\").replace('"', '\\"'
                                                  ).replace("\n", "\\n")
    )
    context += f"Request: {prompt_text}\n"

    prefix = f'{{\n  "prompt": "{safe_prompt}",\n  "name": "'

    encoded = llm.encode(context + prefix).tolist()
    tokens = encoded[0] if isinstance(encoded[0], list) else encoded

    print(prefix, end="", flush=True)

    selected_name = ""
    current_name_str = ""

    for _ in range(50):
        logits = np.array(llm.get_logits_from_input_ids(tokens),
                          dtype=np.float32)
        is_allowed = np.zeros(len(logits), dtype=bool)

        for tid, tstr in inverse_vocab.items():
            clean = tstr.replace("Ġ", "").replace("Ċ", "").replace("ĉ", "")
            if not clean:
                continue

            if (current_name_str in functions_class.list and
                    clean.startswith('"')):
                is_allowed[tid] = True

            potential = current_name_str + clean
            if any(fn.startswith(potential) for fn in functions_class.list):
                is_allowed[tid] = True

        if not np.any(is_allowed):
            is_allowed.fill(True)

        masked_logits = np.where(is_allowed, logits, -1e10)
        next_token = int(np.argmax(masked_logits))
        tokens.append(next_token)

        clean_token = (
            inverse_vocab.get(next_token, "")
            .replace("Ġ", "")
            .replace("Ċ", "")
            .replace("ĉ", "")
        )

        if '"' in clean_token:
            idx = clean_token.find('"')
            current_name_str += clean_token[:idx]
            print(clean_token[:idx] + '"', end="", flush=True)
            selected_name = current_name_str
            break
        else:
            current_name_str += clean_token
            print(clean_token, end="", flush=True)

    if selected_name not in functions_class.definitions:
        selected_name = (functions_class.list[0] if functions_class.list
                         else "unknown")

    expected_params = functions_class.definitions.get(selected_name, {}).get(
        "parameters", {}
    )
    params_obj = {}

    trans_str = ',\n  "parameters": {'
    if expected_params:
        trans_str += "\n"

    encoded_trans = llm.encode(trans_str).tolist()
    tokens.extend(
        encoded_trans[0] if isinstance(encoded_trans[0], list)
        else encoded_trans
    )
    print(trans_str, end="", flush=True)

    keys = list(expected_params.keys())
    for i, key in enumerate(keys):
        param_type = expected_params[key].get("type", "string")
        is_numeric = param_type in ("number", "integer")
        is_bool = param_type == "boolean"

        key_str = f'    "{key}": '
        if not is_numeric and not is_bool:
            key_str += '"'

        encoded_key = llm.encode(key_str).tolist()
        tokens.extend(
            encoded_key[0] if isinstance(encoded_key[0], list) else encoded_key
        )
        print(key_str, end="", flush=True)

        val_raw = ""

        for _ in range(150):
            logits = np.array(llm.get_logits_from_input_ids(tokens),
                              dtype=np.float32)
            is_allowed = np.zeros(len(logits), dtype=bool)

            if is_numeric:
                valid_chars = set("0123456789.-")
                for tid, tstr in inverse_vocab.items():
                    clean = tstr.replace("Ġ", " ").replace("Ċ", "\n")
                    clean = clean.replace("ĉ", "\t")
                    if any(c in clean for c in [",", "}", "\n", " "]):
                        is_allowed[tid] = True
                    elif all(c in valid_chars for c in clean):
                        is_allowed[tid] = True
            elif is_bool:
                valid_chars = set("truefalseTRUEFALSE")
                for tid, tstr in inverse_vocab.items():
                    clean = tstr.replace("Ġ", " ").replace("Ċ", "\n")
                    clean = clean.replace("ĉ", "\t")
                    if any(c in clean for c in [",", "}", "\n", " "]):
                        is_allowed[tid] = True
                    elif all(c in valid_chars for c in clean):
                        is_allowed[tid] = True
            else:
                for tid, tstr in inverse_vocab.items():
                    clean = tstr.replace("Ġ", " ").replace("Ċ", "\n")
                    clean = clean.replace("ĉ", "\t")
                    if "\n" not in clean or '"' in clean:
                        is_allowed[tid] = True

            if not np.any(is_allowed):
                is_allowed.fill(True)

            masked_logits = np.where(is_allowed, logits, -1e10)
            next_token = int(np.argmax(masked_logits))
            tokens.append(next_token)

            raw = inverse_vocab.get(next_token, "")
            clean_token = (
                raw.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")
            )

            if is_numeric or is_bool:
                if any(c in clean_token for c in [",", "}", "\n", " "]):
                    idx_list = [
                        clean_token.find(c)
                        for c in [",", "}", "\n", " "]
                        if clean_token.find(c) != -1
                    ]
                    idx = min(idx_list) if idx_list else len(clean_token)
                    val_raw += clean_token[:idx]
                    print(clean_token[:idx], end="", flush=True)
                    break
                else:
                    val_raw += clean_token
                    print(clean_token, end="", flush=True)
            else:
                val_raw += clean_token

                escape = False
                idx = -1
                for j, c in enumerate(val_raw):
                    if c == "\\" and not escape:
                        escape = True
                    elif c == '"' and not escape:
                        idx = j
                        break
                    else:
                        escape = False

                if idx != -1:
                    prev_len = len(val_raw) - len(clean_token)
                    idx_in_clean = idx - prev_len
                    if idx_in_clean >= 0:
                        print(clean_token[: idx_in_clean + 1],
                              end="", flush=True)

                    val_str_json = '"' + val_raw[: idx + 1]
                    try:
                        params_obj[key] = json.loads(val_str_json)
                    except json.JSONDecodeError:
                        params_obj[key] = val_raw[:idx]
                    break
                else:
                    print(clean_token, end="", flush=True)

        if is_numeric:
            val_clean = val_raw.strip()
            try:
                params_obj[key] = (
                    float(val_clean) if "." in val_clean else int(val_clean)
                )
            except ValueError:
                params_obj[key] = 0
        elif is_bool:
            params_obj[key] = val_raw.strip().lower() == "true"

        if i < len(keys) - 1:
            sep_str = ",\n"
            encoded_sep = llm.encode(sep_str).tolist()
            tokens.extend(
                encoded_sep[0] if isinstance(encoded_sep[0],
                                             list) else encoded_sep
            )
            print(sep_str, end="", flush=True)
        else:
            sep_str = "\n  }"
            print(sep_str, end="", flush=True)

    print("\n}", flush=True)

    return {"prompt": prompt_text, "name": selected_name,
            "parameters": params_obj}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input", type=str, default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output", type=str,
        default="data/output/function_calling_results.json"
    )
    args = parser.parse_args()

    with open(args.functions_definition, "r", encoding="utf-8") as f:
        f_data = json.load(f)
    functions = FunctionsClass(
        list=[f["name"] for f in f_data], definitions={
            f["name"]: f for f in f_data}
    )

    with open(args.input, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    llm = Small_LLM_Model()
    vocab = make_vocab_dict(llm.get_path_to_vocabulary_json())
    inverse_vocab = {int(v): k for k, v in vocab.items()}

    results = []
    for i, item in enumerate(prompts):
        print(f"\n--- Processing {i + 1}/{len(prompts)} ---")
        try:
            res = generate_constrained_call(
                item["prompt"], functions, llm, inverse_vocab
            )
            results.append(res)
        except Exception as e:
            print(f"Erreur: {e}", file=sys.stderr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
