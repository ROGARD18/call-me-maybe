import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set
import numpy as np

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.models import FunctionsClass
from src.vocab_dict import make_vocab_dict


def force_generate(
    target_str: str,
    tokens: List[int],
    full_text: str,
    llm: Small_LLM_Model,
    inverse_vocab: Dict[int, str],
) -> str:
    """Force le LLM à générer une chaîne exacte, token par token."""
    progress: int = 0
    while progress < len(target_str):
        logits: np.ndarray = np.array(llm.get_logits_from_input_ids(tokens),
                                      dtype=np.float32)
        is_allowed: np.ndarray = np.zeros(len(logits), dtype=bool)

        for tid, tstr in inverse_vocab.items():
            clean: str = tstr.replace("Ġ", " ").replace("Ċ", "\n")
            clean = clean.replace("ĉ", "\t")
            if not clean:
                continue
            if target_str[progress:].startswith(clean):
                is_allowed[tid] = True

        if not np.any(is_allowed):
            for tid, tstr in inverse_vocab.items():
                clean = tstr.replace("Ġ", " ").replace("Ċ", "\n")
                clean = clean.replace("ĉ", "\t")
                if clean == target_str[progress]:
                    is_allowed[tid] = True
            if not np.any(is_allowed):
                is_allowed.fill(True)

        masked_logits: np.ndarray = np.where(is_allowed, logits, -1e10)
        next_token: int = int(np.argmax(masked_logits))
        clean_token: str = (
            inverse_vocab.get(next_token, "")
            .replace("Ġ", " ")
            .replace("Ċ", "\n")
            .replace("ĉ", "\t")
        )

        if not clean_token:
            break

        tokens.append(next_token)
        full_text += clean_token
        progress += len(clean_token)
        print(clean_token, end="", flush=True)

    return full_text


def generate_constrained_call(
    prompt_text: str,
    functions_class: FunctionsClass,
    llm: Small_LLM_Model,
    inverse_vocab: Dict[int, str],
) -> Dict[str, Any]:

    schema_lines: List[str] = []
    for name, f in functions_class.definitions.items():
        desc: str = f.get("description", "")
        params: Dict[str, Any] = f.get("parameters", {})
        param_str: str = ", ".join(
            f'"{k}": {v.get("type", "unknown")}' for k, v in params.items()
        )
        schema_lines.append(f"- {name}: {desc}\n  Parameters: {{{param_str}}}")

    context: str = (
        "You are an exact parameter extractor. Select the correct function"
        " and extract the literal arguments.\n"
        "Copy values exactly as they appear in the request, preserving full"
        " file paths, URLs, and special characters.\n"
        f"Available functions:\n{chr(10).join(schema_lines)}\n\n"
    )

    examples: List[Dict[str, Any]] = [
        {
            "prompt": "Read the file at /home/user/data.json with utf-8"
            " encoding",
            "name": "fn_read_file",
            "parameters": {
                "path": "/home/user/data.json", "encoding": "utf-8"},
        },
        {
            "prompt": 'Format template: Say "hello" to {name}',
            "name": "fn_format_template",
            "parameters": {"template": 'Say "hello" to {name}'},
        },
        {
            "prompt": "Calculate compound interest on 1234567.89 at 0.0375"
            " rate for 23 years",
            "name": "fn_calculate_compound_interest",
            "parameters": {
                "principal": 1234567.89, "rate": 0.0375, "years": 23},
        },
    ]

    for ex in examples:
        ex_str: str = json.dumps(ex, indent=2)
        context += f"Request: {ex['prompt']}\n{ex_str}\n\n"

    safe_prompt: str = (
        prompt_text.replace("\\", "\\\\").replace('"', '\\"')
    )
    safe_prompt = safe_prompt.replace("\n", "\\n")
    context += f"Request: {prompt_text}\n"

    prefix: str = f'{{\n  "prompt": "{safe_prompt}",\n  "name": "'

    full_text: str = context + prefix
    encoded: List[int] = llm.encode(full_text).tolist()
    tokens: List[int] = encoded[0] if isinstance(encoded[0], list) else encoded
    print(prefix, end="", flush=True)

    selected_name: str = ""
    current_name_str: str = ""

    for _ in range(50):
        logits = np.array(llm.get_logits_from_input_ids(tokens),
                          dtype=np.float32)
        is_allowed = np.zeros(len(logits), dtype=bool)

        for tid, tstr in inverse_vocab.items():
            clean: str = tstr.replace("Ġ", "").replace("Ċ", "")
            clean = clean.replace("ĉ", "")
            if not clean:
                continue

            if current_name_str in functions_class.list:
                if '"' in clean:
                    is_allowed[tid] = True
            else:
                potential: str = current_name_str + clean
                if any(fn.startswith(potential)
                       for fn in functions_class.list):
                    is_allowed[tid] = True

        if not np.any(is_allowed):
            is_allowed.fill(True)

        masked_logits = np.where(is_allowed, logits, -1e10)
        next_token: int = int(np.argmax(masked_logits))
        clean_token: str = (
            inverse_vocab.get(next_token, "")
            .replace("Ġ", "")
            .replace("Ċ", "")
            .replace("ĉ", "")
        )

        idx: int = clean_token.find('"')
        if idx != -1:
            chunk: str = clean_token[:idx]
            current_name_str += chunk
            full_text += chunk + '"'
            print(chunk + '"', end="", flush=True)
            selected_name = current_name_str

            encoded_new: List[int] = llm.encode(full_text).tolist()
            tokens.clear()
            tokens.extend(encoded_new[0] if isinstance(encoded_new[0], list)
                          else encoded_new)
            break
        else:
            current_name_str += clean_token
            full_text += clean_token
            print(clean_token, end="", flush=True)
            tokens.append(next_token)

    if selected_name not in functions_class.definitions:
        selected_name = (functions_class.list[0] if
                         functions_class.list else "unknown")

    expected_params: Dict[str, Any] = functions_class.definitions.get(
        selected_name, {}).get(
        "parameters", {}
    )
    params_obj: Dict[str, Any] = {}

    trans_str: str = ',\n  "parameters": {'
    if expected_params:
        trans_str += "\n"
    full_text = force_generate(trans_str, tokens, full_text, llm,
                               inverse_vocab)

    keys: List[str] = list(expected_params.keys())
    for i, key in enumerate(keys):
        param_type: str = expected_params[key].get("type", "string")
        is_numeric: bool = param_type in ("number", "integer")
        is_bool: bool = param_type == "boolean"

        key_str: str = f'    "{key}": '
        if not is_numeric and not is_bool:
            key_str += '"'

        full_text = force_generate(key_str, tokens, full_text, llm,
                                   inverse_vocab)

        val_raw: str = ""
        escape: bool = False

        for _ in range(150):
            logits = np.array(llm.get_logits_from_input_ids(tokens),
                              dtype=np.float32)
            is_allowed = np.zeros(len(logits), dtype=bool)

            if is_numeric:
                valid_chars: Set[str] = set("0123456789.-")
                stop_chars: Set[str] = set(",}\n ")
                for tid, tstr in inverse_vocab.items():
                    clean = tstr.replace("Ġ", " ").replace("Ċ", "\n")
                    clean = clean.replace("ĉ", "\t")
                    if not clean:
                        continue
                    if all(c in valid_chars for c in clean):
                        is_allowed[tid] = True
                    else:
                        for i_c, c in enumerate(clean):
                            if c in stop_chars:
                                is_allowed[tid] = True
                                break
                            if c not in valid_chars:
                                break
            elif is_bool:
                valid_chars = set("truefalseTRUEFALSE")
                stop_chars = set(",}\n ")
                for tid, tstr in inverse_vocab.items():
                    clean = tstr.replace("Ġ", " ").replace("Ċ", "\n")
                    clean = clean.replace("ĉ", "\t")
                    if not clean:
                        continue
                    if all(c in valid_chars for c in clean):
                        is_allowed[tid] = True
                    else:
                        for i_c, c in enumerate(clean):
                            if c in stop_chars:
                                is_allowed[tid] = True
                                break
                            if c not in valid_chars:
                                break
            else:
                for tid, tstr in inverse_vocab.items():
                    clean = tstr.replace("Ġ", " ").replace("Ċ", "\n")
                    clean = clean.replace("ĉ", "\t")
                    if not clean:
                        continue
                    if "\n" not in clean:
                        is_allowed[tid] = True
                    else:
                        idx_g: int = clean.find('"')
                        idx_n: int = clean.find("\n")
                        if idx_g != -1 and (idx_n == -1 or idx_g < idx_n):
                            is_allowed[tid] = True

            if not np.any(is_allowed):
                is_allowed.fill(True)

            masked_logits = np.where(is_allowed, logits, -1e10)
            next_token = int(np.argmax(masked_logits))
            clean_token = (
                inverse_vocab.get(next_token, "")
                .replace("Ġ", " ")
                .replace("Ċ", "\n")
                .replace("ĉ", "\t")
            )

            if is_numeric or is_bool:
                stop_idx: int = -1
                for i_c, c in enumerate(clean_token):
                    if c in stop_chars:
                        stop_idx = i_c
                        break
                if stop_idx != -1:
                    chunk = clean_token[:stop_idx]
                    val_raw += chunk
                    print(chunk, end="", flush=True)
                    full_text += chunk

                    encoded_new = llm.encode(full_text).tolist()
                    tokens.clear()
                    tokens.extend(
                        encoded_new[0] if isinstance(encoded_new[0], list)
                        else encoded_new
                    )
                    break
                else:
                    val_raw += clean_token
                    print(clean_token, end="", flush=True)
                    full_text += clean_token
                    tokens.append(next_token)
            else:
                stop_idx = -1
                for i_c, c in enumerate(clean_token):
                    if c == "\\" and not escape:
                        escape = True
                    elif c == '"' and not escape:
                        stop_idx = i_c
                        break
                    else:
                        escape = False

                if stop_idx != -1:
                    chunk = clean_token[: stop_idx + 1]
                    val_raw += chunk[:-1]
                    print(chunk, end="", flush=True)
                    full_text += chunk

                    encoded_new = llm.encode(full_text).tolist()
                    tokens.clear()
                    tokens.extend(
                        encoded_new[0] if isinstance(encoded_new[0], list)
                        else encoded_new
                    )
                    break
                else:
                    val_raw += clean_token
                    print(clean_token, end="", flush=True)
                    full_text += clean_token
                    tokens.append(next_token)

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
        else:
            params_obj[key] = val_raw.replace("\\", "")

        if i < len(keys) - 1:
            sep_str = ",\n"
            full_text = force_generate(sep_str, tokens, full_text, llm,
                                       inverse_vocab)
        else:
            sep_str = "\n  }"
            full_text = force_generate(sep_str, tokens, full_text, llm,
                                       inverse_vocab)

    end_str = "\n}"
    full_text = force_generate(end_str, tokens, full_text, llm, inverse_vocab)
    print("\n", flush=True)

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
