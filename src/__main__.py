import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from llm_sdk import Small_LLM_Model
from src.vocab_dict import make_vocab_dict
from src.prompt import make_prompt
from src.next_token import GrammarConstrainedSampler, step_json
from src.class_ import JSONState, FunctionsClass
from src.validate_json import validate_json as valid_js


def format_text(text: str) -> str:
    """Convert special tokens to their actual characters."""
    text = text.replace("Ġ", " ")
    text = text.replace("Ċ", "\n")
    text = text.replace("ĉ", "\t")
    return text


def new_step(
    text: str, token: str, step: int, js: JSONState, Functions: FunctionsClass
) -> int:
    """Manage state transitions during token generation."""
    if step == 0:
        js.JSON_START = js.JSON_START[len(token) :]
        return 1 if not js.JSON_START else 0
    elif step == 1:
        js.KEY_PROMPT = js.KEY_PROMPT[len(token) :]
        return 2 if not js.KEY_PROMPT else 1
    elif step == 2:
        js.PROMPT_VALUE = js.PROMPT_VALUE[len(token) :]
        return 3 if not js.PROMPT_VALUE else 2
    elif step == 4:
        js.KEY_NAME = js.KEY_NAME[len(token) :]
        return 5 if not js.KEY_NAME else 4
    elif step == 5:
        clean_token = token.replace("Ġ", "")
        Functions.list = [
            function_name
            for function_name in Functions.list
            if function_name.startswith(clean_token)
        ]
        for i in range(len(Functions.list)):
            Functions.list[i] = Functions.list[i][len(clean_token) :]
        return (
            6 if not Functions.list or any(len(f) == 0 for f in Functions.list) else 5
        )
    elif step == 7:
        js.KEY_PARA = js.KEY_PARA[len(token) :]
        return 8 if not js.KEY_PARA else 7
    elif step == 8:
        param_keys = list(Functions.definitions.get(js.FUNCTION or "", {}).keys())
        current_type = (
            (js.TYPES or [])[js.param_order]
            if js.TYPES and js.param_order < len(js.TYPES)
            else "string"
        )

        if js.sub_step == 0:
            if "{" in token:
                js.sub_step = 1
            elif "}" in token:
                if not (js.TYPES) or len(js.TYPES) == 0:
                    return 9
        elif js.sub_step == 1 and '"' in token:
            if js.param_order < len(param_keys):
                js.current_key_remaining = param_keys[js.param_order]
            js.sub_step = 2
        elif js.sub_step == 2:
            if js.current_key_remaining.startswith(token):
                js.current_key_remaining = js.current_key_remaining[len(token) :]
            if not js.current_key_remaining:
                js.sub_step = 3
        elif js.sub_step == 3 and '"' in token:
            js.sub_step = 4
        elif js.sub_step == 4 and ":" in token:
            js.sub_step = 5
        elif js.sub_step == 5 and "Ġ" in token:
            js.sub_step = 6
        elif js.sub_step == 6:
            if current_type == "number":
                if any(c.isdigit() for c in token) or "." in token:
                    js.value_started = True
                    js.number_char_count += len(token)
                elif "," == token and js.value_started:
                    js.param_order += 1
                    js.sub_step = 1
                    js.value_started = False
                    js.string_open = False
                    js.number_char_count = 0
                    js.string_char_count = 0
                elif "}" == token and js.value_started:
                    return 9
            else:
                if token == '"' and not js.value_started:
                    js.value_started = True
                    js.string_open = True
                elif token == '"' and js.string_open:
                    js.string_open = False
                elif js.string_open:
                    js.string_char_count += len(token)
                elif not js.string_open and js.value_started and token == ",":
                    js.param_order += 1
                    js.sub_step = 1
                    js.value_started = False
                    js.string_open = False
                    js.number_char_count = 0
                    js.string_char_count = 0
                elif not js.string_open and js.value_started and token == "}":
                    return 9
        return 8
    elif step == 9:
        js.JSON_END = js.JSON_END[len(token) :]
        return 10 if not js.JSON_END else 9
    elif step in [3, 6]:
        js.LINE_END = js.LINE_END[len(token) :]
        if not js.LINE_END:
            js.LINE_END = ",Ċĉĉ"
            return step + 1
        return step
    return step


def process_prompt(
    text: str,
    functions_class: FunctionsClass,
    llm: Small_LLM_Model,
    vocab: Dict[str, int],
    inverse_vocab: Dict[int, str],
) -> tuple[bool, Any]:
    """Process a single prompt and generate function call."""
    try:
        functions_class_copy = FunctionsClass(
            functions_class.list.copy(), functions_class.definitions
        )
        js: JSONState = JSONState(text, len(max(functions_class_copy.list, key=len)))
        prompt: str = make_prompt(text, functions_class_copy.list)
        prompt_ids: list[int] = llm.encode(prompt)[0].tolist()
        generate_text: str = ""

        cons_sampler = GrammarConstrainedSampler(grammar_valid_fn=step_json)
        step: int = 0

        print("[prompt] generating tokens")

        max_generation_steps = 1200
        generation_count = 0

        while step < 10:
            generation_count += 1
            if generation_count > max_generation_steps:
                return False, {"error": "Generation limit reached"}

            logits: list[float] = llm.get_logits_from_input_ids(prompt_ids)
            logits_array = np.array(logits)
            index_max = cons_sampler.constrained_sample(
                js, logits_array, text, prompt, functions_class_copy, step, vocab
            )

            next_token = inverse_vocab.get(index_max, "")

            if not next_token:
                continue

            generate_text += next_token
            prompt += next_token
            prompt_ids.append(index_max)
            print("<>")
            next_step: int = new_step(text, next_token, step, js, functions_class_copy)
            print(f"[prompt] step={step} token={next_token} " f"next_step={next_step}")
            print("<>")
            if next_step == 6 and step == 5:
                js.sub_step = 0
                js.param_order = 0
                js.current_key_remaining = ""
                js.value_started = False
                js.string_open = False
                js.number_char_count = 0
                js.string_char_count = 0
                js.FUNCTION = (
                    format_text(generate_text)
                    .split('"name": "')[-1]
                    .replace('"', "")
                    .strip()
                )
                print("<>")
                parameters: Dict[str, Any] = functions_class_copy.definitions.get(
                    js.FUNCTION, {}
                )
                types_list: list[str] = []
                for param in parameters:
                    param_type = parameters.get(param, {}).get("type", "string")
                    types_list.append(param_type)
                js.TYPES = types_list
                generate_text += '"'
                prompt += '"'
                print(f"[prompt] selected function: {js.FUNCTION}")

            step = next_step

        generate_text = format_text(generate_text)
        print(f"[prompt] generated text: {generate_text}")
        is_valid, _ = valid_js(generate_text)

        if not is_valid:
            print("[prompt] invalid JSON generated")
            return False, {"error": "Invalid JSON generated"}

        result = json.loads(generate_text)
        print("[prompt] valid JSON generated")
        return True, result
    except Exception as e:
        print(f"[prompt] error: {e}")
        return False, {"error": str(e)}


def load_json_file(file_path: str) -> Dict[str, Any] | list[Any] | None:
    """Load and parse a JSON file with error handling."""
    try:
        path = Path(file_path)
        if not path.exists():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            return None
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return None


def save_results(results: list[Dict[str, Any]], output_path: str) -> bool:
    """Save results to JSON file."""
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        return True
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        return False


def main() -> None:
    """Main entry point with argument parsing."""
    print("[main] starting")
    parser = argparse.ArgumentParser(
        description=("Function calling system using LLM with constrained decoding")
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to functions definition JSON file",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to input prompts JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to output results JSON file",
    )

    args = parser.parse_args()
    print(f"[main] args parsed: input={args.input} output={args.output}")

    # Load function definitions
    print("[main] loading function definitions")
    functions_data = load_json_file(args.functions_definition)
    if functions_data is None:
        sys.exit(1)

    names = [f.get("name") for f in functions_data]
    defs = {f["name"]: f.get("parameters", {}) for f in functions_data}
    functions_class = FunctionsClass(names, defs)
    print(f"[main] loaded {len(names)} functions")

    # Load input prompts
    print("[main] loading prompts")
    input_data = load_json_file(args.input)
    if input_data is None:
        sys.exit(1)

    if not isinstance(input_data, list):
        print(
            "Error: Input file must contain a JSON array of prompts",
            file=sys.stderr,
        )
        sys.exit(1)

    # Initialize LLM
    try:
        print("[main] initializing LLM")
        llm = Small_LLM_Model()
        vocab = make_vocab_dict(llm.get_path_to_vocab_file())
        inverse_vocab: Dict[int, str] = {int(v): k for k, v in vocab.items()}
        print(f"[main] vocab size: {len(vocab)}")
    except Exception as e:
        error_message = str(e).lower()
        if "cuda out of memory" in error_message or "out of memory" in error_message:
            print(
                "Warning: CUDA memory is not available, falling back to CPU.",
                file=sys.stderr,
            )
            try:
                llm = Small_LLM_Model(device="cpu")
                vocab = make_vocab_dict(llm.get_path_to_vocab_file())
                inverse_vocab = {int(v): k for k, v in vocab.items()}
            except Exception as fallback_error:
                print(
                    f"Error initializing LLM: {fallback_error}",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            print(f"Error initializing LLM: {e}", file=sys.stderr)
            sys.exit(1)

    # Process prompts
    print(f"[main] processing {len(input_data)} prompts")
    results: list[Dict[str, Any]] = []
    for i, item in enumerate(input_data):
        if not isinstance(item, dict) or "prompt" not in item:
            print(
                f"Warning: Invalid prompt item at index {i}",
                file=sys.stderr,
            )
            continue

        prompt_text = item["prompt"]
        print(f"[main] prompt {i + 1}/{len(input_data)}")
        success, result = process_prompt(
            prompt_text, functions_class, llm, vocab, inverse_vocab
        )

        if success:
            if isinstance(result, dict):
                result["prompt"] = prompt_text
                results.append(result)
            elif isinstance(result, list):
                for entry in result:
                    if isinstance(entry, dict):
                        entry["prompt"] = prompt_text
                        results.append(entry)
                    else:
                        print(
                            f"Warning: Unexpected result item at index {i}",
                            file=sys.stderr,
                        )
            else:
                print(
                    f"Warning: Unexpected result type at index {i}",
                    file=sys.stderr,
                )
        else:
            print(
                "Warning: Failed to process prompt "
                f"{i}: {result.get('error', 'Unknown error')}",
                file=sys.stderr,
            )

    # Save results
    print(f"[main] saving {len(results)} results to {args.output}")
    if not save_results(results, args.output):
        sys.exit(1)

    print(f"Successfully processed {len(results)} prompts")


if __name__ == "__main__":
    main()
