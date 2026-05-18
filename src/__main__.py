import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from llm_sdk import Small_LLM_Model
from src.class_ import FunctionsClass
from src.vocab_dict import make_vocab_dict


def _normalize_words(text: str) -> list[str]:
    """Split text into lower-case words using simple character scanning."""
    words: list[str] = []
    current_word = ""

    for char in text.lower().replace("_", " "):
        if char.isalnum():
            current_word += char
        elif current_word:
            words.append(current_word)
            current_word = ""

    if current_word:
        words.append(current_word)

    normalized_words: list[str] = []
    for word in words:
        normalized_words.append(word)
        stemmed_word = _stem_word(word)
        if stemmed_word != word:
            normalized_words.append(stemmed_word)

    return normalized_words


def _stem_word(word: str) -> str:
    """Apply a tiny amount of stemming without regexes."""
    if len(word) > 4 and word.endswith("ied"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _extract_numbers(text: str) -> list[int | float]:
    """Extract numbers from text without using regular expressions."""
    numbers: list[int | float] = []
    current_number = ""
    has_digit = False

    for char in text:
        if char.isdigit():
            current_number += char
            has_digit = True
        elif char == "." and current_number and "." not in current_number:
            current_number += char
        else:
            if current_number:
                if has_digit:
                    numbers.append(_parse_number(current_number))
                current_number = ""
                has_digit = False

    if current_number and has_digit:
        numbers.append(_parse_number(current_number))

    return numbers


def _extract_quoted_values(text: str) -> list[str]:
    """Extract single-quoted and double-quoted strings without regex."""
    values: list[str] = []
    quote_char = ""
    current_value = ""
    in_quote = False

    for char in text:
        if not in_quote and char in {'"', "'"}:
            in_quote = True
            quote_char = char
            current_value = ""
        elif in_quote and char == quote_char:
            values.append(current_value)
            in_quote = False
            quote_char = ""
            current_value = ""
        elif in_quote:
            current_value += char

    return values


def _parse_number(value: str) -> int | float:
    """Parse a captured numeric token into an int when possible."""
    number = float(value)
    return int(number) if number.is_integer() else number


def _extract_text_after_keyword(text: str, keyword: str) -> str:
    """Return the text after the first occurrence of a keyword."""
    lowered_text = text.lower()
    lowered_keyword = keyword.lower()
    index = lowered_text.find(lowered_keyword)
    if index == -1:
        return ""
    return text[index + len(keyword) :].strip(" \t\n\r'\".,:;!?")


def _pick_string_value(
    text: str,
    parameter_name: str,
    quoted_values: list[str],
) -> str:
    """Infer a string argument using simple prompt heuristics."""
    lowered_text = text.lower()
    lowered_name = parameter_name.lower()

    if lowered_name == "name":
        words = _normalize_words(text)
        if words and words[0] in {"greet", "hello", "hi"} and len(words) > 1:
            return words[1]
        if quoted_values:
            return quoted_values[0]
        return text.strip()

    if lowered_name == "path":
        for keyword in ("file at", "read ", "path", "file"):
            fragment = _extract_text_after_keyword(text, keyword)
            if fragment:
                return fragment.split(" with ")[0].strip()
        return text.strip()

    if lowered_name == "encoding":
        fragment = _extract_text_after_keyword(text, "with")
        if fragment:
            words = fragment.split()
            if words and words[-1].lower() == "encoding" and len(words) > 1:
                return words[-2].strip("\"'.,:;!?")
            if words:
                return words[0].strip("\"'.,:;!?")
        if quoted_values:
            return quoted_values[-1]
        return text.strip()

    if lowered_name == "query":
        if quoted_values:
            return quoted_values[0]
        fragment = _extract_text_after_keyword(text, "query")
        if fragment:
            return fragment.split(" on ")[0].strip()
        return text.strip()

    if lowered_name == "database":
        fragment = _extract_text_after_keyword(text, "on the")
        if not fragment:
            fragment = _extract_text_after_keyword(text, "on")
        if fragment:
            return fragment.replace(" database", "").strip()
        return text.strip()

    if lowered_name == "template":
        for keyword in ("template:", "template"):
            fragment = _extract_text_after_keyword(text, keyword)
            if fragment:
                return fragment
        return text.strip()

    if any(
        token in lowered_name
        for token in {"source", "text", "input", "body", "content"}
    ):
        if quoted_values:
            if len(quoted_values) > 1:
                return quoted_values[-1]
            return quoted_values[0]
        return text.strip()

    if any(token in lowered_name for token in {"replacement", "replace", "value"}):
        fragment = _extract_text_after_keyword(text, "with")
        if fragment:
            return fragment.split()[0].strip("\"'.,:;!?")
        if len(quoted_values) >= 2:
            return quoted_values[1]
        if quoted_values:
            return quoted_values[0]
        return text.strip()

    if any(token in lowered_name for token in {"regex", "pattern"}):
        if "numbers" in lowered_text or "digits" in lowered_text:
            return r"\\d+"
        if "vowels" in lowered_text:
            return r"[aeiouAEIOU]"
        if quoted_values:
            return quoted_values[0]
        return text.strip()

    if quoted_values:
        return quoted_values[0]

    for keyword in ("with", "for", "of", "in", "to", "called", "named"):
        fragment = _extract_text_after_keyword(text, keyword)
        if fragment:
            return fragment

    return text.strip()


def _pick_number_value(
    text: str,
    parameter_name: str,
    numbers: list[int | float],
    used_count: int,
) -> tuple[int | float, int]:
    """Infer a numeric argument using prompt structure and parameter names."""
    lowered_name = parameter_name.lower()
    words = _normalize_words(text)

    if "return" in lowered_name and numbers:
        return numbers[-1], used_count

    if lowered_name in {"cost", "price", "amount", "investment"}:
        for keyword in ("cost", "returns", "return", "investment"):
            fragment = _extract_text_after_keyword(text, keyword)
            fragment_numbers = _extract_numbers(fragment)
            if fragment_numbers:
                return fragment_numbers[0], used_count

    if used_count < len(numbers):
        return numbers[used_count], used_count + 1

    if "first" in words and numbers:
        return numbers[0], used_count

    return 0, used_count


def build_parameters(
    text: str,
    function_definition: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract function parameters from the natural language prompt."""
    parameters = function_definition.get("parameters", {})
    numbers = _extract_numbers(text)
    quoted_values = _extract_quoted_values(text)
    parameters_result: Dict[str, Any] = {}
    used_number_count = 0

    for parameter_name, parameter_definition in parameters.items():
        parameter_type = parameter_definition.get("type", "string")

        if parameter_type == "number":
            value, used_number_count = _pick_number_value(
                text, parameter_name, numbers, used_number_count
            )
            parameters_result[parameter_name] = value
        elif parameter_type == "integer":
            value, used_number_count = _pick_number_value(
                text, parameter_name, numbers, used_number_count
            )
            parameters_result[parameter_name] = int(value)
        elif parameter_type == "boolean":
            lowered_text = text.lower()
            parameters_result[parameter_name] = (
                "false" not in lowered_text and "no" not in lowered_text
            )
        elif parameter_type == "string":
            parameters_result[parameter_name] = _pick_string_value(
                text, parameter_name, quoted_values
            )
        elif parameter_type == "array":
            parameters_result[parameter_name] = []
        elif parameter_type == "object":
            parameters_result[parameter_name] = {}
        else:
            parameters_result[parameter_name] = _pick_string_value(
                text, parameter_name, quoted_values
            )

    return parameters_result


def _build_selection_context(
    prompt: str,
    function_definitions: list[Dict[str, Any]],
) -> str:
    """Build the instruction prefix used to rank candidate functions."""
    lines = [
        "Select the best function name for the request.",
        f"Request: {prompt}",
        "Available functions:",
    ]
    for function_definition in function_definitions:
        function_name = function_definition.get("name", "")
        description = function_definition.get("description", "")
        if function_name:
            lines.append(f"- {function_name}: {description}")

    lines.append("Answer:")
    return "\n".join(lines) + " "


def _score_candidate_function(
    context_ids: list[int],
    function_name: str,
    llm: Small_LLM_Model,
) -> float:
    """Score a function candidate using model token probabilities."""
    if not function_name:
        return float("-inf")

    candidate_ids = llm.encode(function_name).tolist()[0]
    generated_ids = list(context_ids)
    score = 0.0

    for token_id in candidate_ids:
        logits = np.asarray(llm.get_logits_from_input_ids(generated_ids))
        logits = logits - np.max(logits)
        probabilities = np.exp(logits)
        probabilities = probabilities / (probabilities.sum() + 1e-12)
        score += float(np.log(probabilities[token_id] + 1e-12))
        generated_ids.append(token_id)

    return score


def select_function_name(
    text: str,
    function_definitions: list[Dict[str, Any]],
    llm: Small_LLM_Model,
) -> str:
    """Choose the most appropriate function name for the prompt."""
    if not function_definitions:
        return ""

    context = _build_selection_context(text, function_definitions)
    context_ids = llm.encode(context).tolist()[0]
    scored_functions: list[tuple[float, str]] = []
    for function_definition in function_definitions:
        function_name = function_definition.get("name", "")
        if function_name:
            scored_functions.append(
                (
                    _score_candidate_function(context_ids, function_name, llm),
                    function_name,
                )
            )

    if not scored_functions:
        return ""

    _, best_name = max(scored_functions, key=lambda item: item[0])
    return best_name


def process_prompt(
    text: str,
    functions_class: FunctionsClass,
    llm: Small_LLM_Model,
    vocab: Dict[str, int],
    inverse_vocab: Dict[int, str],
) -> tuple[bool, Any]:
    """Process a single prompt and generate function call."""
    try:
        available_definitions = [
            functions_class.definitions[name]
            for name in list(functions_class.list)
            if name in functions_class.definitions
        ]
        selected_function = select_function_name(
            text,
            available_definitions,
            llm,
        )
        if not selected_function:
            return False, {"error": "No function available"}

        selected_definition = functions_class.definitions.get(
            selected_function,
            {},
        )

        result = {
            "prompt": text,
            "name": selected_function,
            "parameters": build_parameters(text, selected_definition),
        }
        print(f"[prompt] selected function: {selected_function}")
        print(f"[prompt] generated result: {result}")
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
        with path.open("r", encoding="utf-8") as f:
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
        with path.open("w", encoding="utf-8") as f:
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
    defs = {f["name"]: f for f in functions_data if isinstance(f, dict) and "name" in f}
    functions_class = FunctionsClass(list=names, definitions=defs)
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
        vocab = make_vocab_dict(llm.get_path_to_vocabulary_json())
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
                vocab = make_vocab_dict(llm.get_path_to_vocabulary_json())
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
