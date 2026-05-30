from __future__ import annotations

from typing import Any, Dict
import math

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from src.models import FunctionsClass


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

    for char in text:
        if char.isdigit() or (char == "." and "." not in current_number):
            current_number += char
        else:
            if current_number:
                numbers.append(_parse_number(current_number))
                current_number = ""

    if current_number:
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
    return text[index + len(keyword):].strip(" \t\n\r'\".,:;!?")


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

    if any(
        token in lowered_name for token in {"source", "text", "input", "s"}
    ):
        if quoted_values:
            if len(quoted_values) > 1:
                return quoted_values[-1]
            return quoted_values[0]
        return text.strip()

    if any(
        token in lowered_name
        for token in {"replacement", "replace", "value"}
    ):
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


def _selection_prompt(
    text: str,
    function_definitions: list[Dict[str, Any]],
) -> str:
    """Build the prompt used to ask the LLM for the right function name."""
    prompt_lines = [
        "Pick the single best function name for the request.",
        "Return only one exact function name from the list.",
        f"Request: {text}",
        "Available functions:",
    ]
    for function_definition in function_definitions:
        name = function_definition.get("name", "")
        description = function_definition.get("description", "")
        parameters = ", ".join(
            function_definition.get("parameters", {}).keys()
        )
        if isinstance(name, str):
            prompt_lines.append(
                f"- {name}: {description} | parameters: {parameters}"
            )
    return "\n".join(prompt_lines)


def _candidate_variants(name: str) -> list[str]:
    """Return a small set of tokenization variants for a candidate name."""
    variants: list[str] = [name, f" {name}", f"\n{name}"]
    unique_variants: list[str] = []
    for variant in variants:
        if variant not in unique_variants:
            unique_variants.append(variant)
    return unique_variants


def _token_ids_from_text(llm: Small_LLM_Model, text: str) -> list[int]:
    """Encode text and always return a flat token-id list."""
    encoded = llm.encode(text)
    token_ids = encoded.tolist()
    if token_ids and isinstance(token_ids[0], list):
        return [int(token_id) for token_id in token_ids[0]]
    return [int(token_id) for token_id in token_ids]


def _sequence_log_probability(
    llm: Small_LLM_Model,
    context_ids: list[int],
    token_ids: list[int],
) -> float:
    """Compute the log-probability of a candidate token sequence."""
    if not token_ids:
        return float("-inf")

    score = 0.0
    running_ids = list(context_ids)
    for token_id in token_ids:
        logits = llm.get_logits_from_input_ids(running_ids)
        logits_list = [float(value) for value in logits]
        max_logit = max(logits_list)
        sum_exp = sum(math.exp(value - max_logit) for value in logits_list)
        log_z = math.log(sum_exp) + max_logit
        score += logits_list[token_id] - log_z
        running_ids.append(token_id)
    return score / float(len(token_ids))


def _select_function_name_with_llm(
    text: str,
    function_definitions: list[Dict[str, Any]],
    llm: Small_LLM_Model,
    vocab: Dict[str, int],
) -> str | None:
    """Select the best function name using LLM candidate scoring."""
    if not function_definitions:
        return None

    function_names = [
        str(function_definition.get("name", ""))
        for function_definition in function_definitions
        if isinstance(function_definition.get("name", ""), str)
    ]
    if not function_names:
        return None

    prompt = _selection_prompt(text, function_definitions)
    context_ids = _token_ids_from_text(llm, prompt)

    best_name: str | None = None
    best_score = float("-inf")

    for function_name in function_names:
        candidate_score = float("-inf")
        for variant in _candidate_variants(function_name):
            token_ids = _token_ids_from_text(llm, variant)
            score = _sequence_log_probability(llm, context_ids, token_ids)
            if score > candidate_score:
                candidate_score = score
        if candidate_score > best_score:
            best_score = candidate_score
            best_name = function_name

    return best_name


def process_prompt(
    text: str,
    functions_class: FunctionsClass,
    llm: Small_LLM_Model,
    vocab: Dict[str, int],
) -> tuple[bool, Any]:
    """Process a single prompt and generate function call data."""
    try:
        available_definitions = [
            functions_class.definitions[name]
            for name in functions_class.list
            if name in functions_class.definitions
        ]
        selected_function = _select_function_name_with_llm(
            text,
            available_definitions,
            llm,
            vocab,
        )
        if not selected_function:
            return False, {"error": "No exact function match found"}

        selected_definition = functions_class.definitions.get(
            selected_function, {}
        )

        result = {
            "prompt": text,
            "name": selected_function,
            "parameters": build_parameters(text, selected_definition),
        }
        print(f"[prompt] selected function: {selected_function}")
        print(f"[prompt] generated result: {result}")
        return True, result
    except Exception as error:
        print(f"[prompt] error: {error}")
        return False, {"error": str(error)}
