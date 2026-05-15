from typing import Callable, Dict

import numpy as np

from src.class_ import FunctionsClass, JSONState


class GrammarConstrainedSampler:
    """Constrained sampling to enforce grammar rules during generation."""

    def __init__(self, grammar_valid_fn: Callable[[int, JSONState], str]) -> None:
        self.grammar_valid_fn = grammar_valid_fn

    def constrained_sample(
        self,
        js: JSONState,
        logits: np.ndarray,
        text: str,
        prompt: str,
        Functions: FunctionsClass,
        step: int,
        vocab: Dict[str, int],
    ) -> int:
        del prompt  # not used in constrained masking

        target_string = self.grammar_valid_fn(step, js)
        is_allowed: np.ndarray = np.zeros(len(logits), dtype=bool)

        if step == 5:
            for char, id_ in vocab.items():
                clean = char.replace("Ġ", "").replace("Ċ", "").replace("ĉ", "")
                if clean and any(f.startswith(clean) for f in Functions.list):
                    is_allowed[id_] = True

        elif step == 8:
            is_allowed.fill(False)
            params_keys: list[str] = list(
                Functions.definitions.get(js.FUNCTION or "", {}).keys()
            )
            current_type = (
                (js.TYPES or [])[js.param_order]
                if js.TYPES and js.param_order < len(js.TYPES)
                else "string"
            )

            def clean_token(token: str) -> str:
                return token.replace("Ġ", "").replace("Ċ", "").replace("ĉ", "")

            if js.param_order >= len(js.TYPES or []):
                if js.sub_step == 0 and "{" in vocab:
                    is_allowed[vocab["{"]] = True
                if "}" in vocab:
                    is_allowed[vocab["}"]] = True
            else:
                current_key = params_keys[js.param_order]
                key_remaining = js.current_key_remaining or current_key

                for char, id_ in vocab.items():
                    if js.sub_step == 0:
                        if char == "{":
                            is_allowed[id_] = True
                    elif js.sub_step in (1, 3):
                        if char == '"':
                            is_allowed[id_] = True
                    elif js.sub_step == 2:
                        cleaned = clean_token(char)
                        if cleaned and key_remaining.startswith(cleaned):
                            is_allowed[id_] = True
                    elif js.sub_step == 4:
                        if char == ":":
                            is_allowed[id_] = True
                    elif js.sub_step == 5:
                        if char == "Ġ":
                            is_allowed[id_] = True
                    elif js.sub_step == 6:
                        if current_type == "number":
                            cleaned = clean_token(char)
                            is_numeric = cleaned and all(
                                c in "0123456789." for c in cleaned
                            )
                            if is_numeric and js.number_char_count < 12:
                                is_allowed[id_] = True
                            if js.value_started:
                                if js.TYPES and js.param_order >= len(js.TYPES) - 1:
                                    if char == "}":
                                        is_allowed[id_] = True
                                elif char == ",":
                                    is_allowed[id_] = True
                        else:
                            if not js.value_started:
                                if char == '"':
                                    is_allowed[id_] = True
                            elif js.string_open:
                                if char == '"':
                                    is_allowed[id_] = True
                                else:
                                    cleaned = clean_token(char)
                                    if (
                                        js.string_char_count < 12
                                        and cleaned
                                        and cleaned in text
                                        and '"' not in char
                                        and "Ċ" not in char
                                        and "ĉ" not in char
                                    ):
                                        is_allowed[id_] = True
                            else:
                                if js.TYPES and js.param_order >= len(js.TYPES) - 1:
                                    if char == "}":
                                        is_allowed[id_] = True
                                elif char == ",":
                                    is_allowed[id_] = True

            if not np.any(is_allowed):
                if js.sub_step == 0 and "{" in vocab:
                    return int(vocab["{"])
                if js.sub_step in (1, 3) and '"' in vocab:
                    return int(vocab['"'])
                if js.sub_step == 4 and ":" in vocab:
                    return int(vocab[":"])
                if js.sub_step == 5 and "Ġ" in vocab:
                    return int(vocab["Ġ"])
                if js.sub_step == 6:
                    if current_type == "number":
                        if js.value_started and js.TYPES:
                            is_last = js.param_order >= len(js.TYPES) - 1
                            if is_last and "}" in vocab:
                                return int(vocab["}"])
                            if not is_last and "," in vocab:
                                return int(vocab[","])
                        if "0" in vocab:
                            return int(vocab["0"])
                    else:
                        if '"' in vocab:
                            return int(vocab['"'])
                if "}" in vocab:
                    return int(vocab["}"])
                return int(np.argmax(logits))

            masked_logits = np.where(is_allowed, logits, -1e10)
            return int(np.random.choice(len(logits), p=self._softmax(masked_logits)))

        elif target_string != "":
            for char, id_ in vocab.items():
                if target_string.startswith(char):
                    is_allowed[id_] = True
        else:
            is_allowed.fill(True)

        if not np.any(is_allowed):
            return int(np.argmax(logits))

        masked_logits = np.where(is_allowed, logits, -1e10)
        return int(np.random.choice(len(logits), p=self._softmax(masked_logits)))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x: np.ndarray = np.exp(x - np.max(x))
        return e_x / (e_x.sum() + 1e-12)


def step_json(step: int, js: JSONState) -> str:
    if step == 0:
        return js.JSON_START
    if step == 1:
        return js.KEY_PROMPT
    if step == 2:
        return js.PROMPT_VALUE
    if step == 3:
        return js.LINE_END
    if step == 4:
        return js.KEY_NAME
    if step == 6:
        return js.LINE_END
    if step == 7:
        return js.KEY_PARA
    if step == 9:
        return js.JSON_END
    return ""
