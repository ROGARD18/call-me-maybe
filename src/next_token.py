from typing import Callable, Dict, Any

import numpy as np

from src.models import FunctionsClass, JSONState


class GrammarConstrainedSampler:
    """Constrained sampling to enforce grammar rules during generation."""

    def __init__(
        self,
        grammar_valid_fn: Callable[[int, JSONState], str],
    ) -> None:
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
        target_string = self.grammar_valid_fn(step, js)
        is_allowed: np.ndarray = np.zeros(len(logits), dtype=bool)

        if step == 5:
            for char, id_ in vocab.items():
                clean = char.replace("Ġ", "").replace("Ċ", "").replace("ĉ", "")
                if clean and any(f.startswith(clean) for f in Functions.list):
                    is_allowed[id_] = True

        elif step == 8:
            # Generate JSON object for parameters
            is_allowed.fill(False)

            # Check if we've opened the brace by looking at the prompt
            has_opened_brace = "{" in prompt

            for char, id_ in vocab.items():
                cleaned = char.replace("Ġ", "")
                cleaned = cleaned.replace("Ċ", "")
                cleaned = cleaned.replace("ĉ", "")

                # If not opened yet, prioritize opening brace
                if not has_opened_brace:
                    if char == "{":
                        is_allowed[id_] = True
                    elif char in ("Ġ",):  # Allow spaces before brace
                        is_allowed[id_] = True
                # If opened, allow structural content
                else:
                    if (
                        char in '{}": ,.'
                        or char == "Ġ"
                        or (
                            cleaned and all(c.isalnum() or c in "-_ ."
                                            for c in cleaned)
                        )
                    ):
                        is_allowed[id_] = True

            if not np.any(is_allowed):
                if "{" in vocab:
                    return int(vocab["{"])
                is_allowed.fill(True)

            masked_logits = np.where(is_allowed, logits, -1e10)
            return int(
                np.random.choice(
                    len(logits),
                    p=self._softmax(masked_logits),
                )
            )

        elif target_string != "":
            for char, id_ in vocab.items():
                if target_string.startswith(char):
                    is_allowed[id_] = True
        else:
            is_allowed.fill(True)

        if not np.any(is_allowed):
            return int(np.argmax(logits))

        masked_logits = np.where(is_allowed, logits, -1e10)
        return int(
            np.random.choice(
                len(logits),
                p=self._softmax(masked_logits),
            )
        )

    def _softmax(self, x: np.ndarray) -> Any:
        e_x: np.ndarray = np.exp(x - np.max(x))
        return e_x / (e_x.sum() + 1e-12)


def step_json(step: int, js: JSONState) -> str:
    if step == 0:
        return str(js.JSON_START)
    if step == 1:
        return str(js.KEY_PROMPT)
    if step == 2:
        return str(js.PROMPT_VALUE)
    if step == 3:
        return str(js.LINE_END)
    if step == 4:
        return str(js.KEY_NAME)
    if step == 6:
        return str(js.LINE_END)
    if step == 7:
        return str(js.KEY_PARA)
    if step == 9:
        return str(js.JSON_END)
    return ""
