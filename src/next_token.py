import numpy as np
from src.class_ import JSONState, FunctionsClass


class GrammarConstrainedSampler:
    def __init__(self, grammar_valid_fn) -> None:
        self.grammar_valid_fn = grammar_valid_fn

    @staticmethod
    def make_set(text: str, step: int) -> set:
        if step == 8:
            return set(text + '"')
        return set()

    def constrained_sample(
        self,
        js: JSONState,
        logits: np.ndarray,
        text: str,
        prompt: str,
        Functions: FunctionsClass,
        step: int,
        vocab: dict[int, str],
    ) -> int:
        target_string = self.grammar_valid_fn(step, js)
        print("Target ->", target_string)
        print("step =", step)
        is_allowed = np.zeros(len(logits), dtype=bool)

        if step == 5:
            is_allowed.fill(False)

            for char, id_ in vocab.items():
                clean_token = char.replace("Ġ", "")
                if not clean_token:
                    continue

                if any(func.startswith(clean_token) for func in Functions.list):
                    is_allowed[id_] = True
        elif (step == 8):
            allowed_chars = set(text + '0123456789.:,"{} ')
            for char, id_ in vocab.items():
                clean = char.replace('Ġ', ' ').replace('Ċ', '\n').replace('ĉ', '\t')
                if clean and all(c in allowed_chars for c in clean):
                    is_allowed[id_] = True

        elif target_string == "":
            set_char = self.make_set(text, step)
            print(f"setchar == {set_char}")
            for char, id_ in vocab.items():
                clean_char = (
                    char.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")
                )
                if "," in clean_char or "." in clean_char:
                    continue
                elif clean_char and all(c in set_char for c in clean_char):
                    is_allowed[id_] = True
        else:
            for char, id_ in vocab.items():
                if target_string.startswith(char):
                    is_allowed[id_] = True

        if not np.any(is_allowed):
            print("---------------")
            return int(np.argmax(logits))

        masked_logits = np.where(is_allowed, logits, -1e10)
        probs = self._softmax(masked_logits)
        return int(np.random.choice(len(probs), p=probs))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        sum_e_x = e_x.sum()
        if sum_e_x == 0:
            return np.ones(len(x)) / len(x)
        out = e_x / (sum_e_x + 1e-12)
        return np.nan_to_num(out, nan=1.0 / len(x))


def step_json(step: int, js: JSONState) -> str:
    # print("step =", step)
    if step == 0:
        return js.JSON_START
    elif step == 1:
        return js.KEY_PROMPT
    elif step == 2:
        return js.PROMPT_VALUE
    elif step == 3:
        return js.LINE_END
    elif step == 4:
        return js.KEY_NAME
    elif step == 6:
        return js.LINE_END
    elif step == 7:
        return js.KEY_PARA
    elif step == 9:
        return js.JSON_END
    return ""
