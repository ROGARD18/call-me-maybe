import numpy as np
from src.all_step import JSONState


class GrammarConstrainedSampler:
    def __init__(self, grammar_valid_fn) -> None:
        self.grammar_valid_fn = grammar_valid_fn

    @staticmethod
    def make_set(text: str, step: int) -> set:
        if (step == 2):
            return (set(text).union('"'))
        return (set())

    def constrained_sample(
        self,
        js: JSONState,
        logits: np.ndarray,
        text: str,
        prompt: str,
        step: int,
        token_to_char: dict[int, str],
    ) -> int:
        target_string = self.grammar_valid_fn(step, js)
        print("Target ->", target_string)
        is_allowed = np.zeros(len(logits), dtype=bool)

        if target_string == "":
            set_char = self.make_set(text, step)
            # print (f"setchar == {set_char}")
            for char, id_ in token_to_char.items():
                clean_char = char.replace('Ġ', ' ').replace(
                    'Ċ', '\n').replace('ĉ', '\t')
                if ',' in clean_char or '.' in clean_char or '""' in clean_char:
                    continue
                elif clean_char and all(c in set_char for c in clean_char):
                    is_allowed[id_] = True
        else:
            for char, id_ in token_to_char.items():
                if ((not char) or (len(target_string) < len(char))):
                    continue
                elif target_string.startswith(char) or char.startswith(target_string):
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
        return np.nan_to_num(out, nan=1.0/len(x))


def step_json(step: int, js: JSONState) -> str:
    print("step =", step)
    if step == 0:
        return js.JSON_START
    elif step == 1:
        return js.KEY_PROMPT
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

    # if text.endswith('}'):
    #     return ",\n\t{"
    # last_line = text.split('\n')[-1]
    # if '"' in last_line and (last_line.count('"') % 2 == 1) and ":" not in last_line:
    #      return "prompt"
    return ("")
