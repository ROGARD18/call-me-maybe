import numpy as np
from src.class_ import JSONState


class GrammarConstrainedSampler:
    def __init__(self, grammar_valid_fn) -> None:
        self.grammar_valid_fn = grammar_valid_fn

    def constrained_sample(
        self, js, logits, text, prompt, Functions, step, vocab
    ) -> int:
        target_string = self.grammar_valid_fn(step, js)
        is_allowed = np.zeros(len(logits), dtype=bool)

        if step == 5:
            for char, id_ in vocab.items():
                clean = char.replace("Ġ", "").replace("Ċ", "").replace("ĉ", "")
                if clean and any(f.startswith(clean) for f in Functions.list):
                    is_allowed[id_] = True

        elif step == 8:
            is_allowed.fill(False)
            params_keys = list(Functions.definitions.get(js.FUNCTION,
                                                         {}).keys())

            if js.param_order >= len(js.TYPES):
                if "}" in vocab:
                    is_allowed[vocab["}"]] = True
                    masked_logits = np.where(is_allowed, logits, -1e10)
                    return int(
                        np.random.choice(len(logits),
                                         p=self._softmax(masked_logits))
                    )
            else:
                current_letter = params_keys[js.param_order]
                current_type = js.TYPES[js.param_order]

                for char, id_ in vocab.items():
                    if js.sub_step == 0:
                        if char == "{":
                            is_allowed[id_] = True
                    if js.sub_step == 1 or js.sub_step == 3:
                        if char == '"':
                            is_allowed[id_] = True
                    elif js.sub_step == 2:
                        if char == current_letter:
                            is_allowed[id_] = True
                            is_allowed[id_] = True
                    elif js.sub_step == 4:
                        if char == ":":
                            is_allowed[id_] = True
                    elif js.sub_step == 5 or js.sub_step == -1:
                        if char == "Ġ":
                            is_allowed[id_] = True
                    elif js.sub_step == 6:
                        if js.param_order == len(js.TYPES):
                            if "{" == char:
                                is_allowed[id_] = True
                        elif "," == char:
                            is_allowed[id_] = True
                        if current_type == "number":
                            valid_nums = "0123456789."
                            if all(c in valid_nums for c in char):
                                is_allowed[id_] = True
                        else:
                            if all(c in text + '"' for c in char):
                                is_allowed[id_] = True

        elif target_string != "":
            for char, id_ in vocab.items():
                if target_string.startswith(char):
                    is_allowed[id_] = True
        else:
            is_allowed.fill(True)

        if not np.any(is_allowed):
            return int(np.argmax(logits))

        masked_logits = np.where(is_allowed, logits, -1e10)
        return int(np.random.choice(len(logits),
                                    p=self._softmax(masked_logits)))

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / (e_x.sum() + 1e-12)


def step_json(step: int, js: JSONState) -> str:
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
