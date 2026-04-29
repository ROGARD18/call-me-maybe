import numpy as np

class SimpleArithmeticParser:

    def __init__(self) -> None:
        self.stack = ["E"]
        self.depth = 0

    def valid_next_json(self, curr_text: str) -> set[str]:
        last_char = curr_text[-1]
        if (depth == 0):
            return ({"{"})
        if (depth == 1):
            return ({"\n"})
        if (depth == 2):
            return ({'"'})
        if (last_char == ','):
            return ({"\n"})
        return (set())


class GrammarConstrainedSampler:

    def __init__(self, grammar_valid_fn) -> None:
        self.grammar_valid_fn = grammar_valid_fn

    def constrained_sample(
                self,
                logits: list[float],
                curr_text: str,
                token_to_char: dict[int, str],
    ) -> int:
        logits = np.array(logits)
        valid_chars = self.grammar_valid_fn(curr_text)

        valid_token_ids = {
            tid for tid, char in token_to_char.items() if char in valid_chars
        }
        
        masked_logits: np.darray = logits.copy()
        for tid in range(len(logits)):
            if tid not in valid_token_ids:
                masked_logits[tid] = float("-inf")
        
        probs = self._softmax(masked_logits)
        return np.random.choice(len(probs), p=probs)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()