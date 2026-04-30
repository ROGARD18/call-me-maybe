import numpy as np


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