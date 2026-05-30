import numpy as np
from typing import Dict, List, Any
from src.models import FunctionsClass, JSONState


class GrammarConstrainedSampler:

    def constrained_sample(
        self,
        js: JSONState,
        logits: list[float],
        prompt: str,
        functions: FunctionsClass,
        step: int,
        vocab: Dict[str, int],
        inverse_vocab: Dict[int, str]
    ) -> int:
        logits_np = np.array(logits, dtype=np.float32)

        is_allowed = np.zeros(len(logits_np), dtype=bool)

        if step == 0:
            for token, id_ in vocab.items():
                if "{" in token or token in ("Ġ", "Ċ", "ĉ"):
                    is_allowed[id_] = True

        elif step == 5:
            for token, id_ in vocab.items():
                clean_token = token.replace("Ġ", "").replace(
                    "Ċ", "").replace("ĉ", "")
                if (clean_token and any(f.startswith(clean_token)
                                        for f in functions.list)):
                    is_allowed[id_] = True
                elif token in ('"', '",', 'Ġ'):
                    is_allowed[id_] = True

        elif step == 8:
            for token, id_ in vocab.items():
                if (any(c.isalnum() or c in '{}": ,.[]_-'
                        for c in token) or token == "Ġ"):
                    is_allowed[id_] = True

        else:
            is_allowed.fill(True)

        if not np.any(is_allowed):
            return int(np.argmax(logits_np))

        masked_logits = np.where(is_allowed, logits_np, -1e10)

        probabilities = self._softmax(masked_logits)
        return int(np.random.choice(len(logits_np), p=probabilities))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / (e_x.sum() + 1e-12)
