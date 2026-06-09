import numpy as np
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from typing import List, Dict


def force_generate(
    target_str: str,
    tokens: List[int],
    full_text: str,
    llm: Small_LLM_Model,
    inverse_vocab: Dict[int, str],
) -> str:
    progress: int = 0
    while progress < len(target_str):
        logits: np.ndarray = np.array(llm.get_logits_from_input_ids(tokens),
                                      dtype=np.float32)
        is_allowed: np.ndarray = np.zeros(len(logits), dtype=bool)

        for tid, tstr in inverse_vocab.items():
            clean: str = tstr.replace("Ġ", " ").replace("Ċ", "\n")
            clean = clean.replace("ĉ", "\t")
            if not clean:
                continue
            if target_str[progress:].startswith(clean):
                is_allowed[tid] = True

        if not np.any(is_allowed):
            for tid, tstr in inverse_vocab.items():
                clean = tstr.replace("Ġ", " ").replace("Ċ", "\n")
                clean = clean.replace("ĉ", "\t")
                if clean == target_str[progress]:
                    is_allowed[tid] = True
            if not np.any(is_allowed):
                is_allowed.fill(True)

        masked_logits: np.ndarray = np.where(is_allowed, logits, -1e10)
        next_token: int = int(np.argmax(masked_logits))
        clean_token: str = (
            inverse_vocab.get(next_token, "")
            .replace("Ġ", " ")
            .replace("Ċ", "\n")
            .replace("ĉ", "\t")
        )

        if not clean_token:
            break

        tokens.append(next_token)
        full_text += clean_token
        progress += len(clean_token)
        print(clean_token, end="")

    return full_text
