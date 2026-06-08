import json
from typing import Dict, Any


def make_vocab_dict(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        vocab: Dict[str, Any] = json.load(f)
        return vocab
