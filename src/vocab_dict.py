import json
from typing import Dict, Any


def make_vocab_dict(path: str) -> Dict[str, Any]:
    """Load vocabulary dictionary from JSON file.

    Args:
        path: Path to vocabulary JSON file

    Returns:
        Dictionary mapping tokens to their IDs
    """
    with open(path, "r") as f:
        dict_vocab: Dict[str, Any] = json.load(f)
    return dict_vocab
