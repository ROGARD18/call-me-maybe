import os
import json


def make_vocab_dict(path: str) -> dict:
    with open(path, 'r') as f:
        dict_vocab = json.load(f)
        # print(f)
    return (dict_vocab)
