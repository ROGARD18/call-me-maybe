import os
import json


def make_vocab_dict(path: str) -> dict:
    key_list: list = []
    value_list: list = []
    vocab: dict = {}

    file_dir = os.path.dirname(os.path.realpath('__file__'))

    file_name = os.path.join(file_dir, path)
    file_name = os.path.abspath(os.path.realpath(file_name))
    with open(file_name, 'r') as f:
        dict_vocab = json.load(f)
    return (dict_vocab)
