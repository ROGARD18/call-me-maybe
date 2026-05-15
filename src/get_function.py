import json


def get_functions() -> tuple[list, dict]:
    path = 'data/input/functions_definition.json'
    with open(path, 'r') as file:
        data = json.load(file)
    names = [f.get('name') for f in data]
    defs = {f['name']: f.get('parameters', {}) for f in data}
    return names, defs
