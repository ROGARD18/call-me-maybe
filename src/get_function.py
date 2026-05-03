import json
import os


def get_functions() -> list:
    try:
        path: str = 'data/input/functions_definition.json'
        file_dir = os.path.dirname(os.path.realpath('__file__'))
        file_name = os.path.join(file_dir, path)
        file_name = os.path.abspath(os.path.realpath(file_name))

        with open(file_name, 'r') as file:
            functions_data = json.load(file)
        functions_names: list = [f.get('name') for f in functions_data]
        return(functions_names)
    except Exception as e:
        print(e.errors)
