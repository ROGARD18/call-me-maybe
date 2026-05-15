import json
from typing import Any, Dict, List, Tuple


def get_functions() -> Tuple[List[str], Dict[str, Any]]:
    """Load functions from definition file.

    Returns:
        Tuple of (function_names, function_definitions)
    """
    path = "data/input/functions_definition.json"
    with open(path, "r") as file:
        data: List[Dict[str, Any]] = json.load(file)
    names: List[str] = [f.get("name", "") for f in data]
    defs: Dict[str, Any] = {f["name"]: f.get("parameters", {}) for f in data}
    return names, defs
