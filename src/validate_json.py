import json
from typing import Tuple


def validate_json(text: str) -> Tuple[bool, str]:
    """Validate JSON string.

    Args:
        text: JSON string to validate

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        json.loads(text)
        return True, "Valid JSON"
    except json.JSONDecodeError as e:
        return False, f"Invalid: {e.msg}"
