from typing import List


def make_prompt(text: str, functions: List[str]) -> str:
    """Create a prompt for the LLM to identify which function to call.

    Args:
        text: The user's natural language request
        functions: List of available function names

    Returns:
        Formatted prompt string
    """
    return (
        "Pick exactly one function name from the list "
        "below for the user request. "
        "Do not explain your choice and do not add any "
        "extra text. "
        f"Request: {text}. "
        f"Function names: {functions}."
    )
