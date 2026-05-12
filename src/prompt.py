def make_prompt(text: str, functions: list) -> str:
    return (
        f"Find in this list the good one for 'name': {functions}"
    )
