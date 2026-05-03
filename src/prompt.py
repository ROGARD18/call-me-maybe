def make_prompt(text: str, functions: list) -> str:
    return (
        "Generate a JSON with three line: <prompt>, "
        "<name> and <parameters>. You need to choose the "
        f"function in name case, in this list: {functions}"
    )
