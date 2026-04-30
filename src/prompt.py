def make_prompt(text: str) -> str:
    return (
        "Generate a JSON with three line: <prompt>, "
        f"<name> and <parameters>. Like :"
        "["
        "   {"
        f"       'prompt': {text},"
        "        'name': 'function',"
        "        'parameters: 'parameter one, parameter two',"
        "   },"
        "]"
    )