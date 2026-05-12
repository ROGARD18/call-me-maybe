class JSONState():
    def __init__(self, text: str, function_len: int):
        self.JSON_START = '[Ċĉ{Ċĉĉ'
        self.JSON_END = 'Ċĉ},Ċ]'
        self.LINE_END = ',Ċĉĉ'
        self.KEY_PROMPT = '"prompt":Ġ"'
        self.PROMPT_VALUE = f'{text.replace(' ', 'Ġ').replace('\t', 'ĉ').replace('\n', 'Ċ')}"'
        self.KEY_NAME = '"name":Ġ"'
        self.KEY_PARA = '"parameters":Ġ"'
        self.NAME_LEN = function_len
        self.FUNCTION = None


class FunctionsClass():
    def __init__(self, functions: list) -> None:
        self.list: list = functions
        self.valid_functions: list = [False for _ in range(len(functions))]
