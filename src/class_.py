from pydantic import BaseModel
from typing import Dict, Any


class FunctionsClass:
    def __init__(self, functions: list, definitions: dict) -> None:
        self.list: list = functions
        self.definitions: dict = definitions


class FinalResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]


class JSONState:
    def __init__(self, text: str, function_len: int):
        self.JSON_START = "[Ċĉ{Ċĉĉ"
        self.JSON_END = "Ċĉ}]"
        self.LINE_END = ",Ċĉĉ"
        self.KEY_PROMPT = '"prompt":Ġ"'
        self.PROMPT_VALUE = (
            f'{text.replace(" ", "Ġ").replace("\\t", "ĉ").replace("\\n", "Ċ")}"'
        )
        self.KEY_NAME = '"name":Ġ"'
        self.KEY_PARA = '"parameters":Ġ{'
        self.NAME_LEN = function_len
        self.FUNCTION = None
