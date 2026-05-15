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
        self.JSON_END = "Ċĉ}Ċ]"
        self.LINE_END = ",Ċĉĉ"
        self.KEY_PROMPT = '"prompt":Ġ"'
        self.PROMPT_VALUE = (
            f'{text.replace(" ", "Ġ").replace("\\t", "ĉ").replace("\\n", "Ċ")}"'
        )
        self.KEY_NAME: str = '"name":Ġ"'
        self.KEY_PARA: str = '"parameters":Ġ'
        self.NAME_LEN: int = function_len
        self.FUNCTION = None
        self.TYPES: list | None = None
        self.param_order: int = 0
        self.sub_step: int = 0
