from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class FunctionsClass(BaseModel):
    """Container for available functions and their definitions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    list: List[str]
    definitions: Dict[str, Any]


class FinalResult(BaseModel):
    """Final result structure."""

    prompt: str
    name: str
    parameters: Dict[str, Any]


class JSONState(BaseModel):
    """State machine for JSON generation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    JSON_START: str
    JSON_END: str
    LINE_END: str
    KEY_PROMPT: str
    PROMPT_VALUE: str
    KEY_NAME: str
    KEY_PARA: str
    NAME_LEN: int
    FUNCTION: Optional[str] = None
    TYPES: Optional[List[str]] = None
    param_order: int = 0
    sub_step: int = 0
    current_key_remaining: str = ""
    value_started: bool = False
    string_open: bool = False
    number_char_count: int = 0
    string_char_count: int = 0

    def __init__(self, text: str, function_len: int) -> None:
        escaped_text = text.replace("\\", "\\\\").replace('"', '\\"')
        escaped_text = escaped_text.replace("\t", "ĉ").replace("\n", "Ċ")
        super().__init__(
            JSON_START="[Ċĉ{Ċĉĉ",
            JSON_END="Ċĉ}Ċ]",
            LINE_END=",Ċĉĉ",
            KEY_PROMPT='"prompt":Ġ"',
            PROMPT_VALUE=f'{escaped_text.replace(" ", "Ġ")}"',
            KEY_NAME='"name":Ġ"',
            KEY_PARA='"parameters":Ġ',
            NAME_LEN=function_len,
        )
