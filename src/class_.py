from pydantic import BaseModel
from typing import Dict, Any, List, Optional


class FunctionsClass:
    """Container for available functions and their definitions."""

    def __init__(self, functions: List[str], definitions: Dict[str, Any]) -> None:
        """Initialize functions class.

        Args:
            functions: List of available function names
            definitions: Dictionary mapping function names to their parameters
        """
        self.list: List[str] = functions
        self.definitions: Dict[str, Any] = definitions


class FinalResult(BaseModel):
    """Final result structure."""

    prompt: str
    name: str
    parameters: Dict[str, Any]


class JSONState:
    """State machine for JSON generation."""

    def __init__(self, text: str, function_len: int) -> None:
        """Initialize JSON generation state.

        Args:
            text: Input prompt text
            function_len: Maximum function name length
        """
        self.JSON_START: str = "[Ċĉ{Ċĉĉ"
        self.JSON_END: str = "Ċĉ}Ċ]"
        self.LINE_END: str = ",Ċĉĉ"
        self.KEY_PROMPT: str = '"prompt":Ġ"'
        escaped_text = text.replace("\\", "\\\\").replace('"', '\\"')
        escaped_text = escaped_text.replace("\t", "ĉ").replace("\n", "Ċ")
        self.PROMPT_VALUE: str = f'{escaped_text.replace(" ", "Ġ")}"'
        self.KEY_NAME: str = '"name":Ġ"'
        self.KEY_PARA: str = '"parameters":Ġ'
        self.NAME_LEN: int = function_len
        self.FUNCTION: Optional[str] = None
        self.TYPES: Optional[List[str]] = None
        self.param_order: int = 0
        self.sub_step: int = 0
        self.current_key_remaining: str = ""
        self.value_started: bool = False
        self.string_open: bool = False
        self.number_char_count: int = 0
        self.string_char_count: int = 0
