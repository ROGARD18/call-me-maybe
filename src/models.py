from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict


class FunctionsClass(BaseModel):
    """Container for available functions and their definitions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    list: List[str]
    definitions: Dict[str, Any]
