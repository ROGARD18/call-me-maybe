from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class JSONSchema(BaseModel):
    """Minimal JSON schema model used for validation and constrained output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: str
    properties: Optional[Dict[str, "JSONSchema"]] = None
    required: Optional[List[str]] = None
    items: Optional["JSONSchema"] = None
    enum: Optional[List[Any]] = None
