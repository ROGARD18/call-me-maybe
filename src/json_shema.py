from typing import Any


class JSONSchema:

    type: str
    properties: dict[str, "JSONSchema"] | None = None
    requiered: list[str] | None = None
    items: "JSONSchema" | None = None
    enum: list[Any] | None = None


def schema_to_valid_prefixes(current: str) -> set[str]:
    current = current.strip()

    # if shema.type == "object":
    #     if not current:
    #         return {"{"}
    #     if current == "{":
    #         if shema.requiered:
    #             return {'"'}
    #         return {'"', "}"}
    # elif shema.type == "string":
    if not current:
        return {'"'}
    if current.startswith('"') and not current.endswith('"'):
        return set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    if (
        current.startswith('"')
        and current.endswith('"')
        and len(current) > 1
    ):
        return (set())

    return (set())
