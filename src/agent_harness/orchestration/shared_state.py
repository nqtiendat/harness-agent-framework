"""Shared state collaboration substrate."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SharedStateItem(BaseModel):
    key: str
    value: str
    author: str
    metadata: dict = Field(default_factory=dict)


class SharedStateSpace:
    def __init__(self) -> None:
        self._items: dict[str, SharedStateItem] = {}

    def write(self, item: SharedStateItem) -> None:
        self._items[item.key] = item

    def read(self, key: str) -> SharedStateItem | None:
        return self._items.get(key)

    def list_items(self) -> list[SharedStateItem]:
        return list(self._items.values())

