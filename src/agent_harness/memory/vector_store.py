"""In-process vector store with a pluggable embedding function.

The default embedder is a token-overlap scorer kept for backwards compatibility
with the original implementation; production deployments can inject a sentence
embedding callable (e.g. `sentence-transformers`) without changing the
`add` / `search` surface used by `LongTermMemoryService`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from pydantic import BaseModel, Field

Embedder = Callable[[str], Sequence[float]]


class VectorMemoryRecord(BaseModel):
    id: str
    text: str
    metadata: dict = Field(default_factory=dict)


def _hash_embed(text: str, dim: int = 128) -> list[float]:
    """Deterministic hashed bag-of-tokens fallback embedder.

    Pure-Python so the framework keeps zero ML dependencies; downstream users
    inject a real embedder when they need semantic recall.
    """
    vector = [0.0] * dim
    for token in text.lower().split():
        vector[hash(token) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vector))
    if norm:
        vector = [x / norm for x in vector]
    return vector


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


class VectorMemoryStore:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self._records: list[VectorMemoryRecord] = []
        self._embeddings: list[Sequence[float]] = []
        self._embedder: Embedder = embedder or _hash_embed

    def add(self, record: VectorMemoryRecord) -> None:
        self._records.append(record)
        self._embeddings.append(self._embedder(record.text))

    def search(self, query: str, limit: int = 5) -> list[VectorMemoryRecord]:
        if not self._records:
            return []
        query_vec = self._embedder(query)
        scored = [
            (_cosine(query_vec, emb), record)
            for emb, record in zip(self._embeddings, self._records)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for score, record in scored if score > 0][:limit]
