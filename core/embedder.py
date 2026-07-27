"""The embedder interface (batch of texts in, vectors out) -- the pipeline's
one injection point between the real OpenAI model and an offline test double.
"""

import hashlib
import math
import re
from typing import Protocol

from core.openai_client import default_openai_client

_WORD_RE = re.compile(r"[\w']+")


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, model: str, client=None):
        self._client = client or default_openai_client()
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


class FakeEmbedder:
    """Deterministic, offline stand-in for tests.

    Embeds via the hashing trick over whitespace-split words: each word hashes
    to one of `dim` buckets and increments it, then the vector is
    L2-normalized. Similarity therefore tracks literal shared vocabulary
    between texts, which is what fixture corpora and questions are written
    to exploit -- no network, no model weights, same output every run.
    """

    def __init__(self, dim: int = 64):
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for word in _WORD_RE.findall(text.lower()):
            index = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16) % self._dim
            vector[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
