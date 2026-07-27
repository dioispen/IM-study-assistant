"""Retrieval: embed the question, query the Chunk store, return Evidence."""

from core.embedder import Embedder
from core.store import RetrievedChunk, VectorStore


def retrieve(
    question: str,
    embedder: Embedder,
    store: VectorStore,
    top_k: int,
) -> list[RetrievedChunk]:
    [query_embedding] = embedder.embed([question])
    return store.query(embedding=query_embedding, top_k=top_k)
