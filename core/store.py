"""The Chunk vector store: a ChromaDB collection created with cosine distance
and asserted at startup (ADR-0003) -- switching later means recreating the
collection and re-embedding the corpus, so a mismatch fails loudly rather
than silently scoring distances in the wrong space.
"""

from dataclasses import dataclass

import chromadb

from config import CHROMA_PATH, CHUNK_COLLECTION_NAME

COSINE_SPACE = "cosine"


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    ordinal: int
    locator: str
    domain: str
    source_type: str
    title: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    locator: str
    domain: str
    source_type: str
    title: str
    text: str
    distance: float


class VectorStore:
    def __init__(self, path=CHROMA_PATH, collection_name: str = CHUNK_COLLECTION_NAME, client=None):
        self._client = client or chromadb.PersistentClient(path=str(path))
        self.collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": COSINE_SPACE}
        )
        actual_space = (self.collection.metadata or {}).get("hnsw:space")
        if actual_space != COSINE_SPACE:
            raise RuntimeError(
                f"Chunk collection {collection_name!r} must use cosine distance "
                f"(hnsw:space='cosine'); found {actual_space!r}. Recreate the "
                "collection to change this -- switching in place silently "
                "leaves distances scored in the old space."
            )

    def upsert_chunks(self, records: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        self.collection.upsert(
            ids=[r.chunk_id for r in records],
            embeddings=embeddings,
            documents=[r.text for r in records],
            metadatas=[
                {
                    "doc_id": r.doc_id,
                    "ordinal": r.ordinal,
                    "locator": r.locator,
                    "domain": r.domain,
                    "source_type": r.source_type,
                    "title": r.title,
                }
                for r in records
            ],
        )

    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        result = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=meta["doc_id"],
                locator=meta["locator"],
                domain=meta["domain"],
                source_type=meta["source_type"],
                title=meta["title"],
                text=text,
                distance=distance,
            )
            for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances)
        ]
