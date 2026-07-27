import chromadb
import pytest

from core.store import VectorStore, ChunkRecord


def make_client(tmp_path):
    return chromadb.PersistentClient(path=str(tmp_path / "chroma"))


def test_creates_a_collection_with_cosine_distance(tmp_path):
    store = VectorStore(client=make_client(tmp_path))

    assert store.collection.metadata["hnsw:space"] == "cosine"


def test_fails_loudly_when_an_existing_collection_does_not_use_cosine(tmp_path):
    client = make_client(tmp_path)
    client.get_or_create_collection("chunks")  # pre-existing, default L2 space

    with pytest.raises(RuntimeError, match="cosine"):
        VectorStore(client=client)


def test_upsert_then_query_round_trips_chunk_metadata(tmp_path):
    store = VectorStore(client=make_client(tmp_path))
    record = ChunkRecord(
        chunk_id="doc1:000",
        doc_id="doc1",
        ordinal=0,
        locator="BST › Insertion",
        domain="dsa",
        source_type="note",
        title="Red-Black Trees",
        text="Insert by walking left or right from the root.",
    )

    store.upsert_chunks([record], embeddings=[[1.0, 0.0, 0.0]])
    result = store.query(embedding=[1.0, 0.0, 0.0], top_k=1)

    assert result[0].chunk_id == "doc1:000"
    assert result[0].doc_id == "doc1"
    assert result[0].locator == "BST › Insertion"
    assert result[0].title == "Red-Black Trees"
    assert result[0].source_type == "note"
    assert result[0].text == "Insert by walking left or right from the root."
    assert result[0].distance == pytest.approx(0.0, abs=1e-6)
