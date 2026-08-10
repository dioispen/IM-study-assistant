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


def make_record(chunk_id, doc_id="doc1", ordinal=0, text="Insert by walking left or right from the root."):
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        ordinal=ordinal,
        locator="BST › Insertion",
        domain="dsa",
        source_type="note",
        title="Red-Black Trees",
        text=text,
    )


def test_writing_then_querying_round_trips_chunk_metadata(tmp_path):
    store = VectorStore(client=make_client(tmp_path))
    record = make_record("doc1:000")

    store.replace_document_chunks("doc1", [record], embeddings=[[1.0, 0.0, 0.0]])
    result = store.query(embedding=[1.0, 0.0, 0.0], top_k=1)

    assert result[0].chunk_id == "doc1:000"
    assert result[0].doc_id == "doc1"
    assert result[0].locator == "BST › Insertion"
    assert result[0].title == "Red-Black Trees"
    assert result[0].source_type == "note"
    assert result[0].text == "Insert by walking left or right from the root."
    assert result[0].distance == pytest.approx(0.0, abs=1e-6)


def test_replacing_a_documents_chunks_drops_the_previous_generation(tmp_path):
    store = VectorStore(client=make_client(tmp_path))
    store.replace_document_chunks(
        "doc1",
        [make_record("doc1:000", ordinal=0), make_record("doc1:001", ordinal=1)],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )

    store.replace_document_chunks(
        "doc1", [make_record("doc1:000", ordinal=0)], embeddings=[[1.0, 0.0, 0.0]]
    )

    assert store.collection.get(where={"doc_id": "doc1"})["ids"] == ["doc1:000"]


def test_replacing_one_documents_chunks_leaves_other_documents_alone(tmp_path):
    store = VectorStore(client=make_client(tmp_path))
    store.replace_document_chunks(
        "doc1", [make_record("doc1:000")], embeddings=[[1.0, 0.0, 0.0]]
    )
    store.replace_document_chunks(
        "doc2", [make_record("doc2:000", doc_id="doc2")], embeddings=[[0.0, 1.0, 0.0]]
    )

    store.replace_document_chunks(
        "doc1", [make_record("doc1:000")], embeddings=[[1.0, 0.0, 0.0]]
    )

    assert store.collection.get(where={"doc_id": "doc2"})["ids"] == ["doc2:000"]


def test_records_from_another_document_are_rejected_rather_than_written(tmp_path):
    store = VectorStore(client=make_client(tmp_path))

    with pytest.raises(ValueError, match="doc2"):
        store.replace_document_chunks(
            "doc1",
            [make_record("doc1:000"), make_record("doc2:000", doc_id="doc2")],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )

    assert store.collection.get()["ids"] == []


def test_deleting_a_documents_chunks_leaves_it_holding_none(tmp_path):
    store = VectorStore(client=make_client(tmp_path))
    store.replace_document_chunks(
        "doc1",
        [make_record("doc1:000", ordinal=0), make_record("doc1:001", ordinal=1)],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )

    store.delete_document_chunks("doc1")

    assert store.collection.get(where={"doc_id": "doc1"})["ids"] == []


def test_deleting_one_documents_chunks_leaves_other_documents_alone(tmp_path):
    store = VectorStore(client=make_client(tmp_path))
    store.replace_document_chunks(
        "doc1", [make_record("doc1:000")], embeddings=[[1.0, 0.0, 0.0]]
    )
    store.replace_document_chunks(
        "doc2", [make_record("doc2:000", doc_id="doc2")], embeddings=[[0.0, 1.0, 0.0]]
    )

    store.delete_document_chunks("doc1")

    assert store.collection.get(where={"doc_id": "doc2"})["ids"] == ["doc2:000"]
