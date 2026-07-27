from core.registry import Document, Registry, content_hash, derive_doc_id


def test_derive_doc_id_is_stable_for_the_same_source_path():
    assert derive_doc_id("notes/dsa/rbtree.md") == derive_doc_id("notes/dsa/rbtree.md")


def test_derive_doc_id_differs_for_different_source_paths():
    assert derive_doc_id("notes/dsa/rbtree.md") != derive_doc_id("notes/dsa/avl.md")


def test_content_hash_changes_when_text_changes():
    assert content_hash("hello") != content_hash("hello world")


def make_document(doc_id="abc123", content_hash_value="hash1"):
    return Document(
        doc_id=doc_id,
        title="Red-Black Trees",
        domain="dsa",
        source_type="note",
        source_path="notes/dsa/rbtree.md",
        language="zh-tw",
        content_hash=content_hash_value,
        ingested_at="2026-07-27",
    )


def test_upsert_then_get_round_trips_a_document(tmp_path):
    registry = Registry(tmp_path / "documents.sqlite")
    doc = make_document()

    registry.upsert(doc)

    assert registry.get(doc.doc_id) == doc


def test_get_returns_none_for_unknown_doc_id(tmp_path):
    registry = Registry(tmp_path / "documents.sqlite")

    assert registry.get("nope") is None


def test_upsert_replaces_existing_document_with_the_same_doc_id(tmp_path):
    registry = Registry(tmp_path / "documents.sqlite")
    registry.upsert(make_document(content_hash_value="hash1"))

    registry.upsert(make_document(content_hash_value="hash2"))

    assert registry.get("abc123").content_hash == "hash2"
    assert len(registry.list()) == 1


def test_unchanged_is_true_only_when_hash_matches_the_stored_document(tmp_path):
    registry = Registry(tmp_path / "documents.sqlite")
    registry.upsert(make_document(content_hash_value="hash1"))

    assert registry.unchanged("abc123", "hash1") is True
    assert registry.unchanged("abc123", "hash2") is False
    assert registry.unchanged("unknown-doc", "hash1") is False


def test_list_returns_all_documents(tmp_path):
    registry = Registry(tmp_path / "documents.sqlite")
    registry.upsert(make_document(doc_id="doc1"))
    registry.upsert(make_document(doc_id="doc2"))

    doc_ids = {doc.doc_id for doc in registry.list()}

    assert doc_ids == {"doc1", "doc2"}
