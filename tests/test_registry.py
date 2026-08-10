from core.registry import Document, Registry, content_hash, derive_doc_id


def test_derive_doc_id_is_stable_for_the_same_source_path():
    assert derive_doc_id("notes/dsa/rbtree.md") == derive_doc_id("notes/dsa/rbtree.md")


def test_derive_doc_id_differs_for_different_source_paths():
    assert derive_doc_id("notes/dsa/rbtree.md") != derive_doc_id("notes/dsa/avl.md")


def test_content_hash_changes_when_text_changes():
    assert content_hash("hello") != content_hash("hello world")


def make_document(doc_id="abc123", content_hash_value="hash1", **overrides):
    fields = dict(
        doc_id=doc_id,
        title="Red-Black Trees",
        domain="dsa",
        source_type="note",
        source_path="notes/dsa/rbtree.md",
        language="zh-tw",
        content_hash=content_hash_value,
        ingested_at="2026-07-27",
    )
    fields.update(overrides)
    return Document(**fields)


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


def test_unchanged_is_true_only_when_the_stored_document_matches(tmp_path):
    registry = Registry(tmp_path / "documents.sqlite")
    registry.upsert(make_document(content_hash_value="hash1"))

    assert registry.unchanged(make_document(content_hash_value="hash1")) is True
    assert registry.unchanged(make_document(content_hash_value="hash2")) is False
    assert registry.unchanged(make_document(doc_id="unknown-doc")) is False


def test_unchanged_is_false_when_only_the_documents_own_fields_differ(tmp_path):
    # The file is untouched, so the content hash matches -- but domain, source
    # type and language are the Document's, not the file's, and the run asked
    # for different ones. Answering "unchanged" would report an ordinary skip
    # and discard them; domain and source_type reach every Chunk, so the
    # discarded value is also what retrieval filters on.
    registry = Registry(tmp_path / "documents.sqlite")
    registry.upsert(make_document())

    assert registry.unchanged(make_document(domain="os")) is False
    assert registry.unchanged(make_document(source_type="wiki")) is False
    assert registry.unchanged(make_document(language="en")) is False
    # ingested_at is not one of them: comparing it would make every Document
    # changed once a day, and a run that writes nothing is not an ingestion.
    assert registry.unchanged(make_document(ingested_at="2026-08-03")) is True


def test_list_returns_all_documents(tmp_path):
    registry = Registry(tmp_path / "documents.sqlite")
    registry.upsert(make_document(doc_id="doc1"))
    registry.upsert(make_document(doc_id="doc2"))

    doc_ids = {doc.doc_id for doc in registry.list()}

    assert doc_ids == {"doc1", "doc2"}


def test_delete_removes_only_the_named_document(tmp_path):
    # What retiring a Document that has left the corpus rests on: the row goes,
    # and the rest of the corpus does not go with it.
    registry = Registry(tmp_path / "documents.sqlite")
    registry.upsert(make_document(doc_id="doc1"))
    registry.upsert(make_document(doc_id="doc2"))

    registry.delete("doc1")

    assert registry.get("doc1") is None
    assert {doc.doc_id for doc in registry.list()} == {"doc2"}


def test_deleting_an_unknown_doc_id_is_not_an_error(tmp_path):
    # Retirement reads the registry and then writes it; a doc_id that is
    # already gone is the run having nothing to do, not an incident.
    registry = Registry(tmp_path / "documents.sqlite")

    registry.delete("never-existed")

    assert registry.list() == []
