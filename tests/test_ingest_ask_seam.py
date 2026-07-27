"""Seam tests for the walking skeleton: ingest a fixture corpus of Markdown
notes, then ask questions against it. Fully offline (FakeEmbedder,
FakeGenerator) and asserts only on retrieved doc_ids and Locators -- never on
generated text (ADR-0002).
"""

from pathlib import Path

from core.ask import ask
from core.embedder import FakeEmbedder
from core.generator import FakeGenerator
from core.registry import Registry, derive_doc_id
from core.store import VectorStore
from ingestion.common import ingest_folder

FIXTURES = Path(__file__).parent / "fixtures" / "notes"
MIN_TOKENS = 10
MAX_TOKENS = 45

BST_ID = derive_doc_id("dsa/binary_search_tree.md")
SCHEDULING_ID = derive_doc_id("os/process_scheduling.md")
DEADLOCK_ID = derive_doc_id("os/deadlock.md")


def _ingest_corpus(tmp_path):
    registry = Registry(tmp_path / "documents.sqlite")
    store = VectorStore(path=tmp_path / "chroma")
    embedder = FakeEmbedder()

    dsa_report = ingest_folder(
        folder=FIXTURES / "dsa",
        domain="dsa",
        source_type="note",
        registry=registry,
        store=store,
        embedder=embedder,
        min_tokens=MIN_TOKENS,
        max_tokens=MAX_TOKENS,
    )
    os_report = ingest_folder(
        folder=FIXTURES / "os",
        domain="os",
        source_type="note",
        registry=registry,
        store=store,
        embedder=embedder,
        min_tokens=MIN_TOKENS,
        max_tokens=MAX_TOKENS,
    )
    return registry, store, embedder, dsa_report, os_report


def test_ingest_populates_the_registry_with_every_fixture_document(tmp_path):
    registry, store, embedder, dsa_report, os_report = _ingest_corpus(tmp_path)

    doc_ids = {doc.doc_id for doc in registry.list()}
    assert doc_ids == {BST_ID, SCHEDULING_ID, DEADLOCK_ID}
    assert set(dsa_report.ingested) == {BST_ID}
    assert set(os_report.ingested) == {SCHEDULING_ID, DEADLOCK_ID}


def test_chunk_ids_are_doc_id_colon_ordinal(tmp_path):
    _registry, store, _embedder, _dsa, _os = _ingest_corpus(tmp_path)

    result = store.collection.get(where={"doc_id": BST_ID})

    assert set(result["ids"]) == {f"{BST_ID}:000", f"{BST_ID}:001"}


def test_reingesting_an_unchanged_folder_skips_every_document(tmp_path):
    registry = Registry(tmp_path / "documents.sqlite")
    store = VectorStore(path=tmp_path / "chroma")
    embedder = FakeEmbedder()
    kwargs = dict(
        folder=FIXTURES / "dsa",
        domain="dsa",
        source_type="note",
        registry=registry,
        store=store,
        embedder=embedder,
        min_tokens=MIN_TOKENS,
        max_tokens=MAX_TOKENS,
    )

    first = ingest_folder(**kwargs)
    second = ingest_folder(**kwargs)

    assert first.ingested == [BST_ID]
    assert first.skipped == []
    assert second.ingested == []
    assert second.skipped == [BST_ID]


def test_oversized_section_becomes_multiple_chunks_sharing_one_locator(tmp_path):
    _registry, store, _embedder, _dsa, _os = _ingest_corpus(tmp_path)

    result = store.collection.get(where={"doc_id": DEADLOCK_ID})

    assert len(result["ids"]) > 1
    assert {m["locator"] for m in result["metadatas"]} == {"Deadlock › Conditions"}
    assert sorted(result["ids"]) == [f"{DEADLOCK_ID}:000", f"{DEADLOCK_ID}:001"]


def test_no_chunk_straddles_a_heading(tmp_path):
    _registry, store, _embedder, _dsa, _os = _ingest_corpus(tmp_path)

    result = store.collection.get(where={"doc_id": BST_ID})
    text_by_locator = dict(
        zip((m["locator"] for m in result["metadatas"]), result["documents"])
    )

    assert "successor" not in text_by_locator["Binary Search Tree › Insertion"]
    assert "empty spot" not in text_by_locator["Binary Search Tree › Deletion"]


def test_asking_a_question_retrieves_the_document_its_vocabulary_matches(tmp_path):
    _registry, store, embedder, _dsa, _os = _ingest_corpus(tmp_path)
    generator = FakeGenerator()

    answer = ask(
        "What is round robin scheduling and how does the time quantum work?",
        embedder,
        store,
        generator,
        top_k=1,
    )

    assert len(answer.evidence) == 1
    top = answer.evidence[0]
    assert top.doc_id == SCHEDULING_ID
    assert top.locator == "Process Scheduling › Round Robin"
    assert top.source_type == "note"
    assert top.domain == "os"


def test_asking_a_different_question_retrieves_a_different_document(tmp_path):
    _registry, store, embedder, _dsa, _os = _ingest_corpus(tmp_path)
    generator = FakeGenerator()

    answer = ask(
        "What are the four conditions required for deadlock to occur?",
        embedder,
        store,
        generator,
        top_k=1,
    )

    assert answer.evidence[0].doc_id == DEADLOCK_ID


def test_ask_never_asserts_on_generated_text_only_on_evidence(tmp_path):
    _registry, store, embedder, _dsa, _os = _ingest_corpus(tmp_path)
    generator = FakeGenerator()

    answer = ask("What is round robin?", embedder, store, generator, top_k=2)

    assert isinstance(answer.text, str) and answer.text
    assert all(isinstance(chunk.doc_id, str) for chunk in answer.evidence)

