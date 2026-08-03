from core.embedder import FakeEmbedder
from core.registry import Registry, derive_doc_id
from core.store import VectorStore
from ingestion.common import ingest_folder


class CountingEmbedder:
    """FakeEmbedder that records how much embedding work it was asked to do."""

    def __init__(self):
        self._inner = FakeEmbedder()
        self.calls = 0
        self.texts_embedded = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.texts_embedded += len(texts)
        return self._inner.embed(texts)


def _ingest(folder, domain, registry, store, embedder, **overrides):
    kwargs = dict(
        folder=folder,
        domain=domain,
        source_type="note",
        registry=registry,
        store=store,
        embedder=embedder,
        min_tokens=1,
        max_tokens=10,
    )
    kwargs.update(overrides)
    return ingest_folder(**kwargs)


def _registry_and_store(tmp_path):
    return (
        Registry(tmp_path / "documents.sqlite"),
        VectorStore(path=tmp_path / "chroma"),
    )


def test_same_filename_in_different_domains_gets_distinct_doc_ids(tmp_path):
    dsa_dir = tmp_path / "notes" / "dsa"
    os_dir = tmp_path / "notes" / "os"
    dsa_dir.mkdir(parents=True)
    os_dir.mkdir(parents=True)
    (dsa_dir / "overview.md").write_text(
        "# DSA Overview\n\nBinary search trees keep keys ordered for fast lookup.",
        encoding="utf-8",
    )
    (os_dir / "overview.md").write_text(
        "# OS Overview\n\nProcesses are scheduled onto the CPU by the kernel.",
        encoding="utf-8",
    )

    registry, store = _registry_and_store(tmp_path)
    embedder = FakeEmbedder()

    dsa_report = _ingest(dsa_dir, "dsa", registry, store, embedder, max_tokens=100)
    os_report = _ingest(os_dir, "os", registry, store, embedder, max_tokens=100)

    [dsa_doc_id] = dsa_report.ingested
    [os_doc_id] = os_report.ingested
    assert dsa_doc_id != os_doc_id

    docs = {doc.doc_id: doc for doc in registry.list()}
    assert docs[dsa_doc_id].domain == "dsa"
    assert docs[os_doc_id].domain == "os"
    assert docs[dsa_doc_id].content_hash != docs[os_doc_id].content_hash


def test_notes_in_subfolders_are_ingested_at_every_depth(tmp_path):
    # A student's notes live in subfolders. A note that is never read is not a
    # failure anyone sees -- it is a hole in the corpus that surfaces months
    # later as an abstention on a question they know they wrote notes about, so
    # the walk goes all the way down rather than one level.
    folder = tmp_path / "os"
    (folder / "排程" / "深入").mkdir(parents=True)
    (folder / "top.md").write_text(
        "# Top\n\nA note at the folder's own level.", encoding="utf-8"
    )
    (folder / "排程" / "nested.md").write_text(
        "# Scheduling\n\nA note one subfolder down.", encoding="utf-8"
    )
    (folder / "排程" / "深入" / "deeper.md").write_text(
        "# Preemption\n\nA note two subfolders down.", encoding="utf-8"
    )
    registry, store = _registry_and_store(tmp_path)

    report = _ingest(folder, "os", registry, store, FakeEmbedder())

    assert set(report.ingested) == {
        derive_doc_id("os/top.md"),
        derive_doc_id("os/排程/nested.md"),
        derive_doc_id("os/排程/深入/deeper.md"),
    }


def test_same_filename_in_different_subfolders_gets_distinct_doc_ids(tmp_path):
    # Same Domain, same filename, different subfolder. Keying on the filename
    # alone would leave one note silently replacing the other.
    folder = tmp_path / "os"
    (folder / "排程").mkdir(parents=True)
    (folder / "網路").mkdir()
    (folder / "排程" / "overview.md").write_text(
        "# Scheduling\n\nRound robin gives each process a time quantum.",
        encoding="utf-8",
    )
    (folder / "網路" / "overview.md").write_text(
        "# Networking\n\nThe transport layer carries segments end to end.",
        encoding="utf-8",
    )
    registry, store = _registry_and_store(tmp_path)

    report = _ingest(folder, "os", registry, store, FakeEmbedder(), max_tokens=100)

    assert len(set(report.ingested)) == 2
    docs = {doc.source_path: doc for doc in registry.list()}
    # The subfolder segments reach the registry, so a citation can be traced
    # back to the file on disk rather than to whichever "overview.md" won.
    assert set(docs) == {"os/排程/overview.md", "os/網路/overview.md"}
    assert len({doc.content_hash for doc in docs.values()}) == 2


def test_the_report_counts_documents_found_in_subfolders(tmp_path):
    # Ingested, skipped and failed alike: a nested corpus that reports
    # "Ingested 0, skipped 0, failed 0" tells the student nothing is wrong.
    folder = tmp_path / "os"
    (folder / "排程").mkdir(parents=True)
    (folder / "排程" / "kept.md").write_text(
        "# Scheduling\n\nUnchanged between runs.", encoding="utf-8"
    )
    (folder / "排程" / "garbled.md").write_bytes(b"\xff\xfe not decodable")
    registry, store = _registry_and_store(tmp_path)
    embedder = FakeEmbedder()

    _ingest(folder, "os", registry, store, embedder)
    (folder / "排程" / "深入").mkdir()
    (folder / "排程" / "深入" / "added.md").write_text(
        "# Preemption\n\nA sibling note one level further down.", encoding="utf-8"
    )
    report = _ingest(folder, "os", registry, store, embedder)

    assert report.ingested == [derive_doc_id("os/排程/深入/added.md")]
    assert report.skipped == [derive_doc_id("os/排程/kept.md")]
    [failure] = report.failed
    assert failure.source_path == "os/排程/garbled.md"
    assert report.summary() == "Ingested 1, skipped 1 unchanged, failed 1."


def test_notes_under_a_dot_prefixed_directory_are_left_alone(tmp_path):
    # An Obsidian vault carries `.trash/` and `.obsidian/` beside the notes. A
    # walk that goes all the way down would ingest a note the student deleted
    # and cite it back to them as if it were still in the corpus.
    folder = tmp_path / "os"
    (folder / ".trash").mkdir(parents=True)
    (folder / "kept.md").write_text(
        "# Scheduling\n\nStill in the corpus.", encoding="utf-8"
    )
    (folder / ".trash" / "deleted.md").write_text(
        "# Scheduling\n\nDeleted by the student.", encoding="utf-8"
    )
    registry, store = _registry_and_store(tmp_path)

    report = _ingest(folder, "os", registry, store, FakeEmbedder())

    assert report.ingested == [derive_doc_id("os/kept.md")]
    # Not a failure either: a deleted note is not an incident to report, it is
    # not part of the corpus at all.
    assert report.failed == []
    assert [doc.source_path for doc in registry.list()] == ["os/kept.md"]


def test_reingesting_an_unchanged_document_does_no_embedding_work(tmp_path):
    folder = tmp_path / "dsa"
    folder.mkdir()
    (folder / "bst.md").write_text(
        "# BST\n\nKeys smaller than the node go left, larger go right.",
        encoding="utf-8",
    )
    registry, store = _registry_and_store(tmp_path)
    embedder = CountingEmbedder()

    _ingest(folder, "dsa", registry, store, embedder)
    work_after_first_run = embedder.texts_embedded
    second = _ingest(folder, "dsa", registry, store, embedder)

    assert work_after_first_run > 0
    assert embedder.texts_embedded == work_after_first_run
    assert second.ingested == []
    assert second.skipped == [derive_doc_id("dsa/bst.md")]


def test_a_changed_document_leaves_exactly_one_generation_of_chunks(tmp_path):
    folder = tmp_path / "dsa"
    folder.mkdir()
    note = folder / "bst.md"
    note.write_text(
        "# BST\n\n" + " ".join(f"word{i}" for i in range(60)),
        encoding="utf-8",
    )
    registry, store = _registry_and_store(tmp_path)
    embedder = FakeEmbedder()
    doc_id = derive_doc_id("dsa/bst.md")

    _ingest(folder, "dsa", registry, store, embedder)
    first_generation = set(store.collection.get(where={"doc_id": doc_id})["ids"])
    note.write_text("# BST\n\nA much shorter rewrite of the note.", encoding="utf-8")
    report = _ingest(folder, "dsa", registry, store, embedder)

    second_generation = set(store.collection.get(where={"doc_id": doc_id})["ids"])
    assert report.ingested == [doc_id]
    assert len(first_generation) > len(second_generation)
    # Not merely "no duplicates": every ID from the old chunking is gone, so no
    # Chunk of the previous generation can still be retrieved.
    assert second_generation == {f"{doc_id}:000"}


def test_a_changed_document_leaves_no_stale_chunk_text_behind(tmp_path):
    folder = tmp_path / "dsa"
    folder.mkdir()
    note = folder / "bst.md"
    note.write_text("# BST\n\nThe original claim about rotations.", encoding="utf-8")
    registry, store = _registry_and_store(tmp_path)
    embedder = FakeEmbedder()
    doc_id = derive_doc_id("dsa/bst.md")

    _ingest(folder, "dsa", registry, store, embedder)
    note.write_text("# BST\n\nThe corrected claim about rotations.", encoding="utf-8")
    _ingest(folder, "dsa", registry, store, embedder)

    documents = store.collection.get(where={"doc_id": doc_id})["documents"]
    assert not any("original" in text for text in documents)
    assert any("corrected" in text for text in documents)


def test_a_document_that_fails_extraction_is_reported_and_the_run_continues(tmp_path):
    folder = tmp_path / "dsa"
    folder.mkdir()
    (folder / "garbled.md").write_bytes(b"# BST\n\n\xff\xfe not decodable\xff")
    (folder / "usable.md").write_text("# BST\n\nA note that reads fine.", encoding="utf-8")
    registry, store = _registry_and_store(tmp_path)
    embedder = FakeEmbedder()

    report = _ingest(folder, "dsa", registry, store, embedder)

    assert report.ingested == [derive_doc_id("dsa/usable.md")]
    [failure] = report.failed
    assert failure.source_path == "dsa/garbled.md"
    assert "utf-8" in failure.reason.lower()
    # A Document that never extracted must not enter the registry, or the next
    # run would compare hashes against a Document that has no Chunks.
    assert [doc.doc_id for doc in registry.list()] == [derive_doc_id("dsa/usable.md")]


def test_a_document_that_cannot_be_opened_is_reported_and_the_run_continues(tmp_path):
    # A directory matching *.md is the portable stand-in for the real cases --
    # a note locked by an editor, mid-sync, or deleted since the glob. All
    # reach _extract_text as OSError, and none should cost the folder the
    # notes that come after it.
    folder = tmp_path / "dsa"
    folder.mkdir()
    (folder / "unopenable.md").mkdir()
    (folder / "usable.md").write_text("# BST\n\nA note that reads fine.", encoding="utf-8")
    registry, store = _registry_and_store(tmp_path)
    embedder = FakeEmbedder()

    report = _ingest(folder, "dsa", registry, store, embedder)

    assert report.ingested == [derive_doc_id("dsa/usable.md")]
    [failure] = report.failed
    assert failure.source_path == "dsa/unopenable.md"
    assert "could not be read" in failure.reason


def test_a_document_that_yields_no_chunks_is_reported_as_failed(tmp_path):
    folder = tmp_path / "dsa"
    folder.mkdir()
    (folder / "empty.md").write_text("# BST\n\n   \n", encoding="utf-8")
    registry, store = _registry_and_store(tmp_path)
    embedder = CountingEmbedder()

    report = _ingest(folder, "dsa", registry, store, embedder)

    assert report.ingested == []
    [failure] = report.failed
    assert failure.doc_id == derive_doc_id("dsa/empty.md")
    assert "chunk" in failure.reason.lower()
    assert embedder.calls == 0


def test_a_line_ending_change_alone_does_not_re_embed_a_document(tmp_path):
    # A CRLF-vs-LF checkout must not read as changed content: it would re-embed
    # the whole corpus on a different machine and make the skip worthless.
    folder = tmp_path / "dsa"
    folder.mkdir()
    note = folder / "bst.md"
    note.write_bytes(b"# BST\r\n\r\nKeys smaller go left, larger go right.\r\n")
    registry, store = _registry_and_store(tmp_path)
    embedder = CountingEmbedder()

    _ingest(folder, "dsa", registry, store, embedder)
    work_after_first_run = embedder.texts_embedded
    note.write_bytes(b"# BST\n\nKeys smaller go left, larger go right.\n")
    second = _ingest(folder, "dsa", registry, store, embedder)

    assert second.skipped == [derive_doc_id("dsa/bst.md")]
    assert embedder.texts_embedded == work_after_first_run


def test_the_report_counts_ingested_skipped_and_failed(tmp_path):
    folder = tmp_path / "dsa"
    folder.mkdir()
    (folder / "kept.md").write_text("# BST\n\nUnchanged between runs.", encoding="utf-8")
    (folder / "garbled.md").write_bytes(b"\xff\xfe not decodable")
    registry, store = _registry_and_store(tmp_path)
    embedder = FakeEmbedder()

    _ingest(folder, "dsa", registry, store, embedder)
    (folder / "added.md").write_text("# Heaps\n\nA sibling note.", encoding="utf-8")
    report = _ingest(folder, "dsa", registry, store, embedder)

    assert (len(report.ingested), len(report.skipped), len(report.failed)) == (1, 1, 1)
    assert report.summary() == "Ingested 1, skipped 1 unchanged, failed 1."
