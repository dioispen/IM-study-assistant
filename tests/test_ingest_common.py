import pytest

from core.embedder import FakeEmbedder
from core.registry import Registry, derive_doc_id
from core.store import VectorStore
from ingestion.common import FolderNotFound, OutsideCorpusRoot, ingest_folder
from tests.docx_fixture import BODY, resave_untouched, write_docx
from tests.pdf_fixture import truncate, write_pdf


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
        # The folder's parent, so a corpus laid out one Domain folder per
        # subdirectory gives each note the `<domain>/<path>` source_path it had
        # before the root existed. Tests about the root itself pass their own.
        corpus_root=folder.parent,
    )
    kwargs.update(overrides)
    return ingest_folder(**kwargs)


def _registry_and_store(tmp_path):
    return (
        Registry(tmp_path / "documents.sqlite"),
        VectorStore(path=tmp_path / "chroma"),
    )


def test_notes_in_different_domain_folders_reach_the_registry_under_their_domain(tmp_path):
    # What separates these two is their path below the corpus root, not the
    # `domain` argument -- since the root arrived, a Domain is a field on a
    # Document rather than part of its identity (see the re-home test below).
    # What this still pins is that each Document lands in the Domain its own
    # run named, so a corpus grown one Domain folder at a time stays filterable.
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


# The corpus root. Everything above keys a doc_id on the path below the folder
# ingestion was handed, which makes the id a property of how the run was
# invoked rather than of the file. These pin it to the path below one
# configured root instead.


def test_two_same_named_notes_under_different_roots_stay_two_documents(tmp_path):
    # The collision the corpus root exists to prevent: last year's notes and
    # this year's, kept in separate folders and ingested under one Domain.
    # Relative to the folder of the moment both derive `os/deadlock.md`, so the
    # second run replaces the first's registry row and -- through
    # replace_document_chunks, correctly enforcing ADR-0001's one-generation
    # invariant -- deletes every Chunk the first had. Nothing warns: the report
    # calls it an ordinary ingest and a year of notes is gone from the corpus.
    corpus = tmp_path / "corpus"
    for year in ("notes-2024", "notes-2025"):
        (corpus / year / "os").mkdir(parents=True)
        (corpus / year / "os" / "deadlock.md").write_text(
            f"# Deadlock\n\nThe {year} note on circular wait.", encoding="utf-8"
        )
    registry, store = _registry_and_store(tmp_path)
    embedder = FakeEmbedder()

    first = _ingest(
        corpus / "notes-2024" / "os", "os", registry, store, embedder, corpus_root=corpus
    )
    second = _ingest(
        corpus / "notes-2025" / "os", "os", registry, store, embedder, corpus_root=corpus
    )

    [first_id], [second_id] = first.ingested, second.ingested
    assert first_id != second_id
    # Two live sets of Chunks, not merely two registry rows: the assertion that
    # says the second year did not take the first year's Chunks with it.
    assert store.collection.get(where={"doc_id": first_id})["ids"]
    assert store.collection.get(where={"doc_id": second_id})["ids"]
    assert {doc.source_path for doc in registry.list()} == {
        "notes-2024/os/deadlock.md",
        "notes-2025/os/deadlock.md",
    }


def test_a_doc_id_is_the_same_whichever_folder_beneath_the_root_is_ingested(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "os" / "排程").mkdir(parents=True)
    (corpus / "os" / "排程" / "deadlock.md").write_text(
        "# Deadlock\n\nCircular wait among processes holding resources.",
        encoding="utf-8",
    )
    registry, store = _registry_and_store(tmp_path)
    embedder = FakeEmbedder()

    whole = _ingest(corpus, "os", registry, store, embedder, corpus_root=corpus)
    subfolder = _ingest(
        corpus / "os" / "排程", "os", registry, store, embedder, corpus_root=corpus
    )

    assert whole.ingested == [derive_doc_id("os/排程/deadlock.md")]
    # Pointed one folder deeper, the same file on disk is the same Document --
    # so the second run skips it rather than ingesting a second copy of it.
    assert subfolder.ingested == []
    assert subfolder.skipped == whole.ingested


def test_the_same_corpus_under_a_different_absolute_root_derives_the_same_doc_ids(tmp_path):
    # ADR-0001's stability promise holds across machines and checkouts, so no
    # absolute or machine-specific segment may enter the derivation -- the same
    # class of concern as reading through read_text so that a CRLF-vs-LF
    # checkout does not re-embed the corpus.
    ingested = []
    for checkout in ("machine-a", "machine-b"):
        corpus = tmp_path / checkout / "corpus"
        (corpus / "os").mkdir(parents=True)
        (corpus / "os" / "deadlock.md").write_text(
            "# Deadlock\n\nCircular wait among processes.", encoding="utf-8"
        )
        registry, store = _registry_and_store(tmp_path / checkout)
        report = _ingest(
            corpus / "os", "os", registry, store, FakeEmbedder(), corpus_root=corpus
        )
        ingested.append(report.ingested)

    assert ingested[0] == ingested[1] == [derive_doc_id("os/deadlock.md")]


def test_the_registrys_source_path_resolves_back_to_the_file_on_disk(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "os" / "排程").mkdir(parents=True)
    (corpus / "os" / "排程" / "deadlock.md").write_text(
        "# Deadlock\n\nCircular wait among processes.", encoding="utf-8"
    )
    registry, store = _registry_and_store(tmp_path)

    _ingest(corpus / "os", "os", registry, store, FakeEmbedder(), corpus_root=corpus)

    [document] = registry.list()
    assert document.source_path == "os/排程/deadlock.md"
    # What a citation is followed back through: root plus source_path, and no
    # segment of it depends on which folder the run happened to be pointed at.
    assert (corpus / document.source_path).is_file()


# Retirement. A path-derived doc_id makes a moved note a new Document, so
# without these the note at the old path lingers in the registry with every
# Chunk it had still retrievable -- two copies of the same prose to draw
# Evidence from, one of them citing a source_path that no longer resolves.


def test_a_note_that_moves_within_the_corpus_stays_one_document(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "os").mkdir(parents=True)
    note = corpus / "os" / "deadlock.md"
    note.write_text(
        "# Deadlock\n\nCircular wait among processes holding resources.",
        encoding="utf-8",
    )
    registry, store = _registry_and_store(tmp_path)
    embedder = FakeEmbedder()

    _ingest(corpus, "os", registry, store, embedder, corpus_root=corpus)
    (corpus / "os" / "排程").mkdir()
    note.rename(corpus / "os" / "排程" / "deadlock.md")
    report = _ingest(corpus, "os", registry, store, embedder, corpus_root=corpus)

    assert report.ingested == [derive_doc_id("os/排程/deadlock.md")]
    # One Document, not the moved one beside the ghost of where it used to be.
    assert [doc.source_path for doc in registry.list()] == ["os/排程/deadlock.md"]
    assert [retired.source_path for retired in report.retired] == ["os/deadlock.md"]


def test_the_chunks_of_a_moved_notes_old_path_are_no_longer_retrievable(tmp_path):
    # The registry row is the bookkeeping; this is the failure a reader meets.
    # Left behind, the old generation is a second copy of the same prose for
    # retrieval to draw Evidence from, citing a source_path that no longer
    # resolves to a file -- the traceability failure the corpus root was
    # introduced to close, arriving by a different route.
    corpus = tmp_path / "corpus"
    (corpus / "os").mkdir(parents=True)
    note = corpus / "os" / "deadlock.md"
    note.write_text(
        "# Deadlock\n\nCircular wait among processes holding resources.",
        encoding="utf-8",
    )
    registry, store = _registry_and_store(tmp_path)
    embedder = FakeEmbedder()

    _ingest(corpus, "os", registry, store, embedder, corpus_root=corpus)
    (corpus / "os" / "排程").mkdir()
    note.rename(corpus / "os" / "排程" / "deadlock.md")
    _ingest(corpus, "os", registry, store, embedder, corpus_root=corpus)

    [embedding] = embedder.embed(["Circular wait among processes holding resources."])
    retrieved = store.query(embedding, top_k=5)
    assert retrieved  # the moved Document is still answerable, exactly once
    assert {chunk.doc_id for chunk in retrieved} == {
        derive_doc_id("os/排程/deadlock.md")
    }


def test_ingesting_one_subfolder_retires_no_document_outside_it(tmp_path):
    # The rule that keeps retirement from being catastrophic. A run pointed at
    # one folder sees almost nothing of the corpus, so "absent from this walk"
    # is only evidence of absence for what the walk covered -- read the other
    # way, the first run anyone makes on a single folder empties the corpus.
    corpus = tmp_path / "corpus"
    (corpus / "os" / "排程").mkdir(parents=True)
    (corpus / "dsa").mkdir()
    (corpus / "os" / "排程" / "deadlock.md").write_text(
        "# Deadlock\n\nCircular wait among processes.", encoding="utf-8"
    )
    (corpus / "dsa" / "bst.md").write_text(
        "# BST\n\nKeys smaller than the node go left.", encoding="utf-8"
    )
    registry, store = _registry_and_store(tmp_path)
    embedder = FakeEmbedder()

    _ingest(corpus, "os", registry, store, embedder, corpus_root=corpus)
    report = _ingest(
        corpus / "os" / "排程", "os", registry, store, embedder, corpus_root=corpus
    )

    assert report.retired == []
    assert {doc.source_path for doc in registry.list()} == {
        "os/排程/deadlock.md",
        "dsa/bst.md",
    }
    # Not merely the registry: the Chunks a question would draw Evidence from.
    assert store.collection.get(where={"doc_id": derive_doc_id("dsa/bst.md")})["ids"]


def test_a_note_moved_into_a_dot_prefixed_directory_is_retired(tmp_path):
    # What deleting a note in a vault actually does: it moves the file into
    # `.trash/`, which the walk already declines to read. Without retirement the
    # note stays fully answerable and cites a path the student cannot open.
    corpus = tmp_path / "corpus"
    (corpus / "os").mkdir(parents=True)
    (corpus / ".trash").mkdir()
    note = corpus / "os" / "deadlock.md"
    note.write_text("# Deadlock\n\nCircular wait among processes.", encoding="utf-8")
    registry, store = _registry_and_store(tmp_path)
    embedder = FakeEmbedder()

    _ingest(corpus, "os", registry, store, embedder, corpus_root=corpus)
    note.rename(corpus / ".trash" / "deadlock.md")
    report = _ingest(corpus, "os", registry, store, embedder, corpus_root=corpus)

    assert [retired.source_path for retired in report.retired] == ["os/deadlock.md"]
    assert registry.list() == []
    assert store.collection.get()["ids"] == []


def test_a_document_that_fails_extraction_is_not_retired(tmp_path):
    # Found and unreadable is not gone. Retiring on a failed extraction would
    # let one locked or garbled file delete the Chunks that are, until it can
    # be read again, the only copy of that note the corpus has -- and the run
    # already declines to clear them for the same reason.
    corpus = tmp_path / "corpus"
    (corpus / "os").mkdir(parents=True)
    note = corpus / "os" / "deadlock.md"
    note.write_text("# Deadlock\n\nCircular wait among processes.", encoding="utf-8")
    registry, store = _registry_and_store(tmp_path)
    embedder = FakeEmbedder()

    _ingest(corpus, "os", registry, store, embedder, corpus_root=corpus)
    note.write_bytes(b"# Deadlock\n\n\xff\xfe not decodable")
    report = _ingest(corpus, "os", registry, store, embedder, corpus_root=corpus)

    assert report.retired == []
    assert [failure.source_path for failure in report.failed] == ["os/deadlock.md"]
    assert [doc.source_path for doc in registry.list()] == ["os/deadlock.md"]
    assert store.collection.get(where={"doc_id": derive_doc_id("os/deadlock.md")})["ids"]


def test_a_folder_that_does_not_exist_fails_loudly_and_retires_nothing(tmp_path):
    # The way retirement could empty the corpus. A walk of a folder that is not
    # there yields no note, which is indistinguishable from a folder whose
    # notes are all gone -- so pointed at an unmounted vault, a mistyped path or
    # a corpus root that has not synced, one run would retire every Document
    # beneath it and report it as an ordinary success. The run has to be able
    # to tell "looked and found nothing" from "could not look".
    corpus = tmp_path / "corpus"
    (corpus / "os").mkdir(parents=True)
    (corpus / "os" / "deadlock.md").write_text(
        "# Deadlock\n\nCircular wait among processes.", encoding="utf-8"
    )
    registry, store = _registry_and_store(tmp_path)
    embedder = FakeEmbedder()

    _ingest(corpus, "os", registry, store, embedder, corpus_root=corpus)

    with pytest.raises(FolderNotFound) as raised:
        _ingest(
            corpus / "not-synced", "os", registry, store, embedder, corpus_root=corpus
        )

    assert str(corpus / "not-synced") in str(raised.value)
    assert [doc.source_path for doc in registry.list()] == ["os/deadlock.md"]
    assert store.collection.get(where={"doc_id": derive_doc_id("os/deadlock.md")})["ids"]


def test_a_sibling_folder_sharing_a_name_prefix_is_not_retired(tmp_path):
    # `os/` covers `os/deadlock.md` and must not reach into `os-2024/`, which a
    # prefix match without the separator would swallow whole -- the run would
    # retire a year of notes it never looked at.
    corpus = tmp_path / "corpus"
    (corpus / "os").mkdir(parents=True)
    (corpus / "os-2024").mkdir()
    (corpus / "os" / "deadlock.md").write_text(
        "# Deadlock\n\nCircular wait among processes.", encoding="utf-8"
    )
    (corpus / "os-2024" / "deadlock.md").write_text(
        "# Deadlock\n\nLast year's note on circular wait.", encoding="utf-8"
    )
    registry, store = _registry_and_store(tmp_path)
    embedder = FakeEmbedder()

    _ingest(corpus, "os", registry, store, embedder, corpus_root=corpus)
    report = _ingest(
        corpus / "os", "os", registry, store, embedder, corpus_root=corpus
    )

    assert report.retired == []
    assert {doc.source_path for doc in registry.list()} == {
        "os/deadlock.md",
        "os-2024/deadlock.md",
    }


def test_the_report_names_every_document_it_retired(tmp_path):
    # Retirement deletes Chunks, so it is the one outcome of a run that costs
    # the student something they cannot get back by re-running. Counted in the
    # summary even at zero, for the reason failures are, and named by
    # source_path rather than doc_id -- a retired Document is no longer in the
    # registry to look its id up in.
    corpus = tmp_path / "corpus"
    (corpus / "os").mkdir(parents=True)
    for name in ("deadlock.md", "排程.md"):
        (corpus / "os" / name).write_text(
            f"# {name}\n\nA note about {name}.", encoding="utf-8"
        )
    registry, store = _registry_and_store(tmp_path)
    embedder = FakeEmbedder()

    first = _ingest(corpus, "os", registry, store, embedder, corpus_root=corpus)
    (corpus / "os" / "deadlock.md").unlink()
    (corpus / "os" / "排程.md").unlink()
    report = _ingest(corpus, "os", registry, store, embedder, corpus_root=corpus)

    assert first.retired == []
    assert first.summary() == "Ingested 2, skipped 0 unchanged, retired 0, failed 0."
    # Sorted on source_path: registry.list() has no order of its own, and a
    # report of what was deleted has to read the same way twice.
    assert [retired.source_path for retired in report.retired] == [
        "os/deadlock.md",
        "os/排程.md",
    ]
    assert report.summary() == "Ingested 0, skipped 0 unchanged, retired 2, failed 0."
    assert report.retired[0].notice() == (
        "Retired os/deadlock.md — no longer in the corpus at that path"
    )


def test_reingesting_one_note_under_a_different_domain_rehomes_it(tmp_path):
    # Domain left the doc_id derivation when the corpus root arrived, so one
    # file is one Document -- and CONTEXT.md gives a Document exactly one
    # Domain. The second run therefore has to move it rather than report the
    # ordinary skip its unchanged content would otherwise earn, which would
    # discard the Domain the run named and leave every Chunk filtered under the
    # old one.
    corpus = tmp_path / "corpus"
    (corpus / "notes").mkdir(parents=True)
    (corpus / "notes" / "overview.md").write_text(
        "# Overview\n\nA note filed under the wrong Domain the first time.",
        encoding="utf-8",
    )
    registry, store = _registry_and_store(tmp_path)
    embedder = FakeEmbedder()

    first = _ingest(corpus / "notes", "dsa", registry, store, embedder, corpus_root=corpus)
    second = _ingest(corpus / "notes", "os", registry, store, embedder, corpus_root=corpus)

    [doc_id] = first.ingested
    assert second.ingested == [doc_id]
    assert second.skipped == []
    [document] = registry.list()
    assert document.domain == "os"
    # The Chunks moved with it: domain is what retrieval filters on, so a
    # registry row saying "os" over Chunks still saying "dsa" is no re-homing.
    chunks = store.collection.get(where={"doc_id": doc_id})
    assert {m["domain"] for m in chunks["metadatas"]} == {"os"}


def test_a_folder_outside_the_corpus_root_fails_loudly(tmp_path):
    # The alternative is a doc_id derived from a path that means nothing to the
    # next run and collides with whatever note already sits at it.
    corpus = tmp_path / "corpus"
    (corpus / "os").mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere" / "os"
    elsewhere.mkdir(parents=True)
    (elsewhere / "deadlock.md").write_text(
        "# Deadlock\n\nCircular wait among processes.", encoding="utf-8"
    )
    registry, store = _registry_and_store(tmp_path)

    with pytest.raises(OutsideCorpusRoot) as raised:
        _ingest(elsewhere, "os", registry, store, FakeEmbedder(), corpus_root=corpus)

    # Both paths named, because the reader has to decide which of the two to move.
    assert str(elsewhere) in str(raised.value)
    assert str(corpus) in str(raised.value)
    # Raised for the whole run before anything is written, not per Document.
    assert registry.list() == []


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
    assert report.summary() == "Ingested 1, skipped 1 unchanged, retired 0, failed 1."


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
    # reach extract_document as OSError, and none should cost the folder the
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
    assert report.summary() == "Ingested 1, skipped 1 unchanged, retired 0, failed 1."


# Mixed formats. Everything above is Markdown, which is how "the walk looks for
# *.md" survived a green suite: a corpus owner pointing ingestion at a folder
# of Word notes was told "Ingested 0" and nothing was wrong. These pin that the
# walk routes by format, that one bad file in any format costs the run only
# itself, and that every format reaches the registry from a single run (#9,
# and PDF with #7).

ZH_LAYERING_NOTE = (
    "# OSI 參考模型\n\n## 分層架構\n\n應用層之下依序是傳輸層、網路層與連結層。"
)
ZH_LAYERING_BODY = "應用層之下依序是傳輸層、網路層與連結層。"

# A paper, for the format that has pages and no headings. Short enough to sit
# on one page of the fixture writer and long enough to be prose rather than a
# label -- the windowing itself is unit-tested in test_chunking.py, so what
# these need of it is only that it happened.
ZH_PAPER_PAGE_ONE = (
    "本文比較數種分層協定堆疊的實作成本，並以延遲與吞吐量兩項指標評估其取捨。"
)
ZH_PAPER_PAGE_TWO = (
    "實驗在同一組硬體上重複三十次，結果顯示分層帶來的額外複製成本可被批次處理攤平。"
)


def test_a_mixed_folder_ingests_both_formats_in_one_run(tmp_path):
    folder = tmp_path / "network"
    folder.mkdir()
    (folder / "osi_model.md").write_text(ZH_LAYERING_NOTE, encoding="utf-8")
    write_docx(
        folder / "tcp_handshake.docx",
        [
            ("Heading 1", "傳輸控制協定"),
            ("Heading 2", "三向交握"),
            (BODY, "用戶端先送出 SYN 封包，伺服端回覆 SYN-ACK，用戶端再送出 ACK。"),
        ],
    )
    registry, store = _registry_and_store(tmp_path)

    report = _ingest(folder, "network", registry, store, FakeEmbedder(), max_tokens=200)

    assert sorted(report.ingested) == sorted(
        [
            derive_doc_id("network/osi_model.md"),
            derive_doc_id("network/tcp_handshake.docx"),
        ]
    )
    assert {doc.source_path for doc in registry.list()} == {
        "network/osi_model.md",
        "network/tcp_handshake.docx",
    }


def test_a_docx_note_in_a_subfolder_is_ingested_at_its_own_depth(tmp_path):
    # The nesting rules and the format routing are separate walks' worth of
    # logic, and a docx reader bolted onto a flat glob would pass every test
    # above while a Word note two folders down went unread.
    folder = tmp_path / "mis"
    write_docx(
        folder / "流程管理" / "塑模" / "bpmn.docx",
        [
            ("Heading 1", "BPMN"),
            ("Heading 2", "閘道"),
            (BODY, "閘道標示流程在此分支或匯流，依條件決定後續路徑。"),
        ],
    )
    registry, store = _registry_and_store(tmp_path)

    report = _ingest(folder, "mis", registry, store, FakeEmbedder(), max_tokens=200)

    assert report.ingested == [derive_doc_id("mis/流程管理/塑模/bpmn.docx")]


def test_a_malformed_docx_is_reported_and_the_rest_of_the_run_proceeds(tmp_path):
    folder = tmp_path / "network"
    folder.mkdir()
    (folder / "broken.docx").write_bytes(b"not a Word package at all")
    (folder / "readable.md").write_text(ZH_LAYERING_NOTE, encoding="utf-8")
    registry, store = _registry_and_store(tmp_path)

    report = _ingest(folder, "network", registry, store, FakeEmbedder(), max_tokens=200)

    assert report.ingested == [derive_doc_id("network/readable.md")]
    [failure] = report.failed
    assert failure.source_path == "network/broken.docx"
    assert "broken.docx" in failure.warning()
    assert registry.get(failure.doc_id) is None


def test_a_word_lock_file_is_not_read_as_a_note(tmp_path):
    # Word leaves a `~$`-prefixed stub beside a note it has open, and that stub
    # is not a Word package. Walking it would put a failure in every run for as
    # long as the note is open -- a warning the corpus owner learns to ignore,
    # which is the one thing a per-file warning cannot afford to become.
    folder = tmp_path / "network"
    folder.mkdir()
    write_docx(
        folder / "osi_model.docx",
        [("Heading 1", "OSI 參考模型"), ("Heading 2", "分層架構"), (BODY, ZH_LAYERING_BODY)],
    )
    (folder / "~$osi_model.docx").write_bytes(b"\x00\x01 Word lock file")
    registry, store = _registry_and_store(tmp_path)

    report = _ingest(folder, "network", registry, store, FakeEmbedder(), max_tokens=200)

    assert report.failed == []
    assert report.ingested == [derive_doc_id("network/osi_model.docx")]


def test_a_file_of_no_readable_format_is_neither_ingested_nor_reported(tmp_path):
    # A vault holds images, spreadsheets and exports beside the notes. None of
    # them is a Document any reader can make sense of, and none of them is a
    # broken one either -- warning about each one every run would bury the
    # warnings that are about notes. PDF used to be on this list and is not any
    # more: it has a reader of its own now (#7), which is what the tests below
    # pin so this one cannot quietly reclaim it.
    folder = tmp_path / "network"
    folder.mkdir()
    (folder / "diagram.png").write_bytes(b"IMAGE_MAGIC_BYTES")
    (folder / "grades.xlsx").write_bytes(b"PK_A_SPREADSHEET")
    (folder / "osi_model.md").write_text(ZH_LAYERING_NOTE, encoding="utf-8")
    registry, store = _registry_and_store(tmp_path)

    report = _ingest(folder, "network", registry, store, FakeEmbedder(), max_tokens=200)

    assert report.ingested == [derive_doc_id("network/osi_model.md")]
    assert report.failed == []


def test_a_pdf_beside_the_notes_joins_the_walk(tmp_path):
    # One run over the folder the corpus owner actually keeps their material
    # in, with no flag naming a format: the walk routes each file by its own
    # suffix, so a paper filed beside the notes on the subject lands with them.
    folder = tmp_path / "network"
    folder.mkdir()
    (folder / "osi_model.md").write_text(ZH_LAYERING_NOTE, encoding="utf-8")
    write_pdf(folder / "layering_survey.pdf", [ZH_PAPER_PAGE_ONE, ZH_PAPER_PAGE_TWO])
    registry, store = _registry_and_store(tmp_path)

    report = _ingest(folder, "network", registry, store, FakeEmbedder(), max_tokens=200)

    assert sorted(report.ingested) == sorted(
        [
            derive_doc_id("network/osi_model.md"),
            derive_doc_id("network/layering_survey.pdf"),
        ]
    )
    assert report.failed == []


def test_a_broken_pdf_is_reported_and_costs_the_walk_nothing_else(tmp_path):
    # The run report contract from #4 over the new format: a truncated download
    # is one file named in the report, not a traceback that aborts the walk and
    # takes the retirement sweep with it.
    folder = tmp_path / "network"
    folder.mkdir()
    (folder / "osi_model.md").write_text(ZH_LAYERING_NOTE, encoding="utf-8")
    truncate(
        write_pdf(tmp_path / "intact.pdf", [ZH_PAPER_PAGE_ONE]),
        folder / "layering_survey.pdf",
    )
    registry, store = _registry_and_store(tmp_path)

    report = _ingest(folder, "network", registry, store, FakeEmbedder(), max_tokens=200)

    assert report.ingested == [derive_doc_id("network/osi_model.md")]
    [failure] = report.failed
    assert failure.source_path == "network/layering_survey.pdf"
    assert registry.get(failure.doc_id) is None


def test_a_scanned_pdf_is_reported_as_having_no_text_layer(tmp_path):
    # The failure that would otherwise arrive as "yielded no chunks", which
    # sends the corpus owner looking for an empty file rather than for OCR.
    folder = tmp_path / "network"
    folder.mkdir()
    write_pdf(folder / "scan.pdf", ["", ""])
    registry, store = _registry_and_store(tmp_path)

    report = _ingest(folder, "network", registry, store, FakeEmbedder(), max_tokens=200)

    [failure] = report.failed
    assert "text layer" in failure.reason
    assert report.ingested == []


def test_reingesting_an_unchanged_docx_note_does_no_embedding_work(tmp_path):
    folder = tmp_path / "network"
    write_docx(
        folder / "osi_model.docx",
        [("Heading 1", "OSI 參考模型"), ("Heading 2", "分層架構"), (BODY, ZH_LAYERING_BODY)],
    )
    registry, store = _registry_and_store(tmp_path)
    embedder = CountingEmbedder()

    _ingest(folder, "network", registry, store, embedder, max_tokens=200)
    work_after_first_run = embedder.texts_embedded
    second = _ingest(folder, "network", registry, store, embedder, max_tokens=200)

    assert second.skipped == [derive_doc_id("network/osi_model.docx")]
    assert embedder.texts_embedded == work_after_first_run


def test_resaving_a_docx_without_editing_it_does_not_re_embed_it(tmp_path):
    # The docx counterpart of the CRLF test above. Opening a note in Word and
    # closing it rewrites the package, and a hash over the file's bytes would
    # read that as a changed note -- re-embedding a corpus every time its owner
    # opened one of its notes to read it.
    folder = tmp_path / "network"
    note = write_docx(
        folder / "osi_model.docx",
        [("Heading 1", "OSI 參考模型"), ("Heading 2", "分層架構"), (BODY, ZH_LAYERING_BODY)],
    )
    registry, store = _registry_and_store(tmp_path)
    embedder = CountingEmbedder()

    _ingest(folder, "network", registry, store, embedder, max_tokens=200)
    work_after_first_run = embedder.texts_embedded
    resave_untouched(note, note)
    second = _ingest(folder, "network", registry, store, embedder, max_tokens=200)

    assert second.skipped == [derive_doc_id("network/osi_model.docx")]
    assert embedder.texts_embedded == work_after_first_run


def test_editing_a_docx_note_replaces_its_chunks(tmp_path):
    folder = tmp_path / "network"
    note = folder / "osi_model.docx"
    write_docx(
        note,
        [("Heading 1", "OSI 參考模型"), ("Heading 2", "分層架構"), (BODY, ZH_LAYERING_BODY)],
    )
    registry, store = _registry_and_store(tmp_path)
    doc_id = derive_doc_id("network/osi_model.docx")

    _ingest(folder, "network", registry, store, FakeEmbedder(), max_tokens=200)
    write_docx(
        note,
        [
            ("Heading 1", "OSI 參考模型"),
            ("Heading 2", "封裝"),
            (BODY, "資料往下傳遞時逐層加上標頭，往上則逐層拆解。"),
        ],
    )
    second = _ingest(folder, "network", registry, store, FakeEmbedder(), max_tokens=200)

    assert second.ingested == [doc_id]
    stored = store.collection.get(where={"doc_id": doc_id})
    assert {m["locator"] for m in stored["metadatas"]} == {"OSI 參考模型 › 封裝"}
    assert not any("連結層" in text for text in stored["documents"])


def test_a_docx_with_headings_and_no_prose_is_reported_as_failed(tmp_path):
    # An ideographic space is whitespace, so this note has headings and no
    # prose -- the shape the Markdown path already reports rather than
    # registering a Document that contributes no Chunk to any answer.
    folder = tmp_path / "network"
    write_docx(
        folder / "empty.docx",
        [("Heading 1", "網路"), ("Heading 2", "概述"), (BODY, "　　")],
    )
    registry, store = _registry_and_store(tmp_path)
    embedder = CountingEmbedder()

    report = _ingest(folder, "network", registry, store, embedder, max_tokens=200)

    assert report.ingested == []
    [failure] = report.failed
    assert failure.source_path == "network/empty.docx"
    assert "chunk" in failure.reason.lower()
    assert embedder.calls == 0
