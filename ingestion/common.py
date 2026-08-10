"""Ingests a folder of Markdown notes -- every one beneath it, at any depth
outside a dot-prefixed directory -- into the Document registry and Chunk store:
content-hash skip (ADR-0001), structured chunking, batch embedding.

The folder is where the run starts, not what a Document is named by: identity
is the path below `corpus_root` (config.py's CORPUS_ROOT, which says why).
Domain, source type and language are the Document's own rather than part of
that identity, so a run that changes one of them re-ingests the file it names.

Re-running is the routine case, not the exception. An unchanged Document costs
nothing (no chunking, no embedding); a changed one has its old generation of
Chunks replaced wholesale; a Document the walk no longer finds where it used to
be is retired, Chunks and registry row together (ADR-0005), which is what keeps
a moved note one Document rather than two; a Document whose text cannot be
extracted is recorded as failed and the run carries on. Every run returns an
IngestReport, so a garbled file is a logged incident rather than a silent gap
in the corpus, and a retirement is named rather than a silent deletion.
"""

import datetime
from dataclasses import dataclass
from pathlib import Path

from config import CORPUS_ROOT, MAX_SECTION_TOKENS, MIN_SECTION_TOKENS
from core.chunking import chunk_markdown
from core.embedder import Embedder
from core.registry import Document, Registry, content_hash, derive_doc_id
from core.store import ChunkRecord, VectorStore


class ExtractionError(Exception):
    """A Document's text could not be recovered in a usable form.

    Raised per Document and caught by the run, never propagated out of
    `ingest_folder` -- one unreadable file must not cost the corpus every file
    after it in the walk.
    """


class OutsideCorpusRoot(Exception):
    """Ingestion was pointed at a folder that does not lie beneath the corpus root.

    Raised for the whole run and before anything is written, unlike
    ExtractionError: it is not one file that cannot be named but every file the
    run would touch. Propagated rather than collected into the report, since a
    report of nothing ingested is the same shape as an empty folder.
    """


class FolderNotFound(Exception):
    """Ingestion was pointed at a folder that is not there to walk.

    Raised for the whole run, like OutsideCorpusRoot and unlike
    ExtractionError, because the alternative is catastrophic rather than
    merely wrong: a walk of a folder that does not exist yields no note, which
    is byte-for-byte the same evidence as a folder whose notes have all been
    deleted. Retirement would take that as proof and empty everything beneath
    it -- so an unmounted vault, a mistyped path or a corpus root that has not
    finished syncing would delete the corpus and report an ordinary success
    (ADR-0005). A run has to be able to tell "looked and found nothing" from
    "could not look".
    """


@dataclass(frozen=True)
class IngestFailure:
    doc_id: str
    source_path: str
    reason: str

    def warning(self) -> str:
        return f"WARNING: skipped {self.source_path} — {self.reason}"


@dataclass(frozen=True)
class RetiredDocument:
    doc_id: str
    source_path: str

    def notice(self) -> str:
        return f"Retired {self.source_path} — no longer in the corpus at that path"


@dataclass(frozen=True)
class IngestReport:
    ingested: list[str]
    skipped: list[str]
    failed: list[IngestFailure]
    retired: list[RetiredDocument]

    def summary(self) -> str:
        return (
            f"Ingested {len(self.ingested)}, "
            f"skipped {len(self.skipped)} unchanged, "
            f"retired {len(self.retired)}, "
            f"failed {len(self.failed)}."
        )


def _derive_title(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip()


def _extract_text(path: Path) -> str:
    # read_text, not read_bytes().decode(): it translates line endings, so a
    # note's content_hash survives a CRLF-vs-LF checkout instead of the whole
    # corpus re-embedding once on a different machine.
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ExtractionError(f"not valid utf-8 ({error.reason})") from error
    except OSError as error:
        # Locked by an editor, mid-sync, deleted since the glob -- per-file
        # accidents, and no reason to cost the folder every note after this one.
        raise ExtractionError(f"could not be read ({error.strerror or error})") from error


def _notes_beneath(folder: Path) -> list[tuple[str, Path]]:
    """Every Markdown note beneath `folder`, as (path relative to it, path).

    The whole tree, not one flat level: a student's notes live in subfolders,
    and one that is never read is not a failure anyone sees -- it is a hole in
    the corpus that surfaces months later as an abstention on a question they
    know they wrote notes about.

    Dot-prefixed directories are the one exclusion. A note vault keeps
    `.trash/` and `.obsidian/` beside the notes, and a note the student deleted
    cited back to them as current is worse than one never ingested. Only
    directories: a note named `.draft.md` is still a note.

    Sorted on the relative path rather than on Path, so the report reads in the
    same order everywhere -- Path comparison case-folds on Windows and does not
    on POSIX.
    """
    notes = (
        (path.relative_to(folder).as_posix(), path) for path in folder.rglob("*.md")
    )
    return sorted(
        (relative, path)
        for relative, path in notes
        if not any(segment.startswith(".") for segment in relative.split("/")[:-1])
    )


def _folder_beneath_root(folder: Path, corpus_root: Path) -> Path:
    """`folder`'s own path below the corpus root, or a loud failure if outside it.

    Both sides are resolved before comparing, so a relative invocation, a `..`
    segment or a symlinked temp directory still lands against the same root.
    Only what lies below the root is returned -- nothing absolute, nothing that
    differs between two machines holding the same corpus, reaches a doc_id.
    """
    try:
        return folder.resolve().relative_to(corpus_root.resolve())
    except ValueError as error:
        raise OutsideCorpusRoot(
            f"{folder} is not beneath the corpus root {corpus_root}. A Document is "
            "identified by its path below that root, so ingesting from outside it "
            "would key this run's Documents on paths the next run cannot derive, "
            "and collide with any same-named note already in the corpus -- whose "
            "Chunks would then be replaced. Move the notes beneath the root, or "
            "point CORPUS_ROOT in config.py at the root they already share."
        ) from error


def _walked_prefix(beneath_root: Path) -> str:
    """The source_path prefix covering exactly what a walk of this folder sees.

    Empty for the corpus root itself, which covers the whole corpus; otherwise
    the folder's own path below the root, with a trailing slash so that `os/`
    covers `os/deadlock.md` and not the sibling folder `os-2024/deadlock.md`.
    """
    return "".join(f"{segment}/" for segment in beneath_root.parts)


def _retire_absent_documents(
    walked_prefix: str, found: set[str], registry: Registry, store: VectorStore
) -> list[RetiredDocument]:
    """Retire the Documents beneath `walked_prefix` that the walk did not find.

    A Document is retired on the evidence that the walk covering its
    source_path did not reach it: within that prefix the walk is exhaustive, so
    absence from it means the note is not there any more -- deleted, moved
    elsewhere, or moved into a dot-prefixed directory, which is what deleting a
    note in a vault does. Outside the prefix the run has seen nothing and so
    says nothing, which is why ingesting one subfolder cannot empty the corpus.

    The cost of that scoping: a note moved out of the walked folder entirely is
    ingested at its new path while the old Document survives until a run that
    covers both -- pointing ingestion at the corpus root, or at their common
    ancestor, retires it.

    Chunks go first and the registry row second, so that a run interrupted
    between the two leaves a Document whose Chunks are gone -- which the next
    run over the same folder retires again, retirement being idempotent. The
    other order leaves the Chunks with no registry row naming them, and since
    retirement reads the registry to find its candidates, no later run could
    ever reach them: exactly the undetectable stale generation ADR-0001 exists
    to prevent (ADR-0005).
    """
    absent = sorted(
        (
            document
            for document in registry.list()
            if document.source_path.startswith(walked_prefix)
            and document.doc_id not in found
        ),
        key=lambda document: document.source_path,
    )

    for document in absent:
        store.delete_document_chunks(document.doc_id)
        registry.delete(document.doc_id)

    return [
        RetiredDocument(doc_id=document.doc_id, source_path=document.source_path)
        for document in absent
    ]


def ingest_folder(
    folder: Path,
    domain: str,
    source_type: str,
    registry: Registry,
    store: VectorStore,
    embedder: Embedder,
    language: str = "zh-tw",
    min_tokens: int = MIN_SECTION_TOKENS,
    max_tokens: int = MAX_SECTION_TOKENS,
    corpus_root: Path = CORPUS_ROOT,
) -> IngestReport:
    folder = Path(folder)
    if not folder.is_dir():
        raise FolderNotFound(
            f"{folder} is not a folder that can be walked. Ingestion retires the "
            "Documents beneath the folder it walks that the walk does not find, "
            "and a folder that is not there yields the same empty walk as one "
            "whose notes are all gone -- so continuing would retire every "
            "Document beneath it. Check the path, or that the corpus has finished "
            "syncing."
        )
    # Before the walk, so a folder outside the root costs no reading and writes
    # nothing -- the run fails whole rather than half-ingested under ids that
    # mean nothing to the next one.
    beneath_root = _folder_beneath_root(folder, corpus_root)
    ingested: list[str] = []
    skipped: list[str] = []
    failed: list[IngestFailure] = []
    found: set[str] = set()

    for relative, path in _notes_beneath(folder):
        # The path below the corpus root, not below `folder`: identity belongs
        # to the file, not to how the run was invoked. Two same-named notes in
        # different folders -- last year's and this year's -- therefore stay two
        # Documents rather than one silently replacing the other, and pointing
        # ingestion at a subfolder finds the same doc_ids as pointing it at the
        # whole corpus.
        #
        # `domain` is deliberately not part of it, though it used to be. A
        # source_path that resolves back to the file cannot carry a segment the
        # file is not under, and a Document has exactly one Domain anyway -- so
        # the Domain is a field on it, and changing it re-ingests rather than
        # forking a second Document off the same file.
        source_path = (beneath_root / relative).as_posix()
        doc_id = derive_doc_id(source_path)
        # Recorded before the outcome is known, so that a Document whose text
        # this run could not extract counts as found rather than as gone. It
        # keeps whatever generation of Chunks it last had (see below), and
        # retiring it on top of that would let one unreadable file delete the
        # note it is the only remaining copy of.
        found.add(doc_id)

        try:
            was_ingested = _ingest_document(
                path=path,
                source_path=source_path,
                doc_id=doc_id,
                domain=domain,
                source_type=source_type,
                registry=registry,
                store=store,
                embedder=embedder,
                language=language,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
            )
            (ingested if was_ingested else skipped).append(doc_id)
        except ExtractionError as error:
            # Failure leaves both stores untouched: the Document keeps whatever
            # generation it last had (possibly none) and no hash is recorded,
            # so the next run re-reads it rather than comparing against a
            # Document with no Chunks behind it. The cost is that a Document
            # emptied at source keeps serving its old Chunks -- accepted,
            # because clearing on failure would let one broken extractor empty
            # the corpus in a single run, and the report names it every run.
            failed.append(
                IngestFailure(doc_id=doc_id, source_path=source_path, reason=str(error))
            )

    # After the walk, not before it: what retires a Document is that this run
    # looked where it should have been and did not find it.
    retired = _retire_absent_documents(
        _walked_prefix(beneath_root), found, registry, store
    )
    return IngestReport(
        ingested=ingested, skipped=skipped, failed=failed, retired=retired
    )


def _ingest_document(
    *,
    path: Path,
    source_path: str,
    doc_id: str,
    domain: str,
    source_type: str,
    registry: Registry,
    store: VectorStore,
    embedder: Embedder,
    language: str,
    min_tokens: int,
    max_tokens: int,
) -> bool:
    """Ingest one Document; True if it was (re-)ingested, False if unchanged.

    Raises ExtractionError if its text is unusable, having written nothing.
    """
    raw = _extract_text(path)
    # Built before the skip check rather than after it, so what is compared
    # against the registry is exactly what would be written to it -- the
    # Document cannot be skipped on one set of fields and stored with another.
    document = Document(
        doc_id=doc_id,
        title=_derive_title(path),
        domain=domain,
        source_type=source_type,
        source_path=source_path,
        language=language,
        content_hash=content_hash(raw),
        ingested_at=datetime.date.today().isoformat(),
    )

    if registry.unchanged(document):
        return False

    drafts = chunk_markdown(raw, min_tokens=min_tokens, max_tokens=max_tokens)
    if not drafts:
        # Structurally readable but empty of prose -- a heading with no body,
        # or whitespace. Recording it as ingested would put a Document in the
        # registry that contributes no Chunk to any answer.
        raise ExtractionError("yielded no chunks (no extractable text)")

    records = [
        ChunkRecord(
            chunk_id=f"{doc_id}:{draft.ordinal:03d}",
            doc_id=doc_id,
            ordinal=draft.ordinal,
            locator=draft.locator,
            domain=domain,
            source_type=source_type,
            title=document.title,
            text=draft.text,
        )
        for draft in drafts
    ]

    embeddings = embedder.embed([r.text for r in records])
    # Before the registry row, so a Document is never recorded as ingested
    # ahead of the Chunks it is recorded for. Replacing them is also what
    # carries a changed Domain onto the Chunks retrieval filters on.
    store.replace_document_chunks(doc_id, records, embeddings)

    registry.upsert(document)
    return True
