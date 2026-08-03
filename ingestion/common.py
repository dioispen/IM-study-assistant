"""Ingests a folder of Markdown notes -- every one beneath it, at any depth
outside a dot-prefixed directory -- into the Document registry and Chunk store:
content-hash skip (ADR-0001), structured chunking, batch embedding.

The folder is where the run starts, not what a Document is named by: identity
is the path below `corpus_root` (config.py's CORPUS_ROOT, which says why).
Domain, source type and language are the Document's own rather than part of
that identity, so a run that changes one of them re-ingests the file it names.

Re-running is the routine case, not the exception. An unchanged Document costs
nothing (no chunking, no embedding); a changed one has its old generation of
Chunks replaced wholesale; a Document whose text cannot be extracted is
recorded as failed and the run carries on. Every run returns an IngestReport,
so a garbled file is a logged incident rather than a silent gap in the corpus.
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


@dataclass(frozen=True)
class IngestFailure:
    doc_id: str
    source_path: str
    reason: str

    def warning(self) -> str:
        return f"WARNING: skipped {self.source_path} — {self.reason}"


@dataclass(frozen=True)
class IngestReport:
    ingested: list[str]
    skipped: list[str]
    failed: list[IngestFailure]

    def summary(self) -> str:
        return (
            f"Ingested {len(self.ingested)}, "
            f"skipped {len(self.skipped)} unchanged, "
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
    # Before the walk, so a folder outside the root costs no reading and writes
    # nothing -- the run fails whole rather than half-ingested under ids that
    # mean nothing to the next one.
    beneath_root = _folder_beneath_root(folder, corpus_root)
    ingested: list[str] = []
    skipped: list[str] = []
    failed: list[IngestFailure] = []

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

    return IngestReport(ingested=ingested, skipped=skipped, failed=failed)


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
