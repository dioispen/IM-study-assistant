"""Ingests a folder of Markdown notes -- every one beneath it, at any depth
outside a dot-prefixed directory -- into the Document registry and Chunk store:
content-hash skip (ADR-0001), structured chunking, batch embedding.

Re-running is the routine case, not the exception. An unchanged Document costs
nothing (no chunking, no embedding); a changed one has its old generation of
Chunks replaced wholesale; a Document whose text cannot be extracted is
recorded as failed and the run carries on. Every run returns an IngestReport,
so a garbled file is a logged incident rather than a silent gap in the corpus.
"""

import datetime
from dataclasses import dataclass
from pathlib import Path

from config import MAX_SECTION_TOKENS, MIN_SECTION_TOKENS
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
) -> IngestReport:
    folder = Path(folder)
    ingested: list[str] = []
    skipped: list[str] = []
    failed: list[IngestFailure] = []

    for relative, path in _notes_beneath(folder):
        # Prefixed with domain so that same-named files ingested under
        # different domains don't collide on doc_id (a plain path relative to
        # `folder` would drop the domain segment entirely). The subfolder
        # segments are carried too, so two same-named notes in different
        # subfolders of one corpus stay two Documents rather than one silently
        # replacing the other -- and a flat folder's paths, and so its doc_ids,
        # are exactly what they were before the walk went deeper.
        source_path = f"{domain}/{relative}"
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
    hash_ = content_hash(raw)

    if registry.unchanged(doc_id, hash_):
        return False

    title = _derive_title(path)
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
            title=title,
            text=draft.text,
        )
        for draft in drafts
    ]

    embeddings = embedder.embed([r.text for r in records])
    store.replace_document_chunks(doc_id, records, embeddings)

    registry.upsert(
        Document(
            doc_id=doc_id,
            title=title,
            domain=domain,
            source_type=source_type,
            source_path=source_path,
            language=language,
            content_hash=hash_,
            ingested_at=datetime.date.today().isoformat(),
        )
    )
    return True
