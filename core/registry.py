"""The Document registry (ADR-0001): stable identity for Documents, kept
alongside — not inside — the Chunk vector store."""

import hashlib
import sqlite3
from dataclasses import astuple, dataclass, fields
from pathlib import Path


def derive_doc_id(source_path: str) -> str:
    """A stable doc_id derived from a Document's source path, never random."""
    return hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:12]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    domain: str
    source_type: str
    source_path: str
    language: str
    content_hash: str
    ingested_at: str


_COLUMNS = [f.name for f in fields(Document)]


class Registry:
    """SQLite-backed Document registry keyed by doc_id."""

    def __init__(self, db_path: Path | str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS documents (
                {", ".join(f"{col} TEXT NOT NULL" for col in _COLUMNS)},
                PRIMARY KEY (doc_id)
            )
            """
        )
        self._conn.commit()

    def upsert(self, document: Document) -> None:
        placeholders = ", ".join("?" for _ in _COLUMNS)
        self._conn.execute(
            f"INSERT OR REPLACE INTO documents ({', '.join(_COLUMNS)}) "
            f"VALUES ({placeholders})",
            astuple(document),
        )
        self._conn.commit()

    def get(self, doc_id: str) -> Document | None:
        row = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        return Document(*row) if row else None

    def delete(self, doc_id: str) -> None:
        """Remove a Document's row, for a Document that has left the corpus.

        Deleted after the Document's Chunks, never before them: the row is what
        a later run finds the Document by, so while it is there the retirement
        can always be finished. Dropping it first would leave Chunks that
        nothing in the registry names and no run could find again -- the
        undetectable stale generation ADR-0001 exists to prevent (ADR-0005).
        """
        self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self._conn.commit()

    def unchanged(self, document: Document) -> bool:
        """True if `document` is already recorded exactly as this run would write it.

        Not the content hash alone. Domain, source type and language are the
        Document's own, not the file's: a re-ingest that changes one of them
        without touching the file would otherwise be reported as an ordinary
        skip while quietly discarding what the run asked for -- and domain and
        source type reach every Chunk, so the discarded value is also what
        retrieval filters on.

        `ingested_at` is deliberately not compared: a run that writes nothing
        new is not an ingestion, and comparing it would make every Document
        changed once a day.
        """
        existing = self.get(document.doc_id)
        if existing is None:
            return False
        return (
            existing.content_hash == document.content_hash
            and existing.domain == document.domain
            and existing.source_type == document.source_type
            and existing.language == document.language
        )

    def list(self) -> list[Document]:
        rows = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM documents"
        ).fetchall()
        return [Document(*row) for row in rows]

    def source_types(self) -> list[str]:
        """Every Source type the corpus actually holds, in alphabetical order.

        Source type is an open set (CONTEXT.md): there is no configured list of
        them anywhere, so the values only exist here, on the Documents that were
        ingested under them. A menu built over this becomes a real choice the
        moment a second Source type is ingested, with no code change -- which is
        the whole reason a surface asks the registry rather than a constant.

        Ordered rather than left to whatever the rows come back in, so a menu
        does not reshuffle itself between runs over an unchanged corpus.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT source_type FROM documents ORDER BY source_type"
        ).fetchall()
        return [source_type for (source_type,) in rows]

    def documents_per_domain(self) -> dict[str, int]:
        """How many Documents each Domain holds, for the Domains that hold any.

        Documents rather than Chunks, and so read here rather than off the
        vector store: the registry is the authority on what Documents exist
        (ADR-0001), while the store holds one particular chunking of them -- a
        count taken there would report a long note as a large corpus.

        Silent about a configured Domain nothing was ever ingested under. This
        is a report of the corpus rather than of the configuration, and a
        Domain listed at zero would be a Domain the reader has to check twice.
        """
        rows = self._conn.execute(
            "SELECT domain, COUNT(*) FROM documents GROUP BY domain ORDER BY domain"
        ).fetchall()
        return {domain: count for domain, count in rows}
