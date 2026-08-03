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
