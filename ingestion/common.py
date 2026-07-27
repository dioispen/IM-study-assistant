"""Ingests a folder of Markdown notes into the Document registry and Chunk
store: content-hash skip (ADR-0001), structured chunking, batch embedding.
"""

import datetime
from dataclasses import dataclass
from pathlib import Path

from config import MAX_SECTION_TOKENS, MIN_SECTION_TOKENS
from core.chunking import chunk_markdown
from core.embedder import Embedder
from core.registry import Document, Registry, content_hash, derive_doc_id
from core.store import ChunkRecord, VectorStore


def _derive_title(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip()


@dataclass(frozen=True)
class IngestReport:
    ingested: list[str]
    skipped: list[str]


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

    for path in sorted(folder.glob("*.md")):
        # Prefixed with domain so that same-named files ingested under
        # different domains don't collide on doc_id (a plain path relative to
        # `folder` would drop the domain segment entirely).
        source_path = f"{domain}/{path.relative_to(folder).as_posix()}"
        raw = path.read_text(encoding="utf-8")
        doc_id = derive_doc_id(source_path)
        hash_ = content_hash(raw)

        if registry.unchanged(doc_id, hash_):
            skipped.append(doc_id)
            continue

        title = _derive_title(path)
        drafts = chunk_markdown(raw, min_tokens=min_tokens, max_tokens=max_tokens)
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
        store.upsert_chunks(records, embeddings)

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
        ingested.append(doc_id)

    return IngestReport(ingested=ingested, skipped=skipped)
