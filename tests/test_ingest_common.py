from pathlib import Path

from core.embedder import FakeEmbedder
from core.registry import Registry
from core.store import VectorStore
from ingestion.common import ingest_folder


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

    registry = Registry(tmp_path / "documents.sqlite")
    store = VectorStore(path=tmp_path / "chroma")
    embedder = FakeEmbedder()

    dsa_report = ingest_folder(
        folder=dsa_dir,
        domain="dsa",
        source_type="note",
        registry=registry,
        store=store,
        embedder=embedder,
        min_tokens=1,
        max_tokens=100,
    )
    os_report = ingest_folder(
        folder=os_dir,
        domain="os",
        source_type="note",
        registry=registry,
        store=store,
        embedder=embedder,
        min_tokens=1,
        max_tokens=100,
    )

    [dsa_doc_id] = dsa_report.ingested
    [os_doc_id] = os_report.ingested
    assert dsa_doc_id != os_doc_id

    docs = {doc.doc_id: doc for doc in registry.list()}
    assert docs[dsa_doc_id].domain == "dsa"
    assert docs[os_doc_id].domain == "os"
    assert docs[dsa_doc_id].content_hash != docs[os_doc_id].content_hash
