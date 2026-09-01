"""One RetrievedChunk to render, shared by the tests that only need something
to render.

The seam tests build their Chunks by ingesting a fixture corpus and asking a
question, which is the point of them. The surface tests need none of that --
they need a Chunk with a title, a Source type and a Locator -- and two copies
of that shape would be two places to edit when `RetrievedChunk` grows a field.
Beside tests/docx_fixture.py and tests/pdf_fixture.py, which are here for the
same reason.
"""

from core.store import RetrievedChunk


def make_chunk(**overrides) -> RetrievedChunk:
    fields = dict(
        chunk_id="doc1:000",
        doc_id="doc1",
        locator="Hash Table › Buckets",
        domain="dsa",
        source_type="note",
        title="hash table",
        text="A hash table stores each key in a bucket.",
        distance=0.2,
    )
    return RetrievedChunk(**{**fields, **overrides})
