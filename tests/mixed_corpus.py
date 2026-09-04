"""The fixture corpus the scoped-retrieval and app seams both ask questions of.

Four Documents crossing two Domains with two Source types, plus a covered
question and an out-of-corpus trap sitting either side of one test τ. The two
properties are carried by one corpus rather than by one tree each, because a
suite that needed both would otherwise have two sets of numbers to keep in
agreement: the Source-type menu and filter want a corpus where the axis is a
real choice (#5), and the gate cases want a pair of questions whose distances
straddle a threshold (#23), and neither is a reason to ingest a second corpus.

The walking skeleton's corpus (tests/fixtures/notes) has a geometry of its own,
and that is not the duplication this avoids: it is a second corpus with its own
questions, ingested at its own thresholds, and no edit here can move a distance
there. What would have been duplication is a third tree grown for the app suite
alone, carrying a second copy of the crossing this one already has.

Both facts are pinned by tests rather than left to the comments here --
`test_the_fixture_corpus_crosses_both_axes` in test_filtered_retrieval_seam.py
for the cross, and `test_the_covered_question_and_the_trap_straddle_the_test_tau`
in test_app_seam.py for the geometry -- so a fixture edited for one suite fails
loudly in the other rather than quietly making its assertions vacuous.
"""

from pathlib import Path

from core.embedder import FakeEmbedder
from core.registry import Registry, derive_doc_id
from core.store import VectorStore
from ingestion.common import ingest_folder

FIXTURES = Path(__file__).parent / "fixtures" / "mixed"

# The small pair these English fixtures are sized against, as in
# test_ingest_ask_seam.py -- this corpus is about filtering, crowding and the
# gate rather than about tokenization, so its notes stay short enough to read in
# a diff.
MIN_TOKENS = 10
MAX_TOKENS = 45

HASH_TABLE_ID = derive_doc_id("dsa/note/hash_table.md")
HASHING_ID = derive_doc_id("dsa/textbook/hashing.md")
PAGING_ID = derive_doc_id("os/note/paging.md")
VIRTUAL_MEMORY_ID = derive_doc_id("os/textbook/virtual_memory.md")

ALL_IDS = {HASH_TABLE_ID, HASHING_ID, PAGING_ID, VIRTUAL_MEMORY_ID}

# hash_table.md chunks into five, one per section; the other three Documents
# chunk into two each. Eleven, so a top_k at or above it retrieves the whole
# corpus and a filter assertion does not depend on which Chunk is nearest --
# only on which Chunks were eligible at all.
TOTAL_CHUNKS = 11

# Controlled geometry for the gate. FakeEmbedder scores literal shared
# vocabulary, so its distances are diluted by any word the corpus doesn't share
# -- including filler, which is why the pair below is matched at five content
# words and differs only in whether the corpus covers the subject: 0.50 covered,
# 0.83 trap, with GATE_TAU between them.
#
# τ is a test value rather than the configured one because it is a property of
# the embedding model (ADR-0003), and the model here is FakeEmbedder. It travels
# to the app through the same injection its embedder does, for exactly that
# reason -- see interface.Dependencies.
GATE_TAU = 0.70

COVERED_QUESTION = "bucket collision chaining probing keys"
OUT_OF_CORPUS_TRAP = "sourdough fermentation crumb proofing baguette"

# A second covered question, for the suites that ask more than once in a sitting
# (#24). Matched at the same five content words, and covered by the os side of
# the corpus rather than the dsa side -- so the two questions retrieve different
# Evidence, and "each turn kept its own cards" is a claim two unlike card lists
# make rather than one list counted twice. Pinned by
# `test_the_second_covered_question_reads_its_own_evidence` in test_app_seam.py.
SECOND_COVERED_QUESTION = "page fault frame resident disk"


def ingest_mixed(tmp_path):
    """Four ingest runs, one per (Domain, Source type) folder.

    Source type is a property of the run rather than of the file, so a crossed
    corpus is four runs over four folders rather than anything the fixtures
    declare about themselves.
    """
    registry = Registry(tmp_path / "documents.sqlite")
    store = VectorStore(path=tmp_path / "chroma")
    embedder = FakeEmbedder()

    for domain in ("dsa", "os"):
        for source_type in ("note", "textbook"):
            ingest_folder(
                folder=FIXTURES / domain / source_type,
                domain=domain,
                source_type=source_type,
                registry=registry,
                store=store,
                embedder=embedder,
                min_tokens=MIN_TOKENS,
                max_tokens=MAX_TOKENS,
                corpus_root=FIXTURES,
            )
    return registry, store, embedder
