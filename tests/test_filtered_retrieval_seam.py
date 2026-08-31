"""Seam tests for scoped questions and the per-Document Evidence cap: ingest a
fixture corpus that crosses two Domains with two Source types, then ask
questions restricted to one, the other, or both. Fully offline (FakeEmbedder,
FakeGenerator) and asserts only on retrieved doc_ids, Domains and Source types
-- never on generated text (ADR-0002).

The mainline corpus carries one Source type today, so the Source-type filter
would have nothing to discriminate on if these fixtures mirrored it. They cross
the two axes instead: each Domain holds a `note` Document and a `textbook` one,
so "only dsa" and "only my notes" each narrow the corpus to two Documents and
the two together narrow it to one. A filter that quietly ignored one of its two
arguments would still pass a corpus whose axes were correlated; it cannot pass
this one.
"""

from collections import Counter
from pathlib import Path

import pytest

from cli import source_card
from core.ask import Abstention, ask
from core.embedder import FakeEmbedder
from core.gate import ABSTENTION_TEXT
from core.generator import FakeGenerator
from core.registry import Registry, derive_doc_id
from core.retriever import _initial_pool_size
from core.store import ChunkFilter, VectorStore
from ingestion.common import ingest_folder

FIXTURES = Path(__file__).parent / "fixtures" / "mixed"

# The small pair the English fixtures in this tree are sized against, as in
# test_ingest_ask_seam.py -- these fixtures are about filtering and crowding,
# not about tokenization, so they stay short enough to read in a diff.
MIN_TOKENS = 10
MAX_TOKENS = 45

PASS_EVERYTHING = 1.0

HASH_TABLE_ID = derive_doc_id("dsa/note/hash_table.md")
HASHING_ID = derive_doc_id("dsa/textbook/hashing.md")
PAGING_ID = derive_doc_id("os/note/paging.md")
VIRTUAL_MEMORY_ID = derive_doc_id("os/textbook/virtual_memory.md")

ALL_IDS = {HASH_TABLE_ID, HASHING_ID, PAGING_ID, VIRTUAL_MEMORY_ID}

# hash_table.md chunks into five, one per section; the other three Documents
# chunk into two each. Eleven, so a top_k at or above it retrieves the whole
# corpus and the filter assertions below do not depend on which Chunk is
# nearest -- only on which Chunks were eligible at all.
TOTAL_CHUNKS = 11

# Vocabulary spread across every section of hash_table.md, so that one Document
# crowds the top of an uncapped result. The crowding is pinned by
# test_one_document_can_crowd_the_top_of_an_uncapped_result rather than assumed.
CROWDING_QUESTION = "bucket collision chaining probing load factor keys"

# A note with more Chunks than the first candidate pool asks for, written into
# the run's own tmp_path rather than kept beside the other fixtures: what it has
# to be is bigger than a number retrieval derives, and a file whose only
# property is its section count reads better next to the assertion than as
# twenty near-identical sections in the repo.
#
# Ingested under its own top-level folder, not alongside dsa/ and os/, because
# a walk retires the Documents beneath the prefix it covers (ADR-0005) -- a
# second run over dsa/note/ would retire hash_table.md rather than join it.
GIANT_SECTIONS = 20
GIANT_ID = derive_doc_id("crowding/note/many_buckets.md")


def _ingest_mixed(tmp_path):
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


def _ask(store, embedder, question=CROWDING_QUESTION, **kwargs):
    return ask(
        question,
        embedder,
        store,
        FakeGenerator(),
        distance_threshold=PASS_EVERYTHING,
        **kwargs,
    )


def _doc_ids(answer):
    return {chunk.doc_id for chunk in answer.evidence}


def test_the_fixture_corpus_crosses_both_axes(tmp_path):
    # The filter assertions below only mean anything if neither axis predicts
    # the other. Pin the cross, so a fixture moved or re-ingested under one
    # Source type fails here rather than quietly making them vacuous.
    registry, store, _embedder = _ingest_mixed(tmp_path)

    by_axes = {(doc.domain, doc.source_type): doc.doc_id for doc in registry.list()}

    assert by_axes == {
        ("dsa", "note"): HASH_TABLE_ID,
        ("dsa", "textbook"): HASHING_ID,
        ("os", "note"): PAGING_ID,
        ("os", "textbook"): VIRTUAL_MEMORY_ID,
    }
    assert len(store.collection.get()["ids"]) == TOTAL_CHUNKS


def test_an_unfiltered_question_reaches_every_document(tmp_path):
    # The baseline the filtered questions are read against: at this top_k the
    # whole corpus is eligible, so anything missing below was excluded by a
    # filter rather than merely out-ranked.
    _registry, store, embedder = _ingest_mixed(tmp_path)

    answer = _ask(store, embedder, top_k=TOTAL_CHUNKS)

    assert _doc_ids(answer) == ALL_IDS


def test_a_domain_filter_retrieves_only_that_domains_chunks(tmp_path):
    _registry, store, embedder = _ingest_mixed(tmp_path)

    answer = _ask(
        store, embedder, top_k=TOTAL_CHUNKS, chunk_filter=ChunkFilter(domain="dsa")
    )

    assert _doc_ids(answer) == {HASH_TABLE_ID, HASHING_ID}
    assert all(chunk.domain == "dsa" for chunk in answer.evidence)


def test_a_source_type_filter_retrieves_only_that_source_types_chunks(tmp_path):
    _registry, store, embedder = _ingest_mixed(tmp_path)

    answer = _ask(
        store,
        embedder,
        top_k=TOTAL_CHUNKS,
        chunk_filter=ChunkFilter(source_type="textbook"),
    )

    assert _doc_ids(answer) == {HASHING_ID, VIRTUAL_MEMORY_ID}
    assert all(chunk.source_type == "textbook" for chunk in answer.evidence)


def test_both_filters_together_narrow_further_than_either_alone(tmp_path):
    # "My own DSA notes": the intersection is one Document, and it is not the
    # only one either filter leaves on its own, so neither argument can be the
    # one doing all the work.
    _registry, store, embedder = _ingest_mixed(tmp_path)

    answer = _ask(
        store,
        embedder,
        top_k=TOTAL_CHUNKS,
        chunk_filter=ChunkFilter(domain="dsa", source_type="note"),
    )

    assert _doc_ids(answer) == {HASH_TABLE_ID}


def test_a_filter_the_corpus_cannot_satisfy_abstains_rather_than_reaching_wider(
    tmp_path,
):
    # A Domain the corpus holds nothing under. The restriction is not a
    # preference to be relaxed once it turns out to be inconvenient: there is
    # no Evidence inside it, so the gate abstains on the empty result exactly
    # as it does for a question the whole corpus fails to cover.
    _registry, store, embedder = _ingest_mixed(tmp_path)

    answer = _ask(
        store,
        embedder,
        top_k=TOTAL_CHUNKS,
        chunk_filter=ChunkFilter(domain="security"),
    )

    assert answer.abstention is Abstention.DISTANCE_GATE
    assert answer.text == ABSTENTION_TEXT
    assert answer.evidence == []


def test_an_answer_carries_the_filter_the_question_was_answered_under(tmp_path):
    # The scope is part of what happened, not part of how it was asked: a turn
    # redisplayed after the filter has moved on has to show the scope it was
    # answered under, and re-reading the filter in force at display time would
    # relabel it retroactively.
    _registry, store, embedder = _ingest_mixed(tmp_path)
    scope = ChunkFilter(domain="dsa", source_type="note")

    answer = _ask(store, embedder, top_k=TOTAL_CHUNKS, chunk_filter=scope)

    assert answer.chunk_filter == scope


def test_an_unrestricted_question_carries_the_unfiltered_filter(tmp_path):
    # `ChunkFilter()` is the unfiltered question (core/store.py), so the
    # unfiltered case needs no second representation and no reader downstream
    # has to branch on a missing one.
    _registry, store, embedder = _ingest_mixed(tmp_path)

    answer = _ask(store, embedder, top_k=TOTAL_CHUNKS)

    assert answer.chunk_filter == ChunkFilter()


def test_a_gate_abstention_carries_the_scope_that_produced_it(tmp_path):
    # The case the scope exists to disambiguate: nothing came back because the
    # question was restricted to a Domain the corpus holds nothing under, which
    # is not the same fact about the corpus as "nothing covers this".
    _registry, store, embedder = _ingest_mixed(tmp_path)
    scope = ChunkFilter(domain="security")

    answer = _ask(store, embedder, top_k=TOTAL_CHUNKS, chunk_filter=scope)

    assert answer.abstention is Abstention.DISTANCE_GATE
    assert answer.chunk_filter == scope


def test_one_document_can_crowd_the_top_of_an_uncapped_result(tmp_path):
    # What the cap exists to stop, pinned before it is applied. Without this
    # the tests below could pass on a corpus where no Document ever supplied
    # two of the five slots in the first place.
    _registry, store, embedder = _ingest_mixed(tmp_path)

    answer = _ask(store, embedder, top_k=5)

    crowded = Counter(chunk.doc_id for chunk in answer.evidence)
    assert crowded[HASH_TABLE_ID] >= 3
    assert len(crowded) < 4


def test_the_cap_limits_how_many_chunks_one_document_contributes(tmp_path):
    _registry, store, embedder = _ingest_mixed(tmp_path)

    answer = _ask(store, embedder, top_k=5, max_chunks_per_document=2)

    per_document = Counter(chunk.doc_id for chunk in answer.evidence)
    assert max(per_document.values()) <= 2


def test_the_cap_is_configurable_rather_than_a_fixed_number(tmp_path):
    # The same question at a second cap. A hard-coded limit would satisfy the
    # test above at whichever number it was hard-coded to.
    _registry, store, embedder = _ingest_mixed(tmp_path)

    answer = _ask(store, embedder, top_k=5, max_chunks_per_document=1)

    per_document = Counter(chunk.doc_id for chunk in answer.evidence)
    assert max(per_document.values()) == 1
    assert len(answer.evidence) == 4  # one apiece, and the corpus holds four


def test_capping_frees_slots_for_other_documents_rather_than_shrinking_evidence(
    tmp_path,
):
    # The whole point, and the part a cap applied to an already-truncated top_k
    # gets wrong: the slots the crowding Document gives up go to other
    # Documents, so generation is handed no less Evidence than it asked for.
    _registry, store, embedder = _ingest_mixed(tmp_path)

    uncapped = _ask(store, embedder, top_k=5)
    capped = _ask(store, embedder, top_k=5, max_chunks_per_document=2)

    assert len(capped.evidence) == len(uncapped.evidence) == 5
    assert _doc_ids(capped) > _doc_ids(uncapped)


def test_the_cap_never_displaces_the_nearest_chunk(tmp_path):
    # The cap keeps each Document's nearest Chunks, so the nearest Chunk
    # overall always survives it. That is what leaves the distance gate reading
    # the same number capped or not -- a cap that could drop it would turn a
    # diversity limit into a silent second abstention rule (ADR-0003).
    _registry, store, embedder = _ingest_mixed(tmp_path)

    uncapped = _ask(store, embedder, top_k=5)
    capped = _ask(store, embedder, top_k=5, max_chunks_per_document=1)

    assert capped.evidence[0].chunk_id == uncapped.evidence[0].chunk_id
    assert capped.evidence[0].distance == uncapped.evidence[0].distance


def test_a_filter_and_a_cap_apply_together(tmp_path):
    # Both restrictions on one question: the Evidence stays inside the Domain,
    # and within it no Document exceeds the cap.
    _registry, store, embedder = _ingest_mixed(tmp_path)

    answer = _ask(
        store,
        embedder,
        top_k=5,
        chunk_filter=ChunkFilter(domain="dsa"),
        max_chunks_per_document=2,
    )

    per_document = Counter(chunk.doc_id for chunk in answer.evidence)
    assert set(per_document) == {HASH_TABLE_ID, HASHING_ID}
    assert max(per_document.values()) <= 2


def test_the_source_cards_reflect_the_filtered_capped_evidence(tmp_path):
    # What the reader is shown, rather than what retrieval returned. The cards
    # are the only place a Document appears by name, so a card for a Document
    # the filter excluded -- or a sixth card for a Document the cap held to two
    # -- would credit the answer to Evidence generation was never given.
    _registry, store, embedder = _ingest_mixed(tmp_path)

    answer = _ask(
        store,
        embedder,
        top_k=5,
        chunk_filter=ChunkFilter(domain="dsa"),
        max_chunks_per_document=2,
    )
    cards = [source_card(chunk) for chunk in answer.evidence]

    assert len(cards) == len(answer.evidence)
    assert all("(note)" in card or "(textbook)" in card for card in cards)
    assert not any("paging" in card or "virtual memory" in card for card in cards)
    assert sum("hash table" in card for card in cards) == 2


def _ingest_with_a_giant_note(tmp_path):
    """The crossed corpus, plus one Document that alone outnumbers the pool."""
    registry, store, embedder = _ingest_mixed(tmp_path)

    folder = tmp_path / "crowding" / "note"
    folder.mkdir(parents=True)
    sections = "\n".join(
        f"## Bucket {i}\n\nBucket {i} of this table takes the keys that hash to "
        f"bucket {i}, and a collision there is resolved by chaining rather than "
        f"by probing, whichever the load factor favours.\n"
        for i in range(GIANT_SECTIONS)
    )
    (folder / "many_buckets.md").write_text(
        f"# Many Buckets\n\n{sections}", encoding="utf-8"
    )
    ingest_folder(
        folder=folder,
        domain="dsa",
        source_type="note",
        registry=registry,
        store=store,
        embedder=embedder,
        min_tokens=MIN_TOKENS,
        max_tokens=MAX_TOKENS,
        corpus_root=tmp_path,
    )
    return registry, store, embedder


def test_one_document_can_monopolise_the_first_candidate_pool(tmp_path):
    # The precondition the test below is about, pinned the same way the
    # uncapped crowding is: a Document whose own Chunks are the whole first
    # pool, so nothing else is even a candidate to hand a freed slot to.
    _registry, store, embedder = _ingest_with_a_giant_note(tmp_path)
    [embedding] = embedder.embed([CROWDING_QUESTION])

    pool = store.query(embedding=embedding, top_k=_initial_pool_size(5, 2))

    assert GIANT_SECTIONS > _initial_pool_size(5, 2)
    assert {chunk.doc_id for chunk in pool} == {GIANT_ID}


def test_a_document_that_monopolises_the_pool_does_not_shrink_the_evidence(tmp_path):
    # A fixed candidate pool caps the pool's one Document down to two Chunks
    # and returns them as the whole Evidence -- five slots asked for, two
    # filled, the other Documents never reached. The pool has to keep growing
    # until the freed slots are actually fillable.
    _registry, store, embedder = _ingest_with_a_giant_note(tmp_path)

    answer = _ask(store, embedder, top_k=5, max_chunks_per_document=2)

    per_document = Counter(chunk.doc_id for chunk in answer.evidence)
    assert len(answer.evidence) == 5
    assert per_document[GIANT_ID] == 2
    assert len(per_document) >= 3


def test_growing_the_pool_stops_at_the_corpus_rather_than_looping(tmp_path):
    # The other end of the growth: a top_k the corpus cannot fill under the cap
    # returns what there is, once, instead of asking for ever larger pools of a
    # collection that has no more to give.
    _registry, store, embedder = _ingest_mixed(tmp_path)

    answer = _ask(store, embedder, top_k=TOTAL_CHUNKS, max_chunks_per_document=1)

    assert len(answer.evidence) == 4  # one apiece, and the corpus holds four


def test_a_cap_that_admits_no_chunk_is_refused_rather_than_abstaining(tmp_path):
    # Zero would empty the Evidence and abstain, which reads as "the corpus
    # does not cover this" -- indistinguishable from the real abstention.
    _registry, store, embedder = _ingest_mixed(tmp_path)

    with pytest.raises(ValueError, match="max_chunks_per_document"):
        _ask(store, embedder, top_k=5, max_chunks_per_document=0)
