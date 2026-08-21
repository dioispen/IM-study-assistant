"""Seam tests for the walking skeleton: ingest a fixture corpus of notes --
Markdown, and Word beside it -- then ask questions against it. Fully offline
(FakeEmbedder, FakeGenerator) and asserts only on retrieved doc_ids and
Locators -- never on generated text (ADR-0002).
"""

import re
import shutil
from pathlib import Path

from core.ask import ask
from core.embedder import FakeEmbedder
from core.gate import ABSTENTION_TEXT
from core.generator import FakeGenerator
from core.registry import Registry, derive_doc_id
from core.store import VectorStore
from ingestion.common import ingest_folder
from tests.docx_fixture import BODY, corrupt_a_part, write_docx

FIXTURES = Path(__file__).parent / "fixtures" / "notes"

# The English fixtures are sized against this small pair, which keeps them
# short enough to read in a diff. The Chinese fixtures are sized against the
# thresholds config.py actually ships and are ingested with nothing overridden;
# see `_ingest_with_chinese`.
MIN_TOKENS = 10
MAX_TOKENS = 45

# Controlled geometry for the gate tests. FakeEmbedder scores literal shared
# vocabulary, so its distances are diluted by any word the corpus doesn't
# share -- including filler. A verbose covered question and a terse one land
# far apart ("What is round robin?" sits at 0.82, as far out as the trap
# below), which would make a length-mismatched pair prove nothing about
# coverage. So this pair is matched at five content words and differs only in
# whether the corpus covers the subject: 0.60 covered, 0.82 trap, with
# GATE_TAU between them. test_the_gate_tests_geometry_is_what_it_claims pins
# these numbers so the comment cannot drift.
GATE_TAU = 0.70
PASS_EVERYTHING = 1.0

COVERED_QUESTION = "round robin scheduling time quantum"
OUT_OF_CORPUS_TRAP = "baroque counterpoint fugue harpsichord ornamentation"


class ExplodingGenerator:
    """Fails the test if generation is reached -- the gate must abstain first."""

    def generate(self, prompt: str) -> str:
        raise AssertionError("the distance gate should have abstained before generation")

BST_ID = derive_doc_id("dsa/binary_search_tree.md")
SCHEDULING_ID = derive_doc_id("os/process_scheduling.md")
DEADLOCK_ID = derive_doc_id("os/deadlock.md")
HANDSHAKE_ID = derive_doc_id("network/tcp_handshake.md")
OSI_ID = derive_doc_id("network/osi_model.md")

ZH_HANDSHAKE_QUESTION = "三向交握的第三個封包由用戶端送出"
ZH_LAYERING_QUESTION = "應用層之下依序是哪幾層"

# The nested corpus: one Domain folder whose notes sit in subfolders, two of
# them sharing a filename with a third at the folder's own level.
NESTED_FIXTURES = FIXTURES / "mis"

MIS_OVERVIEW_ID = derive_doc_id("mis/概述.md")
DECISION_OVERVIEW_ID = derive_doc_id("mis/決策支援/概述.md")
PROCESS_OVERVIEW_ID = derive_doc_id("mis/流程管理/概述.md")
BPMN_ID = derive_doc_id("mis/流程管理/塑模/bpmn.md")

NESTED_IDS = {MIS_OVERVIEW_ID, DECISION_OVERVIEW_ID, PROCESS_OVERVIEW_ID, BPMN_ID}
ZH_BPMN_QUESTION = "閘道標示流程在此分支或匯流"


def _ingest_corpus(tmp_path, fixtures=FIXTURES):
    """The English corpus, and the one the gate tests read distances off.

    The Chinese fixtures are ingested by `_ingest_with_chinese` instead of
    being added here; see its docstring for why folding the two together makes
    the gate's out-of-corpus trap vacuous.
    """
    registry = Registry(tmp_path / "documents.sqlite")
    store = VectorStore(path=tmp_path / "chroma")
    embedder = FakeEmbedder()

    dsa_report = ingest_folder(
        folder=fixtures / "dsa",
        domain="dsa",
        source_type="note",
        registry=registry,
        store=store,
        embedder=embedder,
        min_tokens=MIN_TOKENS,
        max_tokens=MAX_TOKENS,
        corpus_root=fixtures,
    )
    os_report = ingest_folder(
        folder=fixtures / "os",
        domain="os",
        source_type="note",
        registry=registry,
        store=store,
        embedder=embedder,
        min_tokens=MIN_TOKENS,
        max_tokens=MAX_TOKENS,
        corpus_root=fixtures,
    )
    return registry, store, embedder, dsa_report, os_report


def _ingest_with_chinese(tmp_path, fixtures=FIXTURES):
    """The English corpus with the Chinese one ingested alongside it, the
    Chinese folder at whatever thresholds config.py ships.

    Overriding nothing there is the point. The Chinese fixtures used to be
    ingested at MIN_TOKENS/MAX_TOKENS like the English ones, which meant the
    split and merge branches fired only because this module had substituted a
    smaller pair: the fixtures demonstrated the branches work at that pair and
    said nothing about the configured one, under which `osi_model.md` still
    collapsed into a single Chunk carrying only its last section's Locator. The
    fixtures are now sized against the configured numbers instead, and the tests
    below read the branches off those. The English fixtures keep the small pair
    they are sized for; here they are retrieval competition, not the subject.

    Deliberately not folded into `_ingest_corpus`, because the gate tests read
    distances off that store. FakeEmbedder hashes into 64 buckets and every
    Chinese Chunk sized for the configured thresholds fills at least 47 of
    them, so a five-word English query collides with one on most of its words
    by chance -- the out-of-corpus trap lands at 0.64, inside GATE_TAU, and
    abstains for no reason connected to what it asks about. That is an artifact of the offline
    double's dimensionality rather than anything the pipeline does (real
    embeddings are cross-lingual and 1536-dimensional), but it would make the
    trap prove nothing, so the gate keeps the corpus its geometry was pinned
    against and the Chinese tests take this one.
    """
    registry, store, embedder, _dsa, _os = _ingest_corpus(tmp_path, fixtures=fixtures)
    report = ingest_folder(
        folder=fixtures / "network",
        domain="network",
        source_type="note",
        registry=registry,
        store=store,
        embedder=embedder,
        corpus_root=fixtures,
    )
    return registry, store, embedder, report


def test_ingest_populates_the_registry_with_every_fixture_document(tmp_path):
    registry, store, embedder, dsa_report, os_report = _ingest_corpus(tmp_path)

    doc_ids = {doc.doc_id for doc in registry.list()}
    assert doc_ids == {BST_ID, SCHEDULING_ID, DEADLOCK_ID}
    assert set(dsa_report.ingested) == {BST_ID}
    assert set(os_report.ingested) == {SCHEDULING_ID, DEADLOCK_ID}


def test_chunk_ids_are_doc_id_colon_ordinal(tmp_path):
    _registry, store, _embedder, _dsa, _os = _ingest_corpus(tmp_path)

    result = store.collection.get(where={"doc_id": BST_ID})

    assert set(result["ids"]) == {f"{BST_ID}:000", f"{BST_ID}:001"}


def test_reingesting_an_unchanged_folder_skips_every_document(tmp_path):
    registry = Registry(tmp_path / "documents.sqlite")
    store = VectorStore(path=tmp_path / "chroma")
    embedder = FakeEmbedder()
    kwargs = dict(
        folder=FIXTURES / "dsa",
        domain="dsa",
        source_type="note",
        registry=registry,
        store=store,
        embedder=embedder,
        min_tokens=MIN_TOKENS,
        max_tokens=MAX_TOKENS,
        corpus_root=FIXTURES,
    )

    first = ingest_folder(**kwargs)
    second = ingest_folder(**kwargs)

    assert first.ingested == [BST_ID]
    assert first.skipped == []
    assert second.ingested == []
    assert second.skipped == [BST_ID]


def test_running_ingestion_twice_over_the_same_corpus_changes_nothing(tmp_path):
    _registry, store, _embedder, dsa, os_ = _ingest_corpus(tmp_path)
    after_one_run = store.collection.get()["ids"]

    _registry, store, _embedder, dsa_again, os_again = _ingest_corpus(tmp_path)

    after_two_runs = store.collection.get()["ids"]
    assert sorted(after_two_runs) == sorted(after_one_run)
    assert len(after_two_runs) == len(set(after_two_runs))
    # Pinned so the assertion above can't pass vacuously for the wrong reason:
    # every Document really did take the unchanged-skip path.
    assert (dsa_again.skipped, os_again.skipped) == (dsa.ingested, os_.ingested)


def test_re_ingesting_an_edited_corpus_leaves_no_duplicate_or_stale_chunks(tmp_path):
    # The teeth of the idempotency claim: the run above is all registry skips,
    # so only an edited corpus exercises the replace path at the seam. The
    # fixtures are copied first because they are repo files.
    corpus = tmp_path / "notes"
    shutil.copytree(FIXTURES, corpus)
    _registry, store, _embedder, _dsa, _first = _ingest_corpus(tmp_path, fixtures=corpus)
    # deadlock.md is the fixture that chunks into two, so shrinking it to one
    # is what forces the old generation's tail to be deleted rather than
    # merely overwritten -- the case a blind upsert would leave stale.
    assert len(store.collection.get(where={"doc_id": DEADLOCK_ID})["ids"]) == 2
    (corpus / "os" / "deadlock.md").write_text(
        "# Deadlock\n\n## Conditions\n\nA rewrite far shorter than the note it "
        "replaces, saying nothing about preemption at all.\n",
        encoding="utf-8",
    )

    _registry, store, _embedder, _dsa, second = _ingest_corpus(tmp_path, fixtures=corpus)

    assert second.ingested == [DEADLOCK_ID]
    assert second.skipped == [SCHEDULING_ID]
    ids = store.collection.get()["ids"]
    assert len(ids) == len(set(ids))
    deadlock = store.collection.get(where={"doc_id": DEADLOCK_ID})
    assert deadlock["ids"] == [f"{DEADLOCK_ID}:000"]
    assert not any("circular wait" in text for text in deadlock["documents"])


def test_oversized_section_becomes_multiple_chunks_sharing_one_locator(tmp_path):
    _registry, store, _embedder, _dsa, _os = _ingest_corpus(tmp_path)

    result = store.collection.get(where={"doc_id": DEADLOCK_ID})

    assert len(result["ids"]) > 1
    assert {m["locator"] for m in result["metadatas"]} == {"Deadlock › Conditions"}
    assert sorted(result["ids"]) == [f"{DEADLOCK_ID}:000", f"{DEADLOCK_ID}:001"]


def test_no_chunk_straddles_a_heading(tmp_path):
    _registry, store, _embedder, _dsa, _os = _ingest_corpus(tmp_path)

    result = store.collection.get(where={"doc_id": BST_ID})
    text_by_locator = dict(
        zip((m["locator"] for m in result["metadatas"]), result["documents"])
    )

    assert "successor" not in text_by_locator["Binary Search Tree › Insertion"]
    assert "empty spot" not in text_by_locator["Binary Search Tree › Deletion"]


def test_asking_a_question_retrieves_the_document_its_vocabulary_matches(tmp_path):
    _registry, store, embedder, _dsa, _os = _ingest_corpus(tmp_path)
    generator = FakeGenerator()

    answer = ask(
        COVERED_QUESTION,
        embedder,
        store,
        generator,
        top_k=1,
        distance_threshold=PASS_EVERYTHING,
    )

    assert len(answer.evidence) == 1
    top = answer.evidence[0]
    assert top.doc_id == SCHEDULING_ID
    assert top.locator == "Process Scheduling › Round Robin"
    assert top.source_type == "note"
    assert top.domain == "os"


def test_asking_a_different_question_retrieves_a_different_document(tmp_path):
    _registry, store, embedder, _dsa, _os = _ingest_corpus(tmp_path)
    generator = FakeGenerator()

    answer = ask(
        "What are the four conditions required for deadlock to occur?",
        embedder,
        store,
        generator,
        top_k=1,
        distance_threshold=PASS_EVERYTHING,
    )

    assert answer.evidence[0].doc_id == DEADLOCK_ID


def test_ask_never_asserts_on_generated_text_only_on_evidence(tmp_path):
    _registry, store, embedder, _dsa, _os = _ingest_corpus(tmp_path)
    generator = FakeGenerator()

    answer = ask(
        "What is round robin?",
        embedder,
        store,
        generator,
        top_k=2,
        distance_threshold=PASS_EVERYTHING,
    )

    assert isinstance(answer.text, str) and answer.text
    assert all(isinstance(chunk.doc_id, str) for chunk in answer.evidence)


# Chinese corpus. Every fixture above is English, which is how a Chinese note
# collapsing into one Chunk survived a green suite; these ingest and ask in the
# language the corpus is actually written in (PLAN.md: notes are zh-tw), at the
# thresholds config.py ships rather than the small pair the English ones use.
#
# The fixtures are sized so that one section of each branch exists under those
# numbers: 三向交握 is oversized, 概述 is undersized, and 連線終止, 分層架構 and
# 封裝 all sit between. No test here asserts a token count for a string -- the
# measure behind the counts is a placeholder due for replacement in Week 6
# (ADR-0004), so what is pinned is the chunking that came out of it.


def test_chinese_documents_enter_the_registry_under_their_own_domain(tmp_path):
    registry, _store, _embedder, report = _ingest_with_chinese(tmp_path)

    docs = {doc.doc_id: doc for doc in registry.list()}
    assert set(report.ingested) == {HANDSHAKE_ID, OSI_ID}
    assert docs[HANDSHAKE_ID].domain == docs[OSI_ID].domain == "network"
    assert docs[HANDSHAKE_ID].title == "tcp handshake"


def test_chinese_documents_chunk_into_their_sections_with_chunk_id_ordinals(tmp_path):
    _registry, store, _embedder, _report = _ingest_with_chinese(tmp_path)

    handshake = store.collection.get(where={"doc_id": HANDSHAKE_ID})
    osi = store.collection.get(where={"doc_id": OSI_ID})

    # 三向交握 splits in two and 連線終止 stands alone, so three Chunks over two
    # Locators; 概述 merges away, so two Chunks over the two Locators left.
    assert sorted(handshake["ids"]) == [f"{HANDSHAKE_ID}:{i:03d}" for i in range(3)]
    assert {m["locator"] for m in handshake["metadatas"]} == {
        "傳輸控制協定 › 三向交握",
        "傳輸控制協定 › 連線終止",
    }
    assert sorted(osi["ids"]) == [f"{OSI_ID}:{i:03d}" for i in range(2)]
    assert {m["locator"] for m in osi["metadatas"]} == {
        "OSI 參考模型 › 分層架構",
        "OSI 參考模型 › 封裝",
    }


def test_no_chinese_document_collapses_into_one_chunk_under_its_last_locator(tmp_path):
    # The shape #11 was opened to fix, asserted directly rather than inferred
    # from the counts above: a whole multi-section note as a single Chunk,
    # citable only as whichever heading it happens to end under. It survived
    # #13 because the Chinese seam ran at the small pair and never at the
    # configured one.
    _registry, store, _embedder, _report = _ingest_with_chinese(tmp_path)

    for doc_id, last_locator in (
        (HANDSHAKE_ID, "傳輸控制協定 › 連線終止"),
        (OSI_ID, "OSI 參考模型 › 封裝"),
    ):
        result = store.collection.get(where={"doc_id": doc_id})

        assert len(result["ids"]) > 1
        assert {m["locator"] for m in result["metadatas"]} != {last_locator}


def _texts_under(store, doc_id, locator):
    """Every Chunk of `doc_id` carrying `locator`, in ordinal order.

    Sorted rather than trusting the order `get` returns, because a caller
    reading the pieces of a split section reads them as a sequence.
    """
    result = store.collection.get(where={"doc_id": doc_id})
    in_order = sorted(
        zip(result["metadatas"], result["documents"]), key=lambda pair: pair[0]["ordinal"]
    )
    return [text for meta, text in in_order if meta["locator"] == locator]


def test_an_oversized_chinese_section_splits_into_chunks_sharing_one_locator(tmp_path):
    # The branch a whitespace assumption cannot reach: 三向交握 counts as one
    # word split on whitespace, so it would never read as oversized at all.
    _registry, store, _embedder, _report = _ingest_with_chinese(tmp_path)

    handshake_texts = _texts_under(store, HANDSHAKE_ID, "傳輸控制協定 › 三向交握")

    assert len(handshake_texts) == 2
    assert "".join(handshake_texts).startswith("用戶端先送出 SYN 封包")
    assert not any("四次揮手" in text for text in handshake_texts)


def test_a_chinese_section_between_the_thresholds_becomes_one_chunk_of_its_own(tmp_path):
    # Neither split nor merged: the case that has to hold for the two either
    # side of it to mean anything, since a rule that always splits or always
    # merges would satisfy them both.
    _registry, store, _embedder, _report = _ingest_with_chinese(tmp_path)

    termination_texts = _texts_under(store, HANDSHAKE_ID, "傳輸控制協定 › 連線終止")

    assert len(termination_texts) == 1
    assert "四次揮手" in termination_texts[0]
    assert "SYN cookie" not in termination_texts[0]


def test_an_undersized_chinese_section_merges_into_its_neighbour(tmp_path):
    _registry, store, _embedder, _report = _ingest_with_chinese(tmp_path)

    assert _texts_under(store, OSI_ID, "OSI 參考模型 › 概述") == []
    [layering] = _texts_under(store, OSI_ID, "OSI 參考模型 › 分層架構")
    assert "分層的目的是解耦" in layering


def test_asking_a_chinese_question_retrieves_the_document_its_vocabulary_matches(tmp_path):
    _registry, store, embedder, _report = _ingest_with_chinese(tmp_path)

    answer = ask(
        ZH_HANDSHAKE_QUESTION,
        embedder,
        store,
        FakeGenerator(),
        top_k=1,
        distance_threshold=PASS_EVERYTHING,
    )

    [top] = answer.evidence
    assert top.doc_id == HANDSHAKE_ID
    assert top.locator == "傳輸控制協定 › 三向交握"
    assert top.domain == "network"


def test_a_different_chinese_question_retrieves_the_other_chinese_document(tmp_path):
    # Two Chinese Documents in one Domain, so the assertion above cannot pass
    # merely because the question was the only Chinese text in the store.
    _registry, store, embedder, _report = _ingest_with_chinese(tmp_path)

    answer = ask(
        ZH_LAYERING_QUESTION,
        embedder,
        store,
        FakeGenerator(),
        top_k=1,
        distance_threshold=PASS_EVERYTHING,
    )

    [top] = answer.evidence
    assert top.doc_id == OSI_ID
    assert top.locator == "OSI 參考模型 › 分層架構"


def test_reingesting_the_unchanged_chinese_corpus_skips_every_document(tmp_path):
    _registry, _store, _embedder, first = _ingest_with_chinese(tmp_path)

    _registry, _store, _embedder, second = _ingest_with_chinese(tmp_path)

    assert second.ingested == []
    assert second.skipped == first.ingested


def test_a_chinese_document_with_no_prose_is_reported_as_failed(tmp_path):
    # An ideographic space is whitespace, so this Document has headings and no
    # prose. Recording it as ingested would put a Document in the registry that
    # contributes no Chunk to any answer.
    folder = tmp_path / "network"
    folder.mkdir()
    (folder / "empty.md").write_text("# 網路\n\n## 概述\n\n　　\n", encoding="utf-8")
    registry = Registry(tmp_path / "documents.sqlite")

    report = ingest_folder(
        folder=folder,
        domain="network",
        source_type="note",
        registry=registry,
        store=VectorStore(path=tmp_path / "chroma"),
        embedder=FakeEmbedder(),
        corpus_root=tmp_path,
    )

    assert report.ingested == []
    [failure] = report.failed
    assert failure.source_path == "network/empty.md"
    assert "chunk" in failure.reason.lower()
    assert registry.list() == []


# Nested corpus. Every fixture folder above is one flat level, which is how
# ingestion reading only the top level of the folder it was handed survived a
# green suite: a corpus in subfolders reported "Ingested 0" and the student was
# told nothing was wrong. These ingest a Domain folder whose notes sit one and
# two subfolders down, at the thresholds config.py ships. The nesting rules
# themselves are unit-tested in test_ingest_common.py; what these add is that
# the corpus on disk really is nested, so the flat-only assumption cannot come
# back unnoticed.


def test_the_doc_ids_of_the_flat_fixture_corpus_survive_the_walk(tmp_path):
    # Literals rather than derive_doc_id calls, and read off a real ingest
    # rather than off the hash function: what must hold still is the doc_id
    # ingestion assigns a note in a flat folder. A doc_id that shifts is a full
    # re-embed of the corpus for no behavioural reason, and that stability is
    # what ADR-0001's registry depends on -- so a future change to the walk, or
    # to what a path is derived relative to, fails here rather than silently
    # re-embedding.
    registry, _store, _embedder, _report = _ingest_with_chinese(tmp_path)

    assert {doc.source_path: doc.doc_id for doc in registry.list()} == {
        "dsa/binary_search_tree.md": "e90f373100c8",
        "os/process_scheduling.md": "602b7656a373",
        "os/deadlock.md": "16812f81788d",
        "network/tcp_handshake.md": "11093ebca1a8",
        "network/osi_model.md": "73c926e94948",
    }


def _ingest_nested(tmp_path, fixtures=NESTED_FIXTURES):
    """The nested corpus, in its own registry and store.

    Kept out of `_ingest_corpus` for the reason `_ingest_with_chinese` gives at
    length: the gate tests read distances off that store, and Chinese Chunks
    sized for the configured thresholds fill enough of FakeEmbedder's 64
    buckets to pull the out-of-corpus trap inside GATE_TAU.
    """
    registry = Registry(tmp_path / "documents.sqlite")
    store = VectorStore(path=tmp_path / "chroma")
    embedder = FakeEmbedder()
    report = ingest_folder(
        folder=fixtures,
        domain="mis",
        source_type="note",
        registry=registry,
        store=store,
        embedder=embedder,
        # The fixture tree, one folder above the Domain folder being ingested --
        # the same root the other helpers pass, so `mis/...` is what a Document
        # here is named by whichever helper put it in a registry.
        corpus_root=fixtures.parent,
    )
    return registry, store, embedder, report


def test_every_note_beneath_the_nested_corpus_is_ingested(tmp_path):
    # Three of the four are named 概述.md -- one at the folder's own level and
    # two in subfolders -- so the four distinct doc_ids and four source_paths
    # are also what says none of them replaced another.
    registry, _store, _embedder, report = _ingest_nested(tmp_path)

    assert set(report.ingested) == NESTED_IDS
    assert len(NESTED_IDS) == 4
    assert {doc.source_path for doc in registry.list()} == {
        "mis/概述.md",
        "mis/決策支援/概述.md",
        "mis/流程管理/概述.md",
        "mis/流程管理/塑模/bpmn.md",
    }


def test_a_nested_note_is_cited_back_to_its_file_on_disk(tmp_path):
    # The point of carrying the subfolder segments into source_path: an answer
    # names a Chunk, and the Document behind it names a path that resolves.
    registry, store, embedder, _report = _ingest_nested(tmp_path)

    answer = ask(
        ZH_BPMN_QUESTION,
        embedder,
        store,
        FakeGenerator(),
        top_k=1,
        distance_threshold=PASS_EVERYTHING,
    )

    [top] = answer.evidence
    assert top.doc_id == BPMN_ID
    document = registry.get(BPMN_ID)
    assert document.source_path == "mis/流程管理/塑模/bpmn.md"
    # Resolved under the corpus root the helper passed, which is what following
    # a citation back to disk actually has to hand.
    assert (FIXTURES / document.source_path).is_file()


def test_reingesting_the_unchanged_nested_corpus_skips_every_document(tmp_path):
    # Not the flat skip test over again: the skip keys on a doc_id derived from
    # the path relative to the folder, so a walk that yielded an absolute path
    # -- or one relative to something else -- would re-ingest the whole nested
    # corpus every run while the flat one still skipped.
    _registry, _store, _embedder, first = _ingest_nested(tmp_path)

    _registry, _store, _embedder, second = _ingest_nested(tmp_path)

    assert second.ingested == []
    assert sorted(second.skipped) == sorted(first.ingested)


def _ask_through_the_gate(question, store, embedder, generator):
    return ask(question, embedder, store, generator, top_k=5, distance_threshold=GATE_TAU)


def test_the_gate_tests_geometry_is_what_it_claims(tmp_path):
    # The two gate tests below only mean anything if the trap really is the
    # farther of a length-matched pair. Pin that, so a change to FakeEmbedder
    # or to the fixtures fails here rather than quietly making them vacuous.
    _registry, store, embedder, _dsa, _os = _ingest_corpus(tmp_path)
    [covered], [trap] = embedder.embed([COVERED_QUESTION]), embedder.embed([OUT_OF_CORPUS_TRAP])

    covered_distance = store.query(embedding=covered, top_k=1)[0].distance
    trap_distance = store.query(embedding=trap, top_k=1)[0].distance

    assert len(COVERED_QUESTION.split()) == len(OUT_OF_CORPUS_TRAP.split())
    assert covered_distance < GATE_TAU < trap_distance


def test_an_out_of_corpus_trap_abstains_without_calling_the_llm(tmp_path):
    _registry, store, embedder, _dsa, _os = _ingest_corpus(tmp_path)

    answer = _ask_through_the_gate(OUT_OF_CORPUS_TRAP, store, embedder, ExplodingGenerator())

    assert answer.abstained
    assert answer.text == ABSTENTION_TEXT


def test_an_abstention_cites_nothing(tmp_path):
    _registry, store, embedder, _dsa, _os = _ingest_corpus(tmp_path)

    answer = _ask_through_the_gate(OUT_OF_CORPUS_TRAP, store, embedder, ExplodingGenerator())

    assert answer.evidence == []


def test_a_question_the_corpus_covers_still_answers_under_the_same_gate(tmp_path):
    _registry, store, embedder, _dsa, _os = _ingest_corpus(tmp_path)

    answer = _ask_through_the_gate(COVERED_QUESTION, store, embedder, FakeGenerator())

    assert not answer.abstained
    assert answer.evidence[0].doc_id == SCHEDULING_ID


# Mixed-format corpus. Every fixture above is Markdown, which is how "the walk
# looks for *.md" survived a green suite: a student whose notes are Word files
# was told "Ingested 0" and nothing was wrong. These ingest a folder holding
# both formats in one run, at the thresholds config.py ships, and read the
# Locators back through `ask` -- the entry point a citation actually reaches
# the reader through (#9).
#
# The docx fixtures are written at test time rather than committed, for the
# reason docs/corpus-sources.md gives: a committed fixture has to be reviewable
# in a diff, and a .docx is a zip. See tests/docx_fixture.py.

DOCX_HANDSHAKE_ID = derive_doc_id("network/tcp_handshake.docx")
DOCX_OSI_ID = derive_doc_id("network/osi_model.docx")


def _as_docx_blocks(markdown: str) -> list[tuple[str | None, str]]:
    """A Markdown fixture's headings and prose as the Word note saying the same.

    Written off the committed `.md` fixtures rather than typed out again, so the
    two formats really are carrying the same words -- which is what lets the
    tests below assert that a note chunks and cites the same either way rather
    than merely that each format chunks into something.
    """
    blocks: list[tuple[str | None, str]] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if match := re.match(r"^(#{1,6})\s+(.*\S)\s*$", stripped):
            blocks.append((f"Heading {len(match.group(1))}", match.group(2)))
        else:
            blocks.append((BODY, stripped))
    return blocks


def _ingest_mixed(tmp_path):
    """One `network` folder holding the same two notes as `.md` and as `.docx`,
    ingested in a single run at the thresholds config.py ships.

    Four Documents from one walk, two per format. Kept in its own registry and
    store for the reason `_ingest_with_chinese` gives at length: the gate tests
    read distances off `_ingest_corpus`'s store, and Chinese Chunks sized for
    the configured thresholds pull the out-of-corpus trap inside GATE_TAU.
    """
    corpus = tmp_path / "corpus"
    folder = corpus / "network"
    folder.mkdir(parents=True)
    for source in sorted((FIXTURES / "network").glob("*.md")):
        markdown = source.read_text(encoding="utf-8")
        (folder / source.name).write_text(markdown, encoding="utf-8")
        write_docx(folder / f"{source.stem}.docx", _as_docx_blocks(markdown))

    registry = Registry(tmp_path / "documents.sqlite")
    store = VectorStore(path=tmp_path / "chroma")
    embedder = FakeEmbedder()
    report = ingest_folder(
        folder=folder,
        domain="network",
        source_type="note",
        registry=registry,
        store=store,
        embedder=embedder,
        corpus_root=corpus,
    )
    return registry, store, embedder, report


def test_a_mixed_format_folder_ingests_every_note_in_one_run(tmp_path):
    # One run, no flag naming a format: the walk routes each file by its own
    # suffix, so the corpus owner points ingestion at the folder their notes
    # are in and every readable note lands.
    registry, _store, _embedder, report = _ingest_mixed(tmp_path)

    assert set(report.ingested) == {
        HANDSHAKE_ID,
        OSI_ID,
        DOCX_HANDSHAKE_ID,
        DOCX_OSI_ID,
    }
    assert report.failed == []
    assert {doc.source_path for doc in registry.list()} == {
        "network/tcp_handshake.md",
        "network/tcp_handshake.docx",
        "network/osi_model.md",
        "network/osi_model.docx",
    }


def test_a_docx_note_chunks_exactly_as_the_same_note_in_markdown_does(tmp_path):
    # The structured path, asserted as one path rather than two: same headings,
    # same prose, same thresholds, so the same Locators over the same number of
    # Chunks. A docx reader that grew its own merge or split rules would show
    # up here even while every count it produced looked plausible on its own.
    _registry, store, _embedder, _report = _ingest_mixed(tmp_path)

    for markdown_id, docx_id in (
        (HANDSHAKE_ID, DOCX_HANDSHAKE_ID),
        (OSI_ID, DOCX_OSI_ID),
    ):
        markdown = store.collection.get(where={"doc_id": markdown_id})
        word = store.collection.get(where={"doc_id": docx_id})

        assert len(word["ids"]) == len(markdown["ids"])
        assert [m["locator"] for m in _in_ordinal_order(word)] == [
            m["locator"] for m in _in_ordinal_order(markdown)
        ]


def _in_ordinal_order(result):
    return [meta for meta, _ in sorted(
        zip(result["metadatas"], result["documents"]), key=lambda pair: pair[0]["ordinal"]
    )]


def test_a_docx_note_is_cited_by_its_heading_path_through_ask(tmp_path):
    # The acceptance criterion in the shape the reader meets it: a question
    # answered from a Word note comes back citing a heading path, not a page,
    # an offset or the file's name.
    _registry, store, embedder, _report = _ingest_mixed(tmp_path)

    answer = ask(
        ZH_HANDSHAKE_QUESTION,
        embedder,
        store,
        FakeGenerator(),
        top_k=5,
        distance_threshold=PASS_EVERYTHING,
        # One Chunk per Document, so the Word note contributes its best match
        # and the assertion below is about that Chunk rather than about
        # whichever of its several Chunks happened to come back first.
        max_chunks_per_document=1,
    )

    [from_docx] = [c for c in answer.evidence if c.doc_id == DOCX_HANDSHAKE_ID]
    assert from_docx.locator == "傳輸控制協定 › 三向交握"
    assert from_docx.domain == "network"
    assert from_docx.source_type == "note"


def test_a_docx_notes_oversized_section_splits_under_one_heading_path(tmp_path):
    # 三向交握 is the section sized to exceed MAX_SECTION_TOKENS, and a whitespace
    # split would never read it as oversized at all -- so this is the CJK token
    # measure and the split branch reached through the docx reader.
    _registry, store, _embedder, _report = _ingest_mixed(tmp_path)

    texts = _texts_under(store, DOCX_HANDSHAKE_ID, "傳輸控制協定 › 三向交握")

    assert len(texts) == 2
    assert "".join(texts).startswith("用戶端先送出 SYN 封包")
    assert not any("四次揮手" in text for text in texts)


def test_a_docx_notes_undersized_section_merges_into_its_neighbour(tmp_path):
    # 概述 is below MIN_SECTION_TOKENS, so it must not be citable on its own.
    _registry, store, _embedder, _report = _ingest_mixed(tmp_path)

    assert _texts_under(store, DOCX_OSI_ID, "OSI 參考模型 › 概述") == []
    [layering] = _texts_under(store, DOCX_OSI_ID, "OSI 參考模型 › 分層架構")
    assert "分層的目的是解耦" in layering


def test_a_malformed_docx_is_named_in_the_run_and_costs_the_run_nothing_else(tmp_path):
    # The run report contract from #4, over a format whose files fail in ways
    # a text note cannot: a truncated sync, a renamed zip, a file Word itself
    # would refuse to open.
    corpus = tmp_path / "corpus"
    folder = corpus / "network"
    folder.mkdir(parents=True)
    shutil.copy(FIXTURES / "network" / "osi_model.md", folder / "osi_model.md")
    # A package that opens and then does not parse, rather than bytes that are
    # not a zip at all: the shape that takes the whole walk down if the reader
    # lets its parser's own exception out, costing the corpus every note after
    # this one and the retirement sweep with them.
    corrupt_a_part(
        write_docx(tmp_path / "intact.docx", [("Heading 1", "傳輸控制協定"), (BODY, "三向交握。")]),
        folder / "tcp_handshake.docx",
    )

    registry = Registry(tmp_path / "documents.sqlite")
    report = ingest_folder(
        folder=folder,
        domain="network",
        source_type="note",
        registry=registry,
        store=VectorStore(path=tmp_path / "chroma"),
        embedder=FakeEmbedder(),
        corpus_root=corpus,
    )

    assert report.ingested == [OSI_ID]
    [failure] = report.failed
    assert failure.source_path == "network/tcp_handshake.docx"
    assert "tcp_handshake.docx" in failure.warning()
    assert report.summary() == "Ingested 1, skipped 0 unchanged, retired 0, failed 1."
    assert registry.get(failure.doc_id) is None


def test_reingesting_the_unchanged_mixed_corpus_skips_every_note(tmp_path):
    # A format whose skip does not hold re-embeds its half of the corpus every
    # run, which is invisible until the bill arrives.
    corpus = tmp_path / "corpus"
    folder = corpus / "network"
    folder.mkdir(parents=True)
    for source in sorted((FIXTURES / "network").glob("*.md")):
        markdown = source.read_text(encoding="utf-8")
        (folder / source.name).write_text(markdown, encoding="utf-8")
        write_docx(folder / f"{source.stem}.docx", _as_docx_blocks(markdown))
    registry = Registry(tmp_path / "documents.sqlite")
    store = VectorStore(path=tmp_path / "chroma")
    kwargs = dict(
        folder=folder,
        domain="network",
        source_type="note",
        registry=registry,
        store=store,
        embedder=FakeEmbedder(),
        corpus_root=corpus,
    )

    first = ingest_folder(**kwargs)
    second = ingest_folder(**kwargs)

    assert second.ingested == []
    assert sorted(second.skipped) == sorted(first.ingested)
