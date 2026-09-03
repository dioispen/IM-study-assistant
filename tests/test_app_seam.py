"""Seam tests for the page: type a question into the real app script and read
what a person sitting in front of it would see.

The project's second and final seam (#19). The app is driven through
Streamlit's own app-testing framework, over the fixture corpus in
tests/mixed_corpus.py, with the fake Embedder and a Generator this module wrote
-- so the whole suite runs offline, needs no API key and costs nothing.

Every assertion here is on what the page shows: which cards appeared, what they
name, what they open to, whether the question was visibly in flight. None is on
which functions the app called or in what order -- the app's internal structure
has to stay free to change without a red suite, and a test that reached for a
view model would be evidence that the fact under test belongs in the ask seam's
return value instead.

Nothing here asserts on generated text (ADR-0002): the answer the page renders
is prose this module chose, so what is checked is that the page carried it, not
what a model would have said.
"""

from pathlib import Path

import interface
from config import MAX_CHUNKS_PER_DOCUMENT, TOP_K
from core.ask import ask
from core.gate import ABSTENTION_TEXT
from core.registry import Registry
from core.store import VectorStore
from streamlit.testing.v1 import AppTest
from tests.mixed_corpus import (
    COVERED_QUESTION,
    GATE_TAU,
    OUT_OF_CORPUS_TRAP,
    ingest_mixed,
)

APP = str(Path(__file__).resolve().parent.parent / "app.py")

ANSWER = "A collision is two keys hashing into one bucket."

# The Evidence the covered question retrieves from the fixture corpus, as the
# reader meets it: Document title, Source type, Locator, in the order the cards
# are drawn. Pinned here rather than recomputed in each assertion, so a change
# to the fixtures or to chunking fails against a list someone can read and
# check against the notes on disk.
#
# Five, because TOP_K is five; two of them from hash_table.md rather than the
# four its sections would otherwise supply, because MAX_CHUNKS_PER_DOCUMENT is
# two. paging.md is nowhere in it, which is what the negative assertion below
# reads.
EXPECTED_CARDS = [
    ("hash table", "note", "Hash Table › Open Addressing"),
    ("hash table", "note", "Hash Table › Chaining"),
    ("hashing", "textbook", "Hashing › Hash Functions"),
    ("virtual memory", "textbook", "Virtual Memory › Translation Lookaside Buffer"),
    ("hashing", "textbook", "Hashing › Amortized Cost"),
]


class ScriptedGenerator:
    """Prose this module chose, so that "the page showed the answer" is a claim
    about the page rather than about a model (ADR-0002)."""

    def generate(self, prompt: str) -> str:
        return ANSWER


class ExplodingEmbedder:
    """Fails the test if a question is embedded -- what an empty corpus must be
    reported without."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("an empty corpus should be reported before embedding")


class ExplodingGenerator:
    """Fails the test if generation is reached."""

    def generate(self, prompt: str) -> str:
        raise AssertionError("nothing here should reach generation")


def _factory(tmp_path, generator=None):
    """The page's factory hook, over a corpus ingested into `tmp_path`.

    A factory that opens the corpus on every call, rather than one record built
    here and handed back, because that is what the page does for real: Streamlit
    runs every rerun on a thread of its own, and a SQLite connection belongs to
    the thread that opened it. It also says the thing the corpus claim is about
    -- the page opens the Document registry and the vector store the command
    line opens, rather than being handed a UI-shaped copy of them.

    τ travels with the embedder rather than being read from config, because it
    is a property of the embedding model (ADR-0003) and the model here is
    FakeEmbedder.
    """
    _registry, _store, embedder = ingest_mixed(tmp_path)
    generator = generator or ScriptedGenerator()

    def build_dependencies() -> interface.Dependencies:
        return interface.Dependencies(
            registry=Registry(tmp_path / "documents.sqlite"),
            store=VectorStore(path=tmp_path / "chroma"),
            embedder=embedder,
            generator=generator,
            distance_threshold=GATE_TAU,
        )

    return build_dependencies


def _empty_factory(tmp_path):
    """The same, over a registry and store never ingested into -- a fresh
    checkout, whose embedder and generator fail the test if either is reached."""

    def build_dependencies() -> interface.Dependencies:
        return interface.Dependencies(
            registry=Registry(tmp_path / "documents.sqlite"),
            store=VectorStore(path=tmp_path / "chroma"),
            embedder=ExplodingEmbedder(),
            generator=ExplodingGenerator(),
            distance_threshold=GATE_TAU,
        )

    return build_dependencies


def _open_page(monkeypatch, factory) -> AppTest:
    """The real app script, run with its one factory hook replaced.

    Replaced before the script runs, and on the module the script imports, which
    is the whole reason that module exists: Streamlit re-executes the script
    itself on every run, so a hook defined in the script could never be the one
    a test had substituted.
    """
    monkeypatch.setattr(interface, "build_dependencies", factory)
    page = AppTest.from_file(APP)
    page.run()
    return page


def _ask_on_the_page(page: AppTest, question: str) -> AppTest:
    return page.chat_input[0].set_value(question).run()


def _evidence_through_the_seam(tmp_path, question=COVERED_QUESTION):
    """What `ask` returns for `question` over the same corpus, under the same
    configured retrieval the page runs on.

    The cards are read against this rather than against a list retyped here, so
    what the assertions say is that the page and the pipeline agree -- a page
    that dropped a Chunk or added one of its own is red even if both it and a
    constant were changed together.
    """
    _registry, store, embedder = ingest_mixed(tmp_path)
    return ask(
        question,
        embedder,
        store,
        ScriptedGenerator(),
        top_k=TOP_K,
        distance_threshold=GATE_TAU,
        max_chunks_per_document=MAX_CHUNKS_PER_DOCUMENT,
    ).evidence


def _shown(page: AppTest) -> str:
    """Everything the page wrote in prose, as one string."""
    return "\n".join(
        [block.value for block in page.markdown]
        + [block.value for block in page.warning]
    )


def test_the_covered_question_and_the_trap_straddle_the_test_tau(tmp_path):
    # The gate cases this corpus is for only mean anything if the trap really is
    # the farther of a length-matched pair. Pin it, so a change to FakeEmbedder
    # or to the fixtures fails here rather than quietly making them vacuous.
    _registry, store, embedder = ingest_mixed(tmp_path)
    [covered], [trap] = embedder.embed([COVERED_QUESTION]), embedder.embed([OUT_OF_CORPUS_TRAP])

    covered_distance = store.query(embedding=covered, top_k=1)[0].distance
    trap_distance = store.query(embedding=trap, top_k=1)[0].distance

    assert len(COVERED_QUESTION.split()) == len(OUT_OF_CORPUS_TRAP.split())
    assert covered_distance < GATE_TAU < trap_distance


def test_a_question_typed_into_the_page_is_answered_on_it(monkeypatch, tmp_path):
    page = _open_page(monkeypatch, _factory(tmp_path))

    _ask_on_the_page(page, COVERED_QUESTION)

    assert page.exception == []
    assert ANSWER in _shown(page)


def test_the_page_draws_one_card_per_evidence_chunk(monkeypatch, tmp_path):
    page = _open_page(monkeypatch, _factory(tmp_path))

    _ask_on_the_page(page, COVERED_QUESTION)

    assert len(page.expander) == len(EXPECTED_CARDS)


def test_a_card_names_its_documents_title_source_type_and_locator(monkeypatch, tmp_path):
    # The three things the reader judges an answer's footing by, before opening
    # anything. What a card is worded like is presentation.py's business
    # (test_presentation.py); what is asserted here is that this surface shows
    # a card per Chunk and that each names its own Chunk's three.
    #
    # Named *in* the label rather than compared with it, because a label is the
    # card escaped for a Markdown widget (interface.card_label) -- the whole
    # list is compared, escaping and all, by the test below.
    page = _open_page(monkeypatch, _factory(tmp_path))

    _ask_on_the_page(page, COVERED_QUESTION)

    for card, (title, source_type, locator) in zip(page.expander, EXPECTED_CARDS, strict=True):
        assert title in card.label
        assert source_type in card.label
        assert locator in card.label


def test_a_card_expands_to_the_chunk_text_the_answer_was_given(monkeypatch, tmp_path):
    # Checking a suspicious claim should not mean opening the original file, so
    # the card opens onto the Chunk itself -- the text generation was handed,
    # not a summary of it or the Document around it.
    evidence = _evidence_through_the_seam(tmp_path)
    page = _open_page(monkeypatch, _factory(tmp_path))

    _ask_on_the_page(page, COVERED_QUESTION)

    opened = ["\n".join(block.value for block in card.text) for card in page.expander]
    assert opened == [chunk.text for chunk in evidence]


def test_the_cards_are_exactly_the_evidence_and_nothing_besides(monkeypatch, tmp_path):
    # A card is the only place a Document appears by name, so an extra one would
    # credit the answer to a Document generation was never given (CONTEXT.md:
    # Evidence is the only thing an answer's citations may point at). Read
    # against the Evidence the same question returns through the seam, so the
    # page is shown to add nothing to it and drop nothing from it.
    evidence = _evidence_through_the_seam(tmp_path)
    page = _open_page(monkeypatch, _factory(tmp_path))

    _ask_on_the_page(page, COVERED_QUESTION)

    assert [card.label for card in page.expander] == [
        interface.card_label(chunk) for chunk in evidence
    ]


def test_no_card_credits_a_document_the_evidence_does_not_hold(monkeypatch, tmp_path):
    # The corpus holds four Documents and the Evidence rests on three, so
    # paging.md is the one a page drawing from the corpus rather than from this
    # Answer's Evidence would name. Read off the card labels rather than off the
    # whole page, because one Chunk that *is* in the Evidence mentions paging in
    # its own text -- and what a card claims is whose Document backed the answer.
    page = _open_page(monkeypatch, _factory(tmp_path))

    _ask_on_the_page(page, COVERED_QUESTION)

    assert not any("paging" in card.label for card in page.expander)


def test_a_question_in_flight_is_visible_as_such(monkeypatch, tmp_path):
    # A slow retrieval must not read as a dead page. The framework inspects the
    # page once the run has finished, so what a test can hold onto is the status
    # the answering ran inside -- labelled with what is happening, and marked
    # complete only once the answer is on the page beneath it.
    page = _open_page(monkeypatch, _factory(tmp_path))

    _ask_on_the_page(page, COVERED_QUESTION)

    assert [status.label for status in page.status] == [interface.IN_FLIGHT_LABEL]
    assert page.status[0].state == "complete"


def test_a_page_with_no_question_yet_has_nothing_in_flight(monkeypatch, tmp_path):
    # The other half of the claim above: the status belongs to a question, not
    # to the page, so an untouched page is not permanently mid-thought.
    page = _open_page(monkeypatch, _factory(tmp_path))

    assert page.status == []
    assert page.expander == []
    assert page.chat_input != []


def test_an_empty_corpus_says_so_rather_than_abstaining(monkeypatch, tmp_path):
    # An empty corpus abstains at the gate, which is correct and unreadable: the
    # reader is told the corpus does not cover the question when what happened
    # is that there is no corpus. A fresh checkout should not look broken.
    page = _open_page(monkeypatch, _empty_factory(tmp_path))

    assert page.exception == []
    assert interface.EMPTY_CORPUS_NOTICE in _shown(page)
    assert ABSTENTION_TEXT not in _shown(page)


def test_an_empty_corpus_is_reported_before_a_question_is_embedded(monkeypatch, tmp_path):
    # Read off the Document registry, which is the authority on what Documents
    # exist (ADR-0001) -- not inferred by asking something and seeing what came
    # back. The embedder fails the test if it is reached, so an implementation
    # that probed the corpus with a question would be red here even though its
    # page said the same thing.
    page = _open_page(monkeypatch, _empty_factory(tmp_path))

    assert page.exception == []
    assert page.chat_input == []
