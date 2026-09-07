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

from dataclasses import replace
from pathlib import Path

import interface
import presentation
import pytest
from config import DISTANCE_THRESHOLD, DOMAINS, MAX_CHUNKS_PER_DOCUMENT, TOP_K
from core.ask import ask
from core.gate import ABSTENTION_TEXT
from core.generator import BACKSTOP_ABSTENTION_TEXT, BACKSTOP_SENTINEL
from core.openai_client import ProviderFailure
from core.registry import Registry
from core.store import ChunkFilter, VectorStore
from presentation import describe_filter, provisional_threshold_notice
from streamlit.testing.v1 import AppTest
from tests.mixed_corpus import (
    COVERED_QUESTION,
    GATE_TAU,
    OUT_OF_CORPUS_TRAP,
    SECOND_COVERED_QUESTION,
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
    ("hash table", "dsa", "note", "Hash Table › Open Addressing"),
    ("hash table", "dsa", "note", "Hash Table › Chaining"),
    ("hashing", "dsa", "textbook", "Hashing › Hash Functions"),
    ("virtual memory", "os", "textbook", "Virtual Memory › Translation Lookaside Buffer"),
    ("hashing", "dsa", "textbook", "Hashing › Amortized Cost"),
]

# The corpus tests/mixed_corpus.py ingests, as the sidebar summarises it: four
# Documents, two per Domain, across two Source types. Two Documents rather than
# the seven Chunks the dsa half chunks into, which is what a summary read off
# the vector store instead of the registry would say.
EXPECTED_DOCUMENTS_PER_DOMAIN = {"dsa": 2, "os": 2}
EXPECTED_SOURCE_TYPES = ["note", "textbook"]

# A Source type the corpus does not hold. Source type is an open set
# (CONTEXT.md), so a menu that offered this would be a menu built from
# somewhere other than the registry.
UNINGESTED_SOURCE_TYPE = "paper"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """The fixture corpus, ingested once for every test in this module.

    One directory rather than one per test, because Chroma holds a client per
    store path for as long as the process lives and nothing here ever closes
    one. A corpus per test was therefore a client per test, and the suite ran
    the process out of open file handles -- in a later module, on a PDF
    fixture, with nothing to say which test had spent them.

    Shared for the whole module and so shared mutably, which is safe only
    because nothing here writes to it: every test asks questions of the same
    four Documents, and the re-ingestion `_factory` runs is a no-op skip
    against an unchanged registry (ADR-0001). A test that ingested a fifth
    Document would be leaving it for the ones after it, and wants its own
    directory.

    The empty-corpus cases keep a directory of their own, for the obvious
    reason: what they are about is a registry nothing was ever ingested into.
    """
    return tmp_path_factory.mktemp("corpus")


def _nth_answer(n: int) -> str:
    """The nth answer ScriptedGenerator gives, as it reaches the page."""
    return f"{ANSWER} ({n})"


class ScriptedGenerator:
    """Prose this module chose, so that "the page showed the answer" is a claim
    about the page rather than about a model (ADR-0002).

    Numbered, one line per question it is given, so that an answer on screen
    says which generation produced it. That is what lets a session be read: two
    turns are two answers rather than one answer counted twice, and an answer
    fetched again while a kept turn was being redrawn would arrive wearing a
    number the reader never saw.
    """

    def __init__(self):
        self._answered = 0

    def generate(self, prompt: str) -> str:
        self._answered += 1
        return _nth_answer(self._answered)


class ExplodingEmbedder:
    """Fails the test if a question is embedded -- what an empty corpus must be
    reported without."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("an empty corpus should be reported before embedding")


class ExplodingGenerator:
    """Fails the test if generation is reached."""

    def generate(self, prompt: str) -> str:
        raise AssertionError("nothing here should reach generation")


def _factory(corpus, generator=None):
    """The page's factory hook, over a corpus ingested into `corpus`.

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
    _registry, _store, embedder = ingest_mixed(corpus)
    generator = generator or ScriptedGenerator()

    def build_dependencies() -> interface.Dependencies:
        return interface.Dependencies(
            registry=Registry(corpus / "documents.sqlite"),
            store=VectorStore(path=corpus / "chroma"),
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


def _evidence_through_the_seam(
    corpus,
    question=COVERED_QUESTION,
    chunk_filter=None,
    top_k=TOP_K,
    max_chunks_per_document=MAX_CHUNKS_PER_DOCUMENT,
):
    """What `ask` returns for `question` over the same corpus, under the
    retrieval the page was left set to.

    The cards are read against this rather than against a list retyped here, so
    what the assertions say is that the page and the pipeline agree -- a page
    that dropped a Chunk or added one of its own is red even if both it and a
    constant were changed together.

    The two retrieval arguments default to the configured values, which is what
    the page defaults its own controls to (#27): a test that turns neither knob
    passes nothing about them, and one that turns a knob on the page passes the
    same number here.
    """
    _registry, store, embedder = ingest_mixed(corpus)
    return ask(
        question,
        embedder,
        store,
        ScriptedGenerator(),
        top_k=top_k,
        distance_threshold=GATE_TAU,
        chunk_filter=chunk_filter,
        max_chunks_per_document=max_chunks_per_document,
    ).evidence


def _shown(page: AppTest) -> str:
    """Everything the page wrote in prose, as one string."""
    return "\n".join(
        [block.value for block in page.markdown]
        + [block.value for block in page.warning]
        + [block.value for block in page.info]
    )


def test_the_covered_question_and_the_trap_straddle_the_test_tau(corpus):
    # The gate cases this corpus is for only mean anything if the trap really is
    # the farther of a length-matched pair. Pin it, so a change to FakeEmbedder
    # or to the fixtures fails here rather than quietly making them vacuous.
    _registry, store, embedder = ingest_mixed(corpus)
    [covered], [trap] = embedder.embed([COVERED_QUESTION]), embedder.embed([OUT_OF_CORPUS_TRAP])

    covered_distance = store.query(embedding=covered, top_k=1)[0].distance
    trap_distance = store.query(embedding=trap, top_k=1)[0].distance

    assert len(COVERED_QUESTION.split()) == len(OUT_OF_CORPUS_TRAP.split())
    assert covered_distance < GATE_TAU < trap_distance


def test_a_question_typed_into_the_page_is_answered_on_it(monkeypatch, corpus):
    page = _open_page(monkeypatch, _factory(corpus))

    _ask_on_the_page(page, COVERED_QUESTION)

    assert page.exception == []
    assert ANSWER in _shown(page)


def test_the_page_draws_one_card_per_evidence_chunk(monkeypatch, corpus):
    page = _open_page(monkeypatch, _factory(corpus))

    _ask_on_the_page(page, COVERED_QUESTION)

    assert len(page.expander) == len(EXPECTED_CARDS)


def test_a_card_names_its_documents_title_domain_source_type_and_locator(
    monkeypatch, corpus
):
    # The four things the reader judges an answer's footing by, before opening
    # anything. The Domain among them, so a question asked under no Domain
    # restriction shows which subject each card was read from -- an ambiguous
    # term like "tree" is the case that is for.
    #
    # What a card is worded like is presentation.py's business
    # (test_presentation.py); what is asserted here is that this surface shows
    # a card per Chunk and that each names its own Chunk's four.
    #
    # Named *in* the label rather than compared with it, because a label is the
    # card escaped for a Markdown widget (interface.card_label) -- the whole
    # list is compared, escaping and all, by the test below.
    page = _open_page(monkeypatch, _factory(corpus))

    _ask_on_the_page(page, COVERED_QUESTION)

    for card, (title, domain, source_type, locator) in zip(
        page.expander, EXPECTED_CARDS, strict=True
    ):
        assert title in card.label
        assert domain in card.label
        assert source_type in card.label
        assert locator in card.label


def test_a_card_expands_to_the_chunk_text_the_answer_was_given(monkeypatch, corpus):
    # Checking a suspicious claim should not mean opening the original file, so
    # the card opens onto the Chunk itself -- the text generation was handed,
    # not a summary of it or the Document around it.
    evidence = _evidence_through_the_seam(corpus)
    page = _open_page(monkeypatch, _factory(corpus))

    _ask_on_the_page(page, COVERED_QUESTION)

    opened = ["\n".join(block.value for block in card.text) for card in page.expander]
    assert opened == [chunk.text for chunk in evidence]


def test_the_cards_are_exactly_the_evidence_and_nothing_besides(monkeypatch, corpus):
    # A card is the only place a Document appears by name, so an extra one would
    # credit the answer to a Document generation was never given (CONTEXT.md:
    # Evidence is the only thing an answer's citations may point at). Read
    # against the Evidence the same question returns through the seam, so the
    # page is shown to add nothing to it and drop nothing from it.
    evidence = _evidence_through_the_seam(corpus)
    page = _open_page(monkeypatch, _factory(corpus))

    _ask_on_the_page(page, COVERED_QUESTION)

    assert [card.label for card in page.expander] == [
        interface.card_label(chunk) for chunk in evidence
    ]


def test_no_card_credits_a_document_the_evidence_does_not_hold(monkeypatch, corpus):
    # The corpus holds four Documents and the Evidence rests on three, so
    # paging.md is the one a page drawing from the corpus rather than from this
    # Answer's Evidence would name. Read off the card labels rather than off the
    # whole page, because one Chunk that *is* in the Evidence mentions paging in
    # its own text -- and what a card claims is whose Document backed the answer.
    page = _open_page(monkeypatch, _factory(corpus))

    _ask_on_the_page(page, COVERED_QUESTION)

    assert not any("paging" in card.label for card in page.expander)


def test_a_question_in_flight_is_visible_as_such(monkeypatch, corpus):
    # A slow retrieval must not read as a dead page. The framework inspects the
    # page once the run has finished, so what a test can hold onto is the status
    # the answering ran inside -- labelled with what is happening, and marked
    # complete only once the answer is on the page beneath it.
    page = _open_page(monkeypatch, _factory(corpus))

    _ask_on_the_page(page, COVERED_QUESTION)

    assert [status.label for status in page.status] == [interface.IN_FLIGHT_LABEL]
    assert page.status[0].state == "complete"


def test_a_page_with_no_question_yet_has_nothing_in_flight(monkeypatch, corpus):
    # The other half of the claim above: the status belongs to a question, not
    # to the page, so an untouched page is not permanently mid-thought.
    page = _open_page(monkeypatch, _factory(corpus))

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


def test_the_second_covered_question_reads_its_own_evidence(corpus):
    # The session cases below only mean anything if the two questions really do
    # retrieve different Evidence, and if the second is covered at all. Pin
    # both, so a change to the fixtures or to FakeEmbedder fails here rather
    # than quietly making a two-turn assertion a one-turn one.
    _registry, store, embedder = ingest_mixed(corpus)
    [second] = embedder.embed([SECOND_COVERED_QUESTION])

    assert store.query(embedding=second, top_k=1)[0].distance < GATE_TAU
    assert _evidence_through_the_seam(corpus, SECOND_COVERED_QUESTION) != (
        _evidence_through_the_seam(corpus)
    )


def test_asking_twice_leaves_both_turns_on_screen(monkeypatch, corpus):
    # A revision session rather than a series of shell invocations: the first
    # question and its answer are still there when the second arrives, so a
    # follow-up costs nothing but the thought behind it.
    page = _open_page(monkeypatch, _factory(corpus))

    _ask_on_the_page(page, COVERED_QUESTION)
    _ask_on_the_page(page, SECOND_COVERED_QUESTION)

    assert page.exception == []
    assert COVERED_QUESTION in _shown(page)
    assert SECOND_COVERED_QUESTION in _shown(page)
    assert _shown(page).count(ANSWER) == 2


def test_each_turn_keeps_the_cards_of_its_own_question(monkeypatch, corpus):
    # What a turn shows is read off the turn, not off the session: the cards
    # standing under the first answer are still the first question's Evidence
    # after the second question has been asked and answered above -- neither
    # redrawn from the newer question nor merged with it.
    #
    # Read against what the seam returns for each question on its own, so the
    # claim is that each turn shows its own Evidence entire, in order.
    first = _evidence_through_the_seam(corpus)
    second = _evidence_through_the_seam(corpus, SECOND_COVERED_QUESTION)
    page = _open_page(monkeypatch, _factory(corpus))

    _ask_on_the_page(page, COVERED_QUESTION)
    _ask_on_the_page(page, SECOND_COVERED_QUESTION)

    assert [card.label for card in page.expander] == [
        interface.card_label(chunk) for chunk in first + second
    ]


def test_a_second_question_is_retrieved_as_if_it_were_the_first(monkeypatch, corpus):
    # The session is display-only: what scrolls above is not folded into the
    # query.
    # The trap is the case that shows it -- asked cold it abstains at the gate,
    # and asked after a covered question it has to abstain the same way. A
    # session that appended the first question to the second would be dragging
    # the trap towards the corpus the first question hit, and would answer it.
    #
    # The first turn's cards are still the only cards on the page, which is the
    # other half of the same claim: a gate Abstention cites nothing (ADR-0003),
    # so the second turn contributes none.
    page = _open_page(monkeypatch, _factory(corpus))

    _ask_on_the_page(page, COVERED_QUESTION)
    _ask_on_the_page(page, OUT_OF_CORPUS_TRAP)

    assert page.exception == []
    assert ABSTENTION_TEXT in _shown(page)
    assert len(page.expander) == len(EXPECTED_CARDS)


def test_a_clear_control_empties_the_session(monkeypatch, corpus):
    # Starting a new topic without scrolling past the last one. What is gone is
    # what was on screen -- the question, its answer and its cards -- and what
    # stays is the page itself, ready for the next question.
    page = _open_page(monkeypatch, _factory(corpus))
    _ask_on_the_page(page, COVERED_QUESTION)

    page.button[0].click().run()

    assert page.exception == []
    assert COVERED_QUESTION not in _shown(page)
    assert ANSWER not in _shown(page)
    assert page.expander == []
    assert page.chat_input != []


def test_a_kept_turn_still_shows_the_answer_it_was_given(monkeypatch, corpus):
    # Drawing a turn is a pure function of the record it kept, and the numbering
    # is what tells the two apart: a first turn redrawn from anything but its own
    # record -- re-asked at display time, or re-answered -- comes back wearing a
    # number the reader never saw.
    #
    # This is the claim #25 rests on. Once the sidebar can move the Scope
    # mid-session, a turn re-derived while being redrawn is a turn relabelled
    # retroactively, and nothing on the page would say so.
    page = _open_page(monkeypatch, _factory(corpus))

    _ask_on_the_page(page, COVERED_QUESTION)
    _ask_on_the_page(page, SECOND_COVERED_QUESTION)

    assert _nth_answer(1) in _shown(page)
    assert _nth_answer(2) in _shown(page)
    assert _nth_answer(3) not in _shown(page)


def _scope_the_session(page: AppTest, **restrictions) -> AppTest:
    """The sidebar as a reader sets it: a restriction picked from a menu.

    Keyed by the ChunkFilter field each menu fills, which is what the widgets
    are keyed by -- a test that named the widgets anything else would be
    describing a translation step the page does not have.
    """
    for axis, value in restrictions.items():
        page.selectbox(axis).select(value).run()
    return page


def test_each_restriction_narrows_the_covered_question_to_its_own_evidence(corpus):
    # The sidebar cases below only mean anything if each restriction actually
    # changes what comes back, and if the question stays covered under every one
    # of them -- a scope that abstained at the gate would make "the cards
    # narrowed" true of an empty page. Pin both, so a fixture edit fails here
    # rather than quietly making the assertions vacuous.
    unscoped = _evidence_through_the_seam(corpus)
    dsa = _evidence_through_the_seam(corpus, chunk_filter=ChunkFilter(domain="dsa"))
    notes = _evidence_through_the_seam(
        corpus, chunk_filter=ChunkFilter(source_type="note")
    )
    dsa_notes = _evidence_through_the_seam(
        corpus, chunk_filter=ChunkFilter(domain="dsa", source_type="note")
    )

    assert all(evidence for evidence in (unscoped, dsa, notes, dsa_notes))
    assert unscoped != dsa != notes != unscoped
    # The conjunction is narrower than either axis alone, which is the fact a
    # filter reading its two arguments as a disjunction could not produce.
    assert len(dsa_notes) < len(dsa)
    assert len(dsa_notes) < len(notes)


def test_the_domain_menu_offers_the_configured_domains(monkeypatch, corpus):
    # Domains are declared configuration -- every Document has exactly one, and
    # which ones exist is a decision rather than an observation (CONTEXT.md). So
    # the menu is DOMAINS entire, including the ones nothing has been ingested
    # under yet, which is what makes the corpus summary beside it worth reading.
    page = _open_page(monkeypatch, _factory(corpus))

    assert page.selectbox(interface.DOMAIN_KEY).options == [
        interface.ANY_DOMAIN,
        *DOMAINS,
    ]


def test_the_source_type_menu_offers_what_the_registry_holds(monkeypatch, corpus):
    # Source type is an open set (CONTEXT.md, PLAN.md 2.3): there is no
    # configured list of them, so the menu is built from the Document registry's
    # actual values -- which is what makes it become a real choice the moment a
    # second Source type is ingested, with no code change. A menu carrying a
    # value nobody ingested would be a menu built from a constant.
    page = _open_page(monkeypatch, _factory(corpus))

    menu = page.selectbox(interface.SOURCE_TYPE_KEY)

    assert menu.options == [interface.ANY_SOURCE_TYPE, *EXPECTED_SOURCE_TYPES]
    assert UNINGESTED_SOURCE_TYPE not in menu.options


def test_a_domain_restriction_narrows_the_cards_to_that_domain(monkeypatch, corpus):
    # The sidebar is a front end for the filtered retrieval built in #5 and
    # reimplements none of it, so what the page shows is read against what the
    # seam returns under the same ChunkFilter.
    scoped = _evidence_through_the_seam(corpus, chunk_filter=ChunkFilter(domain="dsa"))
    page = _open_page(monkeypatch, _factory(corpus))

    _scope_the_session(page, domain="dsa")
    _ask_on_the_page(page, COVERED_QUESTION)

    assert page.exception == []
    assert [card.label for card in page.expander] == [
        interface.card_label(chunk) for chunk in scoped
    ]


def test_a_source_type_restriction_narrows_the_cards_on_its_own_axis(
    monkeypatch, corpus
):
    # "What do my notes say" as a question that can actually be asked, and
    # independently of the Domain menu: the os notes stay eligible.
    scoped = _evidence_through_the_seam(
        corpus, chunk_filter=ChunkFilter(source_type="note")
    )
    page = _open_page(monkeypatch, _factory(corpus))

    _scope_the_session(page, source_type="note")
    _ask_on_the_page(page, COVERED_QUESTION)

    assert page.exception == []
    assert [card.label for card in page.expander] == [
        interface.card_label(chunk) for chunk in scoped
    ]


def test_both_restrictions_together_narrow_by_conjunction(monkeypatch, corpus):
    # "My own dsa notes", which is the narrowest thing the sidebar can say. Read
    # as a conjunction rather than a disjunction: a page that combined them the
    # other way would reach the dsa textbook and the os notes as well, and the
    # composition assertion below is what says it did not.
    scoped = _evidence_through_the_seam(
        corpus, chunk_filter=ChunkFilter(domain="dsa", source_type="note")
    )
    page = _open_page(monkeypatch, _factory(corpus))

    _scope_the_session(page, domain="dsa", source_type="note")
    _ask_on_the_page(page, COVERED_QUESTION)

    assert page.exception == []
    assert [card.label for card in page.expander] == [
        interface.card_label(chunk) for chunk in scoped
    ]
    assert {(chunk.domain, chunk.source_type) for chunk in scoped} == {("dsa", "note")}


def test_a_restriction_stays_in_force_until_it_is_changed(monkeypatch, corpus):
    # Scoping a whole revision session is one action rather than one per
    # question: the menu is set once and the second question is answered under
    # it, with nothing touched in between.
    scoped = _evidence_through_the_seam(corpus, chunk_filter=ChunkFilter(domain="dsa"))
    page = _open_page(monkeypatch, _factory(corpus))

    _scope_the_session(page, domain="dsa")
    _ask_on_the_page(page, COVERED_QUESTION)
    _ask_on_the_page(page, COVERED_QUESTION)

    assert page.exception == []
    assert [card.label for card in page.expander] == [
        interface.card_label(chunk) for chunk in scoped + scoped
    ]


def test_a_turn_says_which_scope_it_was_answered_under(monkeypatch, corpus):
    # Scrolling back through a session where the filter moved must not be
    # misleading, so every turn carries the Scope it was given -- read off the
    # Answer, which is where the pipeline put it (CONTEXT.md: a Scope travels
    # with the Answer it produced).
    page = _open_page(monkeypatch, _factory(corpus))

    _scope_the_session(page, domain="dsa", source_type="note")
    _ask_on_the_page(page, COVERED_QUESTION)

    assert [caption.value for caption in page.caption] == [
        interface.scope_caption(ChunkFilter(domain="dsa", source_type="note"))
    ]


def test_an_unrestricted_turn_says_it_was_answered_from_the_whole_corpus(
    monkeypatch, corpus
):
    # The unrestricted Scope is named rather than left blank. A turn with
    # nothing said about its Scope reads as a turn whose Scope was not recorded,
    # which is exactly the ambiguity the caption exists to remove -- and the
    # whole Corpus is one of the Scopes CONTEXT.md defines, not the absence of
    # one.
    page = _open_page(monkeypatch, _factory(corpus))

    _ask_on_the_page(page, COVERED_QUESTION)

    assert [caption.value for caption in page.caption] == [
        interface.scope_caption(ChunkFilter())
    ]


def test_changing_the_scope_leaves_the_turn_already_answered_untouched(
    monkeypatch, corpus
):
    # A filter change is a setting for the next question and never a rewrite of
    # the last one. The first turn keeps the cards it was answered with and goes
    # on saying it was answered from the whole corpus, while the second is
    # answered under the restriction now in force -- which is only true because
    # a turn is drawn from its own record (#24).
    unscoped = _evidence_through_the_seam(corpus)
    scoped = _evidence_through_the_seam(corpus, chunk_filter=ChunkFilter(domain="dsa"))
    page = _open_page(monkeypatch, _factory(corpus))
    _ask_on_the_page(page, COVERED_QUESTION)

    _scope_the_session(page, domain="dsa")

    assert [card.label for card in page.expander] == [
        interface.card_label(chunk) for chunk in unscoped
    ]

    _ask_on_the_page(page, COVERED_QUESTION)

    assert page.exception == []
    assert [card.label for card in page.expander] == [
        interface.card_label(chunk) for chunk in unscoped + scoped
    ]
    assert [caption.value for caption in page.caption] == [
        interface.scope_caption(ChunkFilter()),
        interface.scope_caption(ChunkFilter(domain="dsa")),
    ]


def test_the_sidebar_summarises_the_corpus_it_is_scoping(monkeypatch, corpus):
    # Knowing what you are querying before you blame retrieval. Read off the
    # Document registry, which is the authority on what Documents exist
    # (ADR-0001): the counts are Documents rather than the Chunks the vector
    # store holds a particular chunking into.
    page = _open_page(monkeypatch, _factory(corpus))

    summary = "\n".join(block.value for block in page.sidebar.markdown)

    for domain, documents in EXPECTED_DOCUMENTS_PER_DOMAIN.items():
        assert interface.domain_count_line(domain, documents) in summary
    assert interface.source_types_line(EXPECTED_SOURCE_TYPES) in summary


class AbstainingGenerator:
    """Declares the prompt backstop the one way the prompt asks for it: the
    sentinel on its own line and nothing else (ADR-0008).

    Not prose that sounds like giving up -- a model losing confidence in its
    own words declares nothing, and its answer is an ordinary answer
    (core/ask.py). So this double is the contract itself, and what the page is
    shown to render is what the sentinel became rather than anything about real
    model prose (ADR-0002).
    """

    def generate(self, prompt: str) -> str:
        return BACKSTOP_SENTINEL


def test_a_gate_abstention_shows_no_cards_and_never_reaches_generation(
    monkeypatch, corpus
):
    # The gate abstains before generation is given anything, so there is
    # nothing to cite -- and nothing to pay for. The generator fails the test
    # if it is called, which is what makes "a hopeless question costs zero
    # generation tokens" a claim about this surface rather than one inherited
    # from the command line: moving to a page must not quietly start paying for
    # questions the corpus cannot answer.
    page = _open_page(monkeypatch, _factory(corpus, generator=ExplodingGenerator()))

    _ask_on_the_page(page, OUT_OF_CORPUS_TRAP)

    assert page.exception == []
    assert page.expander == []
    assert [block.value for block in page.warning] == [ABSTENTION_TEXT]


def test_a_scoped_gate_abstention_names_the_restriction_it_was_asked_under(
    monkeypatch, corpus
):
    # The abstention text speaks of "the corpus", and a scoped question
    # abstained on the corpus it was scoped to -- so the narrowing has to be on
    # screen beside it, or "your notes don't cover this" reads the same as "not
    # under that Domain" when the two ask different things of the reader.
    #
    # Asserted as the restriction's wording standing beside the abstention
    # rather than as one particular widget carrying it: what the page owes the
    # reader is that the narrowing is visible, and the turn's own Scope caption
    # (#25) is where this page says it.
    page = _open_page(monkeypatch, _factory(corpus, generator=ExplodingGenerator()))

    _scope_the_session(page, domain="dsa")
    _ask_on_the_page(page, OUT_OF_CORPUS_TRAP)

    assert page.exception == []
    restriction = describe_filter(ChunkFilter(domain="dsa"))
    assert any(restriction in caption.value for caption in page.caption)


def test_a_backstop_abstention_keeps_the_evidence_it_was_judged_from(
    monkeypatch, corpus
):
    # The backstop fired over Evidence that was assembled and handed over, so
    # the cards are exactly the material the model judged insufficient -- and
    # this is the one case where the reader wants to look at that Evidence
    # themselves and disagree (#21). Read against what the seam returns for the
    # same question, so the page is shown to keep all of it and add nothing.
    evidence = _evidence_through_the_seam(corpus)
    page = _open_page(monkeypatch, _factory(corpus, generator=AbstainingGenerator()))

    _ask_on_the_page(page, COVERED_QUESTION)

    assert page.exception == []
    assert BACKSTOP_ABSTENTION_TEXT in _shown(page)
    assert [card.label for card in page.expander] == [
        interface.card_label(chunk) for chunk in evidence
    ]


def test_a_backstop_abstention_shows_the_reader_no_contract_token(monkeypatch, corpus):
    # The sentinel is a contract between the prompt and the pipeline and never
    # leaves it (core/ask.py). A page showing it would be a page that had read
    # generated text to work out what happened.
    page = _open_page(monkeypatch, _factory(corpus, generator=AbstainingGenerator()))

    _ask_on_the_page(page, COVERED_QUESTION)

    assert BACKSTOP_SENTINEL not in _shown(page)


def test_the_two_abstentions_are_told_apart_by_more_than_their_wording(
    monkeypatch, corpus
):
    # ADR-0003's two layers, made observable on the page: one branch, opposite
    # renderings. The gate's abstention is a warning with nothing beneath it;
    # the backstop's is an informational note with its cards intact. A reader
    # tells them apart by the shape of the turn rather than by reading either
    # text closely -- and a two-layer design nobody can observe on screen is a
    # layer nobody can tell is broken.
    gated = _open_page(monkeypatch, _factory(corpus, generator=ExplodingGenerator()))
    _ask_on_the_page(gated, OUT_OF_CORPUS_TRAP)
    gated_warnings = [block.value for block in gated.warning]
    gated_notes = [block.value for block in gated.info]
    gated_cards = list(gated.expander)

    backstopped = _open_page(
        monkeypatch, _factory(corpus, generator=AbstainingGenerator())
    )
    _ask_on_the_page(backstopped, COVERED_QUESTION)

    assert gated_warnings == [ABSTENTION_TEXT]
    assert gated_notes == []
    assert gated_cards == []

    assert [block.value for block in backstopped.info] == [BACKSTOP_ABSTENTION_TEXT]
    assert backstopped.warning == []
    assert backstopped.expander != []


def test_an_ordinary_answer_is_not_dressed_as_an_abstention(monkeypatch, corpus):
    # The other half of the claim above: the callouts belong to the two
    # Abstentions, so an answer the system stands behind is not framed as one.
    # A gate Abstention that read as a weak answer, and an answer that read as
    # an Abstention, both leave the reader unable to tell what the system did.
    page = _open_page(monkeypatch, _factory(corpus))

    _ask_on_the_page(page, COVERED_QUESTION)

    assert ANSWER in _shown(page)
    assert page.warning == []
    assert page.info == []


def test_the_provisional_tau_notice_stands_beside_a_gate_abstention(monkeypatch, corpus):
    # An Abstention is the one moment an underived tau visibly costs the reader
    # an answer, so that is where the gate admits it is uncalibrated. What the
    # notice says is presentation.py's business (test_presentation.py); what is
    # asserted here is that this surface shows it, as the aside it is, and has
    # nothing of its own to say once there is no notice to show.
    #
    # That an ordinary answer carries no notice is already said by the exact
    # caption lists in the Scope tests above.
    page = _open_page(monkeypatch, _factory(corpus, generator=ExplodingGenerator()))

    _ask_on_the_page(page, OUT_OF_CORPUS_TRAP)

    captions = [caption.value for caption in page.caption]
    notice = provisional_threshold_notice()
    if notice is None:
        assert not any("tau=" in caption for caption in captions)
    else:
        assert interface.as_written(notice) in captions


def test_a_derived_tau_leaves_this_page_with_no_notice_to_show(monkeypatch, corpus):
    # The Week 6 sweep flipping the flag in config.py has to be the only edit
    # (#22), so the notice is read off the configured threshold rather than
    # restated here -- a literal copy of today's wording would survive this and
    # outlive the condition it describes. The Abstention itself is untouched:
    # what goes is the admission, not the refusal, and what is left beside it is
    # the turn's Scope and nothing else.
    monkeypatch.setattr(
        presentation,
        "DISTANCE_THRESHOLD",
        replace(DISTANCE_THRESHOLD, provisional=False),
    )
    page = _open_page(monkeypatch, _factory(corpus, generator=ExplodingGenerator()))

    _ask_on_the_page(page, OUT_OF_CORPUS_TRAP)

    assert page.exception == []
    assert [block.value for block in page.warning] == [ABSTENTION_TEXT]
    assert [caption.value for caption in page.caption] == [
        interface.scope_caption(ChunkFilter())
    ]


# What each control is turned to when a test turns it. Both are away from the
# configured default in the direction that narrows the Evidence, because that
# is the change a reader can count on the page: fewer cards, or cards drawn
# from more Documents.
FEWER_CHUNKS = 3
ONE_PER_DOCUMENT = 1

# A model DISTANCE_THRESHOLD was not set for. Any name but the configured one
# would do; a real sibling model is used because the mistake this guards
# against is a real one -- an embedding model swapped in config.py with tau
# left behind (ADR-0003).
OTHER_EMBEDDING_MODEL = "text-embedding-3-large"

# The provider's own words in the two operational failures this page has to
# render. Taken from what the OpenAI SDK says rather than invented, because
# what the page must not do is replace them: the missing-key text is the one
# that tells the reader which environment variable to set, and it is
# underscored either side.
MISSING_KEY_DETAIL = (
    "The api_key client option must be set either by passing api_key to the "
    "client or by setting the OPENAI_API_KEY environment variable"
)
PROVIDER_ERROR_DETAIL = "Connection error."


class FailingGenerator:
    """A provider that will not answer: the operational failure an outage, a
    rate limit or a revoked key arrives as."""

    def generate(self, prompt: str) -> str:
        raise ProviderFailure(PROVIDER_ERROR_DETAIL)


class FailsOnceGenerator:
    """Fails the first question and answers every one after it -- a hiccup
    rather than a page that can never work, which is what lets a test ask what
    the session kept afterwards."""

    def __init__(self):
        self._scripted = ScriptedGenerator()
        self._failed = False

    def generate(self, prompt: str) -> str:
        if not self._failed:
            self._failed = True
            raise ProviderFailure(PROVIDER_ERROR_DETAIL)
        return self._scripted.generate(prompt)


def _unreachable_provider_factory(detail):
    """A factory that cannot build its dependencies at all.

    The missing-key case: the client is constructed before any question is
    typed, so this failure arrives while the page is being opened rather than
    while an answer is being produced -- and the page has to survive it with
    something on screen.
    """

    def build_dependencies() -> interface.Dependencies:
        raise ProviderFailure(detail)

    return build_dependencies


def _misconfigured_tau_factory():
    """A factory whose tau was set for another embedding model.

    Runs the real assertion out of config.py rather than raising something
    shaped like it, so the test below goes red the day that assertion is
    softened into a caught failure.
    """

    def build_dependencies() -> interface.Dependencies:
        DISTANCE_THRESHOLD.for_model(OTHER_EMBEDDING_MODEL)
        raise AssertionError("the threshold pairing should have failed first")

    return build_dependencies


def _set_retrieval(page: AppTest, **controls) -> AppTest:
    """The advanced controls as a reader turns them.

    Keyed by the `ask` parameter each control fills, exactly as the Scope menus
    are keyed by the ChunkFilter field they fill -- so a test that named a
    widget anything else would be describing a translation step the page does
    not have.
    """
    for parameter, value in controls.items():
        page.number_input(parameter).set_value(value).run()
    return page


def test_the_advanced_controls_default_to_the_configured_retrieval(monkeypatch, corpus):
    # A page opened and not touched answers exactly as the command line does
    # with no flags: the controls are a way to depart from the configuration,
    # not a second place the configuration is written down.
    page = _open_page(monkeypatch, _factory(corpus))

    assert page.number_input(interface.TOP_K_KEY).value == TOP_K
    assert (
        page.number_input(interface.MAX_PER_DOCUMENT_KEY).value
        == MAX_CHUNKS_PER_DOCUMENT
    )


def test_the_controls_can_be_turned_to_every_configured_value():
    # A ceiling beneath a value a control defaults to is a page that cannot
    # draw its own defaults: Streamlit refuses a number_input whose value is
    # above its max rather than clamping it, so the page would die on a
    # traceback before anything else on it was reached. The cap is the one that
    # bites -- a cap well above top-k is the non-binding baseline the Week 7
    # sweep reads against (PLAN.md §第 7 週), and config.py is where it would be
    # set.
    assert interface.MAX_RETRIEVED >= TOP_K
    assert interface.MAX_RETRIEVED >= MAX_CHUNKS_PER_DOCUMENT
    assert interface.MIN_RETRIEVED <= min(TOP_K, MAX_CHUNKS_PER_DOCUMENT)


def test_asking_for_fewer_chunks_shows_fewer_cards(monkeypatch, corpus):
    # Turning the knob has to change what is on screen, or it is a setting the
    # reader cannot tell is connected to anything. Read against the seam under
    # the same top-k, so what is asserted is that the page passed the number on
    # rather than that it drew some smaller number of cards of its own.
    fewer = _evidence_through_the_seam(corpus, top_k=FEWER_CHUNKS)
    page = _open_page(monkeypatch, _factory(corpus))

    _set_retrieval(page, top_k=FEWER_CHUNKS)
    _ask_on_the_page(page, COVERED_QUESTION)

    assert page.exception == []
    assert [card.label for card in page.expander] == [
        interface.card_label(chunk) for chunk in fewer
    ]
    assert len(page.expander) == FEWER_CHUNKS < len(EXPECTED_CARDS)


def test_a_tighter_cap_spreads_the_cards_across_more_documents(monkeypatch, corpus):
    # What the cap is for: noticing when every card comes from one Document.
    # At the configured two, hash_table.md supplies two of the five cards; at
    # one it cannot, and the slot it gives up goes to a Document further down
    # the ranking. So the cap changes which cards appear rather than only how
    # many, which is the change a reader reads it by.
    uncapped = _evidence_through_the_seam(corpus)
    capped = _evidence_through_the_seam(
        corpus, max_chunks_per_document=ONE_PER_DOCUMENT
    )
    page = _open_page(monkeypatch, _factory(corpus))

    _set_retrieval(page, max_chunks_per_document=ONE_PER_DOCUMENT)
    _ask_on_the_page(page, COVERED_QUESTION)

    assert page.exception == []
    assert [card.label for card in page.expander] == [
        interface.card_label(chunk) for chunk in capped
    ]
    assert capped != uncapped
    assert len({chunk.doc_id for chunk in capped}) == len(capped)


def test_tau_is_not_adjustable_from_the_page(monkeypatch, corpus):
    # tau is a derived property of the embedding model (ADR-0003) rather than a
    # preference, and a per-question float on this page would slip past the
    # pairing DistanceThreshold exists to enforce -- the same reason the command
    # line has no --distance-threshold. So the page's numeric controls are
    # exactly the two retrieval knobs, and no other widget offers a threshold.
    page = _open_page(monkeypatch, _factory(corpus))

    assert [control.label for control in page.number_input] == [
        interface.TOP_K_LABEL,
        interface.MAX_PER_DOCUMENT_LABEL,
    ]
    assert page.slider == []
    assert page.text_input == []


def test_a_missing_api_key_is_a_message_rather_than_a_traceback(monkeypatch):
    # A configuration problem must not reach the reader as a traceback, and
    # must not be mistakable for a knowledge gap: the page says the provider
    # could not be reached, in neither of the shapes an Abstention wears.
    page = _open_page(monkeypatch, _unreachable_provider_factory(MISSING_KEY_DETAIL))

    assert page.exception == []
    assert [block.value for block in page.error] == [
        interface.provider_failure_notice(ProviderFailure(MISSING_KEY_DETAIL))
    ]
    assert page.warning == []
    assert page.info == []
    assert ABSTENTION_TEXT not in _shown(page)


def test_a_missing_api_key_message_keeps_the_providers_own_words(monkeypatch):
    # The SDK's text is the half that says what to do about it, and it names an
    # environment variable underscored either side. A page that let that be
    # read as Markdown would show the reader a differently spelled variable
    # name, which is the failure `as_written` exists to prevent.
    page = _open_page(monkeypatch, _unreachable_provider_factory(MISSING_KEY_DETAIL))

    [message] = [block.value for block in page.error]

    assert interface.as_written(MISSING_KEY_DETAIL) in message


def test_a_provider_error_while_answering_is_a_message_rather_than_a_traceback(
    monkeypatch, corpus
):
    # The same for a provider that fails mid-question. Nothing is drawn as an
    # answer and nothing as an Abstention: the question was not answered, and a
    # page that showed the retrieved cards under an error would be citing
    # Evidence for an answer that was never produced.
    page = _open_page(monkeypatch, _factory(corpus, generator=FailingGenerator()))

    _ask_on_the_page(page, COVERED_QUESTION)

    assert page.exception == []
    assert [block.value for block in page.error] == [
        interface.provider_failure_notice(ProviderFailure(PROVIDER_ERROR_DETAIL))
    ]
    assert page.warning == []
    assert page.info == []
    assert page.expander == []


def test_a_question_the_provider_failed_on_is_not_kept_as_a_turn(monkeypatch, corpus):
    # A Turn is a question together with the Answer it produced (CONTEXT.md),
    # and a failed question produced none -- so the session keeps nothing, and
    # the reader is not left scrolling past an error message dressed as an
    # answer they once got.
    page = _open_page(monkeypatch, _factory(corpus, generator=FailsOnceGenerator()))

    _ask_on_the_page(page, COVERED_QUESTION)
    _ask_on_the_page(page, SECOND_COVERED_QUESTION)

    assert page.exception == []
    assert page.error == []
    assert SECOND_COVERED_QUESTION in _shown(page)
    assert COVERED_QUESTION not in _shown(page)
    assert [caption.value for caption in page.caption] == [
        interface.scope_caption(ChunkFilter())
    ]


def test_a_tau_from_another_embedding_model_still_fails_loudly(monkeypatch, corpus):
    # The one failure this page must not render politely. A tau belonging to
    # another embedding model is a misconfiguration rather than an operational
    # hiccup, and a page that caught it would leave a mis-set gate running
    # behind a message the reader is invited to dismiss (ADR-0003).
    page = _open_page(monkeypatch, _misconfigured_tau_factory())

    assert page.exception != []
    assert page.error == []


def test_the_threshold_pairing_is_asserted_before_the_page_reaches_the_provider(
    monkeypatch,
):
    # And it is loud by the class hierarchy rather than by the order the except
    # clauses happen to be written in: a RuntimeError is not a ProviderFailure,
    # so nothing this page catches can catch it.
    monkeypatch.setattr(interface, "EMBEDDING_MODEL", OTHER_EMBEDDING_MODEL)

    with pytest.raises(RuntimeError) as raised:
        interface.build_dependencies()

    assert not isinstance(raised.value, ProviderFailure)
