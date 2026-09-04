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
import pytest
from config import DOMAINS, MAX_CHUNKS_PER_DOCUMENT, TOP_K
from core.ask import ask
from core.gate import ABSTENTION_TEXT
from core.registry import Registry
from core.store import ChunkFilter, VectorStore
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


def _evidence_through_the_seam(corpus, question=COVERED_QUESTION, chunk_filter=None):
    """What `ask` returns for `question` over the same corpus, under the same
    configured retrieval the page runs on.

    The cards are read against this rather than against a list retyped here, so
    what the assertions say is that the page and the pipeline agree -- a page
    that dropped a Chunk or added one of its own is red even if both it and a
    constant were changed together.
    """
    _registry, store, embedder = ingest_mixed(corpus)
    return ask(
        question,
        embedder,
        store,
        ScriptedGenerator(),
        top_k=TOP_K,
        distance_threshold=GATE_TAU,
        chunk_filter=chunk_filter,
        max_chunks_per_document=MAX_CHUNKS_PER_DOCUMENT,
    ).evidence


def _shown(page: AppTest) -> str:
    """Everything the page wrote in prose, as one string."""
    return "\n".join(
        [block.value for block in page.markdown]
        + [block.value for block in page.warning]
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
