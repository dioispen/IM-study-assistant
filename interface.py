"""The page: a revision session of questions, answers, and the Evidence they
rest on.

The second of the two surfaces over one pipeline. It is an adapter and nothing
more -- it composes the same embedder, store, retriever, gate and generator the
terminal does, through the same `ask`, and reimplements no retrieval, filtering,
gating or prompt logic of its own. Anything it needs to *know* about an answer
is carried on the Answer (core/ask.py) rather than re-derived from what came
back; anything it *says* about one is presentation.py's wording, which neither
surface owns (#22).

It reads the Document registry and the vector store the command line reads.
There is one corpus, not a UI-shaped copy of it.

Not the script Streamlit runs -- that is app.py, and the split is what makes
`build_dependencies` replaceable; see its docstring.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from config import (
    CHROMA_PATH,
    DISTANCE_THRESHOLD,
    DOMAINS,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    MAX_CHUNKS_PER_DOCUMENT,
    REGISTRY_PATH,
    TOP_K,
)
from core.ask import Abstention, Answer, ask
from core.embedder import Embedder, OpenAIEmbedder
from core.generator import Generator, OpenAIGenerator
from core.openai_client import ProviderFailure
from core.registry import Registry
from core.store import ChunkFilter, RetrievedChunk, VectorStore
from presentation import (
    describe_filter,
    evidence_card,
    provisional_threshold_notice,
)

TITLE = "IM Study Assistant"

QUESTION_PROMPT = "Ask a question about your notes"

# Where the session's turns live between runs. Streamlit re-executes this
# script on every interaction, so a list held in a module-level name would be
# one reader's session shown to whoever opened the page next; session state is
# per-session, and is the only thing here that outlives a run.
TURNS_KEY = "turns"

# Ends a topic without scrolling past the last one. Named for what it does to
# the session rather than to the screen -- what goes is the record, and the
# page redrawing empty is a consequence of that.
#
# Not in presentation.py, for the reason EMPTY_CORPUS_NOTICE is not: that module
# holds what *both* surfaces show, and the terminal has no session to clear.
CLEAR_LABEL = "Clear session"

# Shown while the question is being answered, and left on the page afterwards
# as the record of that having happened. A slow retrieval must be readable as
# work in progress rather than as a page that has died.
IN_FLIGHT_LABEL = "Retrieving Evidence and answering…"

# The sidebar holds the Scope: what this session's questions are allowed to
# retrieve from. Headed with the word CONTEXT.md uses for it, because the two
# menus beneath narrow what may be *found* and never what may be *said*.
SCOPE_HEADER = "Scope"

# Each menu is keyed by the ChunkFilter field it fills. The key is what
# Streamlit holds a selection under between runs, so naming it for the field
# leaves no table anywhere translating a widget's name into a filter's.
DOMAIN_KEY = "domain"
SOURCE_TYPE_KEY = "source_type"

DOMAIN_LABEL = "Domain"
SOURCE_TYPE_LABEL = "Source type"

# How each menu spells "no restriction on this axis". None is what the option
# actually carries, because a ChunkFilter field left None is not a restriction
# (core/store.py) -- so these are a wording for None rather than a sentinel the
# page then has to translate back.
ANY_DOMAIN = "Any Domain"
ANY_SOURCE_TYPE = "Any Source type"

# The two knobs a student can meaningfully turn while reading, beneath the
# Scope and distinct from it: the Scope decides what a question may be answered
# from, and these decide how much of it comes back. Headed for what they do
# rather than for who turns them -- "advanced" describes the reader, and a
# reader deciding whether to open a heading is better served by the subject.
#
# Not folded into an expander, though that is where advanced settings usually
# go: an expander on this page is how a card opens onto its Chunk, and a second
# kind of expander would cost the reader the one thing that made the cards
# recognisable at a glance.
EVIDENCE_HEADER = "How much Evidence"

# Each control is keyed by the `ask` parameter it fills, exactly as each menu
# above is keyed by the ChunkFilter field it fills, and for the same reason:
# nothing anywhere translates a widget's name into an argument's.
TOP_K_KEY = "top_k"
MAX_PER_DOCUMENT_KEY = "max_chunks_per_document"

TOP_K_LABEL = "Chunks retrieved"
MAX_PER_DOCUMENT_LABEL = "Most Chunks from one Document"

TOP_K_HELP = (
    "How many Chunks are retrieved for a question and handed to the answer as "
    "its Evidence — one card each."
)
MAX_PER_DOCUMENT_HELP = (
    "The most Chunks any single Document may contribute. Lower it to see "
    "whether an answer rests on several Documents or on one long note "
    "crowding the others out."
)

# The furthest either control may be turned. Every Chunk retrieved is prompt
# tokens spent and one more card to read, and past a point the Evidence is
# longer than the reader will actually check -- which is the opposite of what
# the cards are for.
#
# Derived from the configured values rather than written as its own number, and
# from both of them: a ceiling beneath a value a control defaults to is a page
# that cannot draw its own defaults, and Streamlit refuses the widget outright
# rather than clamping it. The larger cap is not hypothetical -- a cap well
# above top-k is exactly the non-binding baseline the Week 7 diversity sweep
# reads against (PLAN.md §第 7 週), so config.py carrying one has to leave this
# page standing.
#
# It is also what makes uncapped retrieval reachable here without the command
# line's `--max-per-document off`: a cap at or above the top-k in force binds
# nothing, so that baseline can be set on the page by raising the cap rather
# than by a sentinel value the reader would have to be taught.
MAX_RETRIEVED = max(4 * TOP_K, MAX_CHUNKS_PER_DOCUMENT)

# The smallest either may be turned to. A cap of zero admits no Chunk of any
# Document, so every question would abstain as though the corpus were empty --
# the abstention core/retriever.py refuses to serve, refused here before it can
# be typed rather than raised after it has been.
MIN_RETRIEVED = 1

CORPUS_HEADER = "In the corpus"
DOMAIN_COUNTS_HEADING = "Documents per Domain:"

# The Scope of a question that restricted neither axis. Named rather than left
# blank: the whole Corpus is one of the Scopes CONTEXT.md defines, and a turn
# that said nothing about its Scope would read as a turn whose Scope went
# unrecorded.
WHOLE_CORPUS = "the whole corpus"

# What a corpus with no Document in it says for itself. An empty corpus abstains
# at the distance gate -- correctly, since there is nothing within τ -- and that
# abstention tells the reader their corpus does not cover the question, when
# what happened is that there is no corpus. A fresh checkout should not look
# like a broken system.
#
# Not in presentation.py: that module holds what *both* surfaces show, and the
# terminal has nothing to say here -- `ask` is one command with its answer, not
# a page standing open in front of someone who has not typed yet.
EMPTY_CORPUS_NOTICE = (
    "The corpus is empty — no Document has been ingested yet, so there is "
    "nothing to answer from. Ingest a folder of notes first: "
    "`python cli.py ingest <folder> --domain <domain>`."
)

# What an operational failure says: the language model could not be reached, so
# nothing here is a claim about the corpus. A missing key and a provider outage
# share one wording, because the reader's question in both cases is the same
# one -- "is this my notes or my setup?" -- and the provider's own words below
# are what separate them.
#
# Said in as many words rather than left to a red box, because the failure this
# is guarding against is not an ugly traceback but a misread one: a
# configuration problem that looks like a knowledge gap teaches the reader
# their corpus is worse than it is. Neither of ADR-0003's Abstentions is on
# screen here, and this says so.
#
# Not in presentation.py: that module holds what *both* surfaces show, and the
# terminal does not catch this yet -- a traceback in a console is a diagnosis,
# while on a page it is a dead end. The day cli.py catches it too, the wording
# moves there and this becomes the framing around it, as the cards' is.
PROVIDER_FAILURE_NOTICE = (
    "The language model could not be reached, so nothing can be answered "
    "right now. That is a configuration or provider problem rather than "
    "anything your notes do or do not say — it is neither an answer nor an "
    "Abstention. What went wrong: {detail}"
)


# Every ASCII punctuation mark, which is exactly the set CommonMark lets a
# backslash escape back into itself.
_MARKDOWN_PUNCTUATION = re.compile(r"([!-/:-@\[-`{-~])")


def as_written(text: str) -> str:
    """`text` as itself, once a Markdown widget has read it.

    Streamlit reads a widget label and a caption as Markdown, and nearly
    everything this page shows is arbitrary text out of someone's corpus or
    their own keyboard: a Locator taken from their headings, a Source type they
    typed at the command line, the question they just asked. The corpus carries
    thirty Locators containing an asterisk today, and a heading with two of them
    -- or a Source type with an underscore either side -- would be typeset
    rather than shown. The terminal prints the same strings literally, so the
    page would be describing one Evidence differently from the other surface,
    which is the drift presentation.py exists to prevent (#22).

    Escaped here rather than worded around there: what a card says is
    presentation.py's, and getting a plain string intact onto a widget is this
    surface's own business, as the terminal's bullet is its.
    """
    return _MARKDOWN_PUNCTUATION.sub(r"\\\1", text)


def card_label(chunk: RetrievedChunk) -> str:
    """The shared Evidence card, as this surface has to write it onto a widget."""
    return as_written(evidence_card(chunk))


@dataclass(frozen=True)
class Models:
    """The two language models a question is answered through -- the ones
    config.py names as EMBEDDING_MODEL and GENERATION_MODEL -- and the τ that
    belongs to the first of them.

    Held apart from the corpus half of Dependencies because they are the half
    that needs a provider: constructing either one reaches for a key, while the
    registry and the store are readable without one. That is the whole reason
    the two halves are separable at all -- see Dependencies.

    The threshold is here rather than beside the store, for the reason ADR-0003
    gives: τ is a property of the embedding model. A substituted embedder scores
    distances in its own geometry, so it has to bring the τ that was set for it,
    and putting the two in one record is what makes reading a configured τ over
    an injected model impossible rather than merely discouraged -- that pairing
    is the confound `DistanceThreshold.for_model` exists to refuse.

    Deferred with the models, but not *derived* with them: the assertion that τ
    was set for the model in use is made eagerly, in build_dependencies, and
    only its result is carried here. What that assertion reads is a model name
    out of config.py, which needs nothing constructed -- so the misconfiguration
    still fails at startup rather than waiting behind a corpus.
    """

    embedder: Embedder
    generator: Generator
    distance_threshold: float


@dataclass(frozen=True)
class Dependencies:
    """Everything the page reaches the outside world through, in one value.

    One record built by one factory, rather than a hook per dependency, so that
    what a test substitutes is the whole set and what production reads is
    configuration and secrets -- with no test-only branch anywhere inside the
    page.

    Staged rather than flat, which is what #31 is about. The corpus half is open
    by the time the record is handed over; the models are behind a call that has
    not been made yet. What the registry says -- whether anything has been
    ingested at all -- is knowable without a provider, so the page can read it
    and answer for an empty corpus before anything needs a key. A record holding
    two live clients could not be built at all on a fresh checkout, and the
    notice written for precisely that reader would sit unreachable behind the
    traceback of building it.

    Only what needs a provider is behind the call. τ is not on this half at all
    -- it rides with the embedder it was set for, on Models -- while the
    assertion pairing it to the model in use is made eagerly, before this record
    exists. Deferring a *client* costs nothing; deferring that assertion would
    let a misconfiguration hide behind an empty corpus.
    """

    registry: Registry
    store: VectorStore
    open_models: Callable[[], Models]


# The .env beside this file, as a path rather than as an expression at the point
# of use. A fallback and not the authority -- an OPENAI_API_KEY already in the
# environment wins -- and named here so that a test meaning "with no key" can
# say that a machine's own .env is not read either. Clearing the environment
# alone would have the key handed straight back, and the test would pass without
# ever having been in the situation it describes.
ENV_PATH = Path(__file__).resolve().parent / ".env"


def build_dependencies(
    registry_path: Path = REGISTRY_PATH,
    chroma_path: Path = CHROMA_PATH,
    env_path: Path = ENV_PATH,
) -> Dependencies:
    """Production's set: the configured models, over the corpus on disk.

    The page's one hook, and the one name a test replaces
    (tests/test_app_seam.py).

    Every path it reads is a parameter defaulted to the configured value, so
    that this construction -- this ordering, this laziness, this assertion -- is
    reachable from a test rather than only from a machine whose data directory
    and .env are the ones config.py names. A suite that could only substitute a
    double for this function could never have caught #31, because what was wrong
    was what the function itself built, and when.
    """
    # OPENAI_API_KEY reaches the OpenAI client through the environment, and the
    # .env beside this file is the fallback rather than the authority -- as
    # cli.py, for the reasons written there. Loaded here rather than at import
    # time so that importing this module still costs nothing and needs no key.
    load_dotenv(env_path)

    # Asserted before the store is opened and before any client is built,
    # exactly as the command line asserts it: a τ belonging to another embedding
    # model is a misconfiguration rather than a bad answer, so it fails loudly
    # at startup instead of being rendered politely as an operational hiccup.
    # First of everything here, so that an empty corpus cannot keep it quiet.
    threshold = DISTANCE_THRESHOLD.for_model(EMBEDDING_MODEL)

    def open_models() -> Models:
        """The provider clients, built at the moment the page asks for them.

        Not at the moment this record is: `OpenAI()` reaches for a key as it is
        constructed, and the page reads the registry first (#31). A
        ProviderFailure out of here is therefore a page that has a corpus and
        cannot reach the model for it -- which is the failure
        PROVIDER_FAILURE_NOTICE is actually about.
        """
        return Models(
            embedder=OpenAIEmbedder(model=EMBEDDING_MODEL),
            generator=OpenAIGenerator(model=GENERATION_MODEL),
            distance_threshold=threshold,
        )

    return Dependencies(
        registry=Registry(registry_path),
        store=VectorStore(path=chroma_path),
        open_models=open_models,
    )


@dataclass(frozen=True)
class Retrieval:
    """What the sidebar settles about the next question, in one value.

    The Scope it may be answered from, and how much Evidence may come back.
    Three fields named for the three `ask` parameters they fill, so the page
    hands the pipeline what the reader set rather than translating it, and a
    fourth control added to the sidebar arrives here rather than as a fourth
    return value threaded through the same functions.

    τ is not among them, and that is the design (ADR-0003). The other three are
    plain arguments to retrieval -- a preference about how much Evidence to
    look at -- while τ is a property of the embedding model, carried in a
    DistanceThreshold that refuses to be read without naming the model it was
    set for. A widget can hand `ask` an int; nothing can hand it a paired
    float, which is why the threshold reaches the page on Dependencies beside
    the embedder instead.
    """

    chunk_filter: ChunkFilter
    top_k: int
    max_chunks_per_document: int


@dataclass(frozen=True)
class Turn:
    """One question the session keeps, together with the Answer it produced.

    Everything a turn is drawn from is in here, and drawing one is a pure
    function of it. That is what makes a session redisplayed rather than
    re-derived: the Scope a question was answered under travels on its Answer
    (core/ask.py), so a turn scrolled back to once the sidebar can move the
    Scope mid-session (#25) still reads as the turn that happened, rather than
    as the turn today's settings would have produced.

    The Scope is therefore not copied onto the turn beside the Answer that
    already carries it. Two places for one fact is two places to disagree, and
    the one a card could be drawn from is the Answer's.
    """

    question: str
    answer: Answer


def session_turns() -> list[Turn]:
    """The session's turns, in the order they were asked; empty on a fresh page.

    The list itself rather than a copy of it, because appending a turn has to be
    appending to the session -- Streamlit holds the object across runs, and a
    caller that appended to a copy would have answered a question the next rerun
    knows nothing about.

    Nothing here survives a page reload, which is the intended lifetime: a
    session is a sitting, and persistence would be a corpus of its own.
    """
    if TURNS_KEY not in st.session_state:
        st.session_state[TURNS_KEY] = []
    return st.session_state[TURNS_KEY]


def render_page() -> None:
    st.title(TITLE)

    # Nothing here reaches a provider: the registry and the store are opened and
    # the threshold pairing is asserted, both of which are knowable without a
    # key. No ProviderFailure can come out of this line, so nothing catches one
    # around it -- and the RuntimeError a τ set for another embedding model
    # raises is left alone, to reach the console as the traceback config.py
    # spelled its message for.
    dependencies = build_dependencies()

    # Read off the registry, which is ADR-0001's authority on what Documents
    # exist, and read before anything is typed -- so the emptiness is reported
    # rather than discovered by embedding a question and interpreting the
    # abstention that comes back.
    #
    # Read before the models are opened, too, and that ordering is the whole of
    # #31. On a fresh checkout both things are true -- nothing is ingested and
    # no key is set -- and only one of them is what the reader has to do next.
    # Told about the key, they would go and configure a provider before having a
    # corpus to query, and the ingest command this notice names needs that key
    # anyway. What the registry says is knowable without a provider, so it is
    # what the page finds out first.
    #
    # Returned from before the sidebar as well as before the chat input: there
    # is nothing to scope and no corpus to summarise, and a Scope menu over an
    # empty corpus would offer the reader a way to narrow nothing.
    if not dependencies.registry.list():
        st.warning(EMPTY_CORPUS_NOTICE)
        return

    try:
        models = dependencies.open_models()
    except ProviderFailure as failure:
        # The missing-key case, and every other way the provider cannot be
        # reached before a question is typed: the clients are constructed here,
        # so this is a page that fails while it is being opened. Reported and
        # returned from, because there is nothing behind it to draw -- a chat
        # input over models that do not exist is an invitation to type into a
        # page that cannot answer.
        #
        # There is a corpus by the time this can happen, which is what makes the
        # message the true one: the provider is what stands between this reader
        # and an answer, rather than one of two things wrong at once (#31).
        #
        # Not the threshold-to-embedding-model misconfiguration, which
        # build_dependencies asserts above and raises RuntimeError for: that is
        # not a hiccup to render politely, and ProviderFailure is not one of its
        # ancestors, so this clause cannot catch it (core/openai_client.py).
        render_provider_failure(failure)
        return

    # Read before anything is drawn from it, because these are settings the
    # session sits under rather than arguments to one question: what comes back
    # is what the sidebar says now, however many questions ago it was set.
    retrieval = render_sidebar(dependencies.registry)

    turns = session_turns()

    # Offered above the turns it clears, because a click is read on the run it
    # triggers: a control drawn below them would empty a list that had already
    # been rendered, and the page would go on showing a session that no longer
    # exists until the next interaction.
    if st.button(CLEAR_LABEL):
        turns.clear()

    for turn in turns:
        render_turn(turn)

    question = st.chat_input(QUESTION_PROMPT)
    if not question:
        return

    # Drawn before the answering rather than with the turn it will belong to,
    # so the question the reader is waiting on is on screen above the thing
    # that says it is being worked on.
    render_question(question)

    # The answering happens inside the status, so what the reader watches spin
    # is the work itself: it completes when the answer is on the page beneath
    # it, and not a moment before.
    #
    # The catch is outside the status rather than inside it, so a question the
    # provider failed on leaves the status showing an error rather than
    # completing over a failure -- st.status reads the exception on its way out
    # and does not swallow it.
    try:
        with st.status(IN_FLIGHT_LABEL):
            answer = ask(
                # The question as it was typed, and nothing of the session with
                # it. What scrolls above is display-only: every question is
                # retrieved independently, and folding what is on screen into
                # the query here would be history-aware retrieval and query
                # rewriting done by accident -- Week 7 experiments (PLAN.md §第
                # 7 週), confounded before they start by a baseline that
                # already did some of it.
                question,
                models.embedder,
                dependencies.store,
                models.generator,
                # What the sidebar is set to, passed straight through: the page
                # is a front end for the retrieval built in #5 and reimplements
                # none of it. The Scope travels back on the Answer, which is
                # what lets the turn below say what it was answered under.
                top_k=retrieval.top_k,
                distance_threshold=models.distance_threshold,
                chunk_filter=retrieval.chunk_filter,
                max_chunks_per_document=retrieval.max_chunks_per_document,
            )
    except ProviderFailure as failure:
        # Embedding or generation could not be reached. Nothing is kept and
        # nothing is drawn as an answer: there is no Answer, so there is no
        # Turn to keep (CONTEXT.md), and cards under an error message would be
        # citing Evidence for an answer that was never produced.
        render_provider_failure(failure)
        return

    # Kept before it is drawn, and drawn from what was kept: the fresh turn
    # reaches the page by the same route it will take on every rerun after
    # this one, rather than by a shorter one that only the newest turn takes.
    turn = Turn(question=question, answer=answer)
    turns.append(turn)
    render_answer(turn.answer)


def render_sidebar(registry: Registry) -> Retrieval:
    """The Scope and the retrieval in force, as visible settings rather than
    remembered flags.

    Returns what the sidebar says and acts on none of it: the page is a front
    end for the filtered retrieval and the per-Document Evidence cap built in
    #5, so a Retrieval goes to `ask` and nothing is narrowed, ranked or capped
    here.

    The two menus read from different places, and that is the design rather
    than an inconsistency. Domains are declared configuration, so that menu is
    DOMAINS entire -- including a Domain nothing has been ingested under, which
    the summary below then explains. Source type is an open enumeration whose
    values are whatever has been ingested (CONTEXT.md), so that menu is the
    registry's own values, and becomes a real choice the moment a second Source
    type arrives with no edit here.

    Beneath them, the two controls that decide how much Evidence comes back
    rather than what it may be drawn from. They default to the configured
    values, so a page nobody has touched answers exactly as the command line
    does with no flags -- these are a way to depart from the configuration, not
    a second place it is written down. τ is not among them, for the reason
    Retrieval gives.

    Streamlit holds a keyed widget's value across runs, which is the whole of
    what makes any of this stay in force until it is changed: this is drawn
    afresh on every rerun and hands back what was set however many questions
    ago.
    """
    with st.sidebar:
        st.header(SCOPE_HEADER)
        domain = st.selectbox(
            DOMAIN_LABEL,
            [None, *DOMAINS],
            format_func=lambda domain: ANY_DOMAIN if domain is None else domain,
            key=DOMAIN_KEY,
        )
        source_type = st.selectbox(
            SOURCE_TYPE_LABEL,
            [None, *registry.source_types()],
            format_func=lambda source_type: (
                ANY_SOURCE_TYPE if source_type is None else source_type
            ),
            key=SOURCE_TYPE_KEY,
        )

        # Beneath the Scope, because they are read in that order: what a
        # question may be answered from, and then how much of it to look at.
        st.subheader(EVIDENCE_HEADER)
        top_k = st.number_input(
            TOP_K_LABEL,
            min_value=MIN_RETRIEVED,
            max_value=MAX_RETRIEVED,
            value=TOP_K,
            step=1,
            help=TOP_K_HELP,
            key=TOP_K_KEY,
        )
        max_chunks_per_document = st.number_input(
            MAX_PER_DOCUMENT_LABEL,
            min_value=MIN_RETRIEVED,
            max_value=MAX_RETRIEVED,
            value=MAX_CHUNKS_PER_DOCUMENT,
            step=1,
            help=MAX_PER_DOCUMENT_HELP,
            key=MAX_PER_DOCUMENT_KEY,
        )

        st.markdown(corpus_summary(registry))

    return Retrieval(
        chunk_filter=ChunkFilter(domain=domain, source_type=source_type),
        top_k=top_k,
        max_chunks_per_document=max_chunks_per_document,
    )


def corpus_summary(registry: Registry) -> str:
    """What the corpus holds, so the reader knows what they are querying before
    they blame retrieval.

    Read off the Document registry, which is ADR-0001's authority on what
    Documents exist. The vector store holds one particular chunking of those
    Documents, so a count taken there would report a single long note as a
    large corpus -- and this sits beside the menus a reader is about to narrow
    with, where that would mislead most.
    """
    return "\n".join(
        [f"**{CORPUS_HEADER}**", "", DOMAIN_COUNTS_HEADING]
        + [
            domain_count_line(domain, documents)
            for domain, documents in registry.documents_per_domain().items()
        ]
        + ["", source_types_line(registry.source_types())]
    )


def domain_count_line(domain: str, documents: int) -> str:
    """One Domain of the corpus summary: the Domain, and how many Documents.

    The Domain is escaped and the bullet is not, which is the split every line
    of this summary has: the wording around a value is written here as Markdown
    and meant as Markdown, while the value itself came out of the registry and
    is only ever text (`as_written`).
    """
    return f"- {as_written(domain)}: {documents}"


def source_types_line(source_types: list[str]) -> str:
    """The Source types the corpus holds, as one line.

    One line rather than a list of its own, because while the corpus holds a
    single Source type this is read as "is there anything here but my notes
    yet?" -- and that answer is a word, not a bullet.

    Each value escaped for the same reason a card's is: a Source type is
    whatever was typed at the command line, and one carrying Markdown
    punctuation would be typeset in this sidebar while appearing literally on
    the cards below it.
    """
    present = ", ".join(as_written(source_type) for source_type in source_types)
    return f"Source types present: {present}"


def render_turn(turn: Turn) -> None:
    """One kept turn, redrawn: the question as it was asked, then its Answer.

    The same two halves the live question is drawn from, in the same order, so
    a turn the reader scrolls back to is not a second rendering of the same
    thing written twice. What the live path adds between them is the status --
    which belongs to a question being answered now, and to no turn the session
    has already kept.
    """
    render_question(turn.question)
    render_answer(turn.answer)


def render_question(question: str) -> None:
    """The question in the reader's own words, as they typed it.

    Escaped onto the widget, because a chat message is read as Markdown and a
    question is the last arbitrary string on this page that was still going
    onto one raw (`as_written`). A study corpus is made of identifiers, so this
    is the string it mangles most: a question about `open_addressing` and
    `hash_table` came back as one italic run, and a turn is only worth
    scrolling back to if the question above an answer is the question that
    produced it.

    Display-only, and stops here. What `render_page` hands to `ask` is the
    string the reader typed -- escaping it there would embed the backslashes
    and retrieve against a question nobody asked.

    Both paths, and by construction: the live question and the same question
    redrawn from its kept turn are this one function, so neither can be
    escaped without the other.
    """
    with st.chat_message("user"):
        st.markdown(as_written(question))


def scope_caption(chunk_filter: ChunkFilter) -> str:
    """The Scope one turn was answered under, as that turn says it.

    Written over the ChunkFilter the Answer carries and never over the menus,
    which is what makes a session honest to scroll back through: a caption read
    off the sidebar would relabel every turn on screen the moment the reader
    moved the filter, and a turn answered across the whole corpus would come to
    claim it had been scoped.

    The restriction itself is presentation.py's wording -- the terminal names
    the same one beside a gate Abstention -- and only the framing around it,
    and getting it onto a widget intact, belong to this surface: a caption is
    read as Markdown, and a Source type is arbitrary text (`as_written`).
    """
    return as_written(f"Scope: {describe_filter(chunk_filter) or WHOLE_CORPUS}")


def render_answer(answer: Answer) -> None:
    """One Answer as the page reads it: the Scope, the text, then the Evidence.

    The Scope leads, because it is the frame the rest is read inside: an answer
    from "my own dsa notes" and an answer from the whole corpus are different
    claims, and a reader who learns which one this was only after reading it
    has already read it wrongly.

    Which layer declined, if either, is read off the Answer rather than
    recognised in its wording -- the pipeline already decided, and a surface
    that decided again could disagree with it (core/ask.py). The branch below
    asks which layer fired rather than whether any did, because the two
    Abstentions are different things: the gate's cites nothing and the
    backstop's cites the Evidence the model judged insufficient, and they call
    for different responses from the reader (ADR-0003, #26).

    Each of the three lands in a widget of its own -- a warning, an
    informational note, plain prose -- so the difference is in the shape of the
    turn rather than in wording the reader has to compare. A two-layer design
    nobody can observe on screen is a layer nobody can tell is broken.

    One card per Chunk of the Evidence and none besides. A card is the only
    place a Document appears by name, so a card drawn from the corpus rather
    than from this Answer's Evidence would credit the answer to a Document
    generation was never given (CONTEXT.md).

    The card opens onto the Chunk's own text -- what generation was handed,
    rather than a summary of it or the Document around it -- because judging a
    suspicious claim should not mean leaving the page to go and find the file.
    Shown verbatim for that reason: the notes are Markdown, so a Chunk carries
    its author's links, admonitions and indentation, and rendering them would
    show the reader a typeset version of the text rather than the text. The
    answer above it is model prose, which is written to be rendered.
    """
    with st.chat_message("assistant"):
        st.caption(scope_caption(answer.chunk_filter))

        if answer.abstention is Abstention.DISTANCE_GATE:
            # Nothing came back within τ, so generation was never given
            # anything and there is nothing to cite. The branch is what makes
            # that visible rather than incidental: an Answer whose Evidence
            # happened to be empty would draw no cards either, and the reader
            # would be left to infer from an absence.
            st.warning(answer.text)
            # An Abstention is the one moment an underived τ visibly costs the
            # reader an answer, so this is where the gate admits it is
            # uncalibrated. Whether there is anything to admit is the
            # threshold's own business (presentation.py); a caption beneath the
            # refusal is how this surface says an aside, as the terminal's
            # parentheses are its.
            notice = provisional_threshold_notice()
            if notice:
                st.caption(as_written(notice))
            # The restriction the question was narrowed to is named by the
            # Scope caption above, which every turn carries (#25) -- so nothing
            # is repeated here. The terminal prints its own restriction line
            # because it has no per-answer Scope of its own to read.
            return

        if answer.abstention is Abstention.PROMPT_BACKSTOP:
            # Evidence was assembled, handed over and judged insufficient, and
            # the cards below are exactly that material -- which is the one
            # case where the reader wants to look at it and disagree. A
            # different callout from the gate's, because the two Abstentions
            # ask different things of the reader: go and find material, against
            # read what was found and judge it. Told apart by the shape of the
            # turn rather than by reading either text closely (ADR-0003).
            st.info(answer.text)
        else:
            st.markdown(answer.text)

        for chunk in answer.evidence:
            # The card says what the Chunk is; the expander is how a page
            # offers to open one.
            with st.expander(card_label(chunk)):
                st.text(chunk.text)


def provider_failure_notice(failure: ProviderFailure) -> str:
    """An operational failure as the page puts it to the reader.

    The provider's own words are kept and framed rather than replaced: they are
    the half that says what to do next -- "set the OPENAI_API_KEY environment
    variable" is an instruction, and a page that summarised it into "something
    went wrong" would leave the reader with nothing to act on.

    Escaped, and only there: the framing is written here as Markdown and meant
    as Markdown, while the detail arrived from outside and is only ever text
    (`as_written`). The missing-key message is exactly why -- it names an
    environment variable underscored either side, and a page that let that be
    typeset would show the reader a variable name they cannot set.
    """
    return PROVIDER_FAILURE_NOTICE.format(detail=as_written(str(failure)))


def render_provider_failure(failure: ProviderFailure) -> None:
    """The failure, in the one widget neither Abstention uses.

    An error rather than the gate's warning or the backstop's informational
    note, because the reader's first question is which of the three happened:
    the corpus does not cover this, the Evidence did not support it, or the
    system never got as far as asking. Three renderings for three outcomes, as
    the two Abstentions already are (ADR-0003) -- and a failure that borrowed
    an Abstention's shape would be the misconfiguration-as-knowledge-gap this
    exists to prevent.
    """
    st.error(provider_failure_notice(failure))
