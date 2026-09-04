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
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from config import (
    DISTANCE_THRESHOLD,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    MAX_CHUNKS_PER_DOCUMENT,
    REGISTRY_PATH,
    TOP_K,
)
from core.ask import Answer, ask
from core.embedder import Embedder, OpenAIEmbedder
from core.generator import Generator, OpenAIGenerator
from core.registry import Registry
from core.store import RetrievedChunk, VectorStore
from presentation import evidence_card

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


# Every ASCII punctuation mark, which is exactly the set CommonMark lets a
# backslash escape back into itself.
_MARKDOWN_PUNCTUATION = re.compile(r"([!-/:-@\[-`{-~])")


def card_label(chunk: RetrievedChunk) -> str:
    """The shared Evidence card, as this surface has to write it onto a widget.

    Streamlit reads a widget label as Markdown, and a Locator is arbitrary text
    from someone's headings: the corpus carries thirty Locators containing an
    asterisk today, and a heading with two of them, or with brackets around a
    word, would be typeset rather than shown. The terminal prints the same
    string literally, so the page would be describing one Evidence differently
    from the other surface -- which is the drift presentation.py exists to
    prevent (#22).

    Escaped here rather than worded around there: what a card says is
    presentation.py's, and getting a plain string intact onto a widget is this
    surface's own business, as the terminal's bullet is its.
    """
    return _MARKDOWN_PUNCTUATION.sub(r"\\\1", evidence_card(chunk))


@dataclass(frozen=True)
class Dependencies:
    """Everything the page reaches the outside world through, in one value.

    One record built by one factory, rather than a hook per dependency, so that
    what a test substitutes is the whole set and what production reads is
    configuration and secrets -- with no test-only branch anywhere inside the
    page.

    The threshold belongs here beside the embedder, not read from config at the
    point of use, for the reason ADR-0003 gives: τ is a property of the
    embedding model. A substituted embedder scores distances in its own
    geometry, so it has to bring the τ that was set for it; reading a configured
    τ over an injected model is exactly the confound `DistanceThreshold.for_model`
    exists to refuse.
    """

    registry: Registry
    store: VectorStore
    embedder: Embedder
    generator: Generator
    distance_threshold: float


def build_dependencies() -> Dependencies:
    """Production's set: the configured models, over the corpus on disk.

    The page's one hook, and the one name a test replaces
    (tests/test_app_seam.py).
    """
    # OPENAI_API_KEY reaches the OpenAI client through the environment, and the
    # .env beside this file is the fallback rather than the authority -- as
    # cli.py, for the reasons written there. Loaded here rather than at import
    # time so that importing this module still costs nothing and needs no key.
    load_dotenv(Path(__file__).resolve().parent / ".env")

    # Asserted before the store is opened or a key is needed, exactly as the
    # command line asserts it: a τ belonging to another embedding model is a
    # misconfiguration rather than a bad answer, so it fails loudly at startup
    # instead of being rendered politely as an operational hiccup.
    threshold = DISTANCE_THRESHOLD.for_model(EMBEDDING_MODEL)

    return Dependencies(
        registry=Registry(REGISTRY_PATH),
        store=VectorStore(),
        embedder=OpenAIEmbedder(model=EMBEDDING_MODEL),
        generator=OpenAIGenerator(model=GENERATION_MODEL),
        distance_threshold=threshold,
    )


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

    dependencies = build_dependencies()

    # Read off the registry, which is ADR-0001's authority on what Documents
    # exist, and read before anything is typed -- so the emptiness is reported
    # rather than discovered by embedding a question and interpreting the
    # abstention that comes back.
    if not dependencies.registry.list():
        st.warning(EMPTY_CORPUS_NOTICE)
        return

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
    with st.status(IN_FLIGHT_LABEL):
        answer = ask(
            # The question as it was typed, and nothing of the session with it.
            # What scrolls above is display-only: every question is retrieved
            # independently, and folding what is on screen into the query here
            # would be history-aware retrieval and query rewriting done by
            # accident -- Week 7 experiments (PLAN.md §第 7 週), confounded
            # before they start by a baseline that already did some of it.
            question,
            dependencies.embedder,
            dependencies.store,
            dependencies.generator,
            top_k=TOP_K,
            distance_threshold=dependencies.distance_threshold,
            max_chunks_per_document=MAX_CHUNKS_PER_DOCUMENT,
        )

    # Kept before it is drawn, and drawn from what was kept: the fresh turn
    # reaches the page by the same route it will take on every rerun after
    # this one, rather than by a shorter one that only the newest turn takes.
    turn = Turn(question=question, answer=answer)
    turns.append(turn)
    render_answer(turn.answer)


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
    """The question in the reader's own words, as they typed it."""
    with st.chat_message("user"):
        st.markdown(question)


def render_answer(answer: Answer) -> None:
    """One Answer as the page reads it: the text, then the Evidence behind it.

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
        st.markdown(answer.text)
        for chunk in answer.evidence:
            # The card says what the Chunk is; the expander is how a page
            # offers to open one.
            with st.expander(card_label(chunk)):
                st.text(chunk.text)
