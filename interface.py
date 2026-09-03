"""The page: a question typed in, the answer, and the Evidence it rests on.

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

    question = st.chat_input(QUESTION_PROMPT)
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    # The answering happens inside the status, so what the reader watches spin
    # is the work itself: it completes when the answer is on the page beneath
    # it, and not a moment before.
    with st.status(IN_FLIGHT_LABEL):
        answer = ask(
            question,
            dependencies.embedder,
            dependencies.store,
            dependencies.generator,
            top_k=TOP_K,
            distance_threshold=dependencies.distance_threshold,
            max_chunks_per_document=MAX_CHUNKS_PER_DOCUMENT,
        )

    render_answer(answer)


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
