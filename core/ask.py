"""The ask seam: retrieve Evidence, gate on distance, generate an answer.

Kept separate from the CLI so tests can inject a fake Embedder and Generator
and run fully offline (see ADR-0002 -- these tests assert on Evidence, not on
generated text).
"""

from dataclasses import dataclass
from enum import Enum

from core.embedder import Embedder
from core.gate import ABSTENTION_TEXT, should_abstain
from core.generator import Generator, build_prompt
from core.retriever import retrieve
from core.store import ChunkFilter, RetrievedChunk, VectorStore


class Abstention(Enum):
    """Which of ADR-0003's two abstention layers declined to answer, if either.

    Three states rather than a boolean, because the two layers are not two
    spellings of one outcome: the gate abstains before generation is given
    anything, so its Abstention cites nothing, while the backstop abstains over
    Evidence that was assembled and handed over, so its Evidence is exactly the
    material the model judged insufficient. A surface that showed them alike
    would tell the reader "the corpus does not cover this" in the one case where
    it demonstrably does.

    Which layer fired is decided here and travels as data, so no reader
    downstream re-derives it from the answer's wording -- the words are model
    prose in one case and a constant in the other, and neither is a decision.

    PROMPT_BACKSTOP is unreachable until the backstop has a way to declare
    itself (#21); the state exists as what that signal maps onto.
    """

    NONE = "none"
    DISTANCE_GATE = "distance_gate"
    PROMPT_BACKSTOP = "prompt_backstop"


@dataclass(frozen=True)
class Answer:
    """What one question produced, together with what produced it.

    `chunk_filter` is the scope the question was answered under and not the
    scope in force now: a turn kept on screen while the scope moves on has to
    stay readable as what it was, and a filter read at display time would
    relabel it retroactively. The unfiltered question carries `ChunkFilter()`
    (core/store.py) rather than nothing, so no reader branches on a missing one.
    """

    text: str
    evidence: list[RetrievedChunk]
    abstention: Abstention
    chunk_filter: ChunkFilter


def ask(
    question: str,
    embedder: Embedder,
    store: VectorStore,
    generator: Generator,
    top_k: int,
    distance_threshold: float,
    chunk_filter: ChunkFilter | None = None,
    max_chunks_per_document: int | None = None,
) -> Answer:
    chunk_filter = chunk_filter if chunk_filter is not None else ChunkFilter()

    evidence = retrieve(
        question,
        embedder,
        store,
        top_k=top_k,
        chunk_filter=chunk_filter,
        max_chunks_per_document=max_chunks_per_document,
    )

    if should_abstain(evidence, distance_threshold):
        # Evidence is what generation was given (CONTEXT.md), and the only
        # thing an answer's citations may point at. Nothing was given here, so
        # the abstention carries none and cites none.
        return Answer(
            text=ABSTENTION_TEXT,
            evidence=[],
            abstention=Abstention.DISTANCE_GATE,
            chunk_filter=chunk_filter,
        )

    prompt = build_prompt(question, evidence)
    text = generator.generate(prompt)
    return Answer(
        text=text,
        evidence=evidence,
        abstention=Abstention.NONE,
        chunk_filter=chunk_filter,
    )
