"""The ask seam: retrieve Evidence, gate on distance, generate an answer.

Kept separate from the CLI so tests can inject a fake Embedder and Generator
and run fully offline (see ADR-0002 -- these tests assert on Evidence, not on
generated text).
"""

from dataclasses import dataclass

from core.embedder import Embedder
from core.gate import ABSTENTION_TEXT, should_abstain
from core.generator import Generator, build_prompt
from core.retriever import retrieve
from core.store import ChunkFilter, RetrievedChunk, VectorStore


@dataclass(frozen=True)
class Answer:
    text: str
    evidence: list[RetrievedChunk]
    abstained: bool


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
        return Answer(text=ABSTENTION_TEXT, evidence=[], abstained=True)

    prompt = build_prompt(question, evidence)
    text = generator.generate(prompt)
    return Answer(text=text, evidence=evidence, abstained=False)
