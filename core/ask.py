"""The ask seam: retrieve Evidence, build the prompt, generate an answer.

Kept separate from the CLI so tests can inject a fake Embedder and Generator
and run fully offline (see ADR-0002 -- these tests assert on Evidence, not on
generated text).
"""

from dataclasses import dataclass

from core.embedder import Embedder
from core.generator import Generator, build_prompt
from core.retriever import retrieve
from core.store import RetrievedChunk, VectorStore


@dataclass(frozen=True)
class Answer:
    text: str
    evidence: list[RetrievedChunk]


def ask(
    question: str,
    embedder: Embedder,
    store: VectorStore,
    generator: Generator,
    top_k: int,
) -> Answer:
    evidence = retrieve(question, embedder, store, top_k=top_k)
    prompt = build_prompt(question, evidence)
    text = generator.generate(prompt)
    return Answer(text=text, evidence=evidence)
