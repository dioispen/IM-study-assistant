"""Prompt assembly and the generation call.

The prompt is the abstention layer 2 backstop of ADR-0003: independent of the
distance gate, the model itself is told to refuse when the Evidence does not
support an answer.
"""

from typing import Protocol

from core.openai_client import default_openai_client
from core.store import RetrievedChunk

_INSTRUCTIONS = """You are a study assistant answering from the Evidence below only.

- Answer only using the Evidence. Do not use outside knowledge.
- Cite the Chunk(s) you rely on for each claim, referring to them by their title and Locator.
- If the Evidence does not contain the answer, say you don't know rather than guessing.
"""


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


def build_prompt(question: str, evidence: list[RetrievedChunk]) -> str:
    if evidence:
        evidence_block = "\n\n".join(
            f"[{i}] Title: {chunk.title} | Source type: {chunk.source_type} | "
            f"Locator: {chunk.locator}\n{chunk.text}"
            for i, chunk in enumerate(evidence, start=1)
        )
    else:
        evidence_block = "(no Evidence retrieved)"

    return f"{_INSTRUCTIONS}\nEvidence:\n{evidence_block}\n\nQuestion: {question}\n"


class OpenAIGenerator:
    def __init__(self, model: str, client=None):
        self._client = client or default_openai_client()
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class FakeGenerator:
    """Deterministic, offline stand-in for tests -- never calls an API."""

    def generate(self, prompt: str) -> str:
        return "FAKE ANSWER (offline test double; not asserted on generated text -- ADR-0002)"
