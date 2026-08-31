"""Prompt assembly, the generation call, and the backstop's declared signal.

The prompt is the abstention layer 2 backstop of ADR-0003: independent of the
distance gate, the model itself is told to refuse when the Evidence does not
support an answer. This module owns both halves of that contract -- the
sentinel it asks for and the reading of the reply -- because a contract split
across two modules is one that can drift apart while both halves still pass
their own tests (ADR-0008).
"""

from typing import Protocol

from core.openai_client import default_openai_client
from core.store import RetrievedChunk

# The one line the backstop declares itself with. Uppercase and underscored so
# that nothing in an ordinary answer -- in either language the corpus is
# written in -- is this string on its own line, and short enough that a model
# reproduces it exactly rather than approximately.
BACKSTOP_SENTINEL = "INSUFFICIENT_EVIDENCE"

# What the reader sees when the sentinel was declared. Deliberately not
# core/gate.py's ABSTENTION_TEXT: the gate found nothing close enough to answer
# from, whereas here Evidence was retrieved, handed over and judged
# insufficient -- and it is cited alongside this, so telling the reader "the
# corpus has nothing close" over a list of Chunks would contradict the cards.
BACKSTOP_ABSTENTION_TEXT = (
    "I don't know — the Evidence retrieved for this question does not answer "
    "it. It is cited with this answer, so you can judge that for yourself."
)

_INSTRUCTIONS = f"""You are a study assistant answering from the Evidence below only.

- Answer only using the Evidence. Do not use outside knowledge.
- Cite the Chunk(s) you rely on for each claim, referring to them by their title and Locator.
- If the Evidence does not contain the answer, reply with exactly this line and nothing else:
{BACKSTOP_SENTINEL}
  Do not apologise, explain, or add anything to that line. Do not guess.
"""


def declares_abstention(text: str) -> bool:
    """True when the generated text is the backstop sentinel and nothing else.

    The whole of what is read out of a generated answer (ADR-0008). Exact and
    case-sensitive: a declaration is a signal the prompt asked for, so
    recognising an approximation of it would be the "does this sound like a
    refusal" heuristic this design exists to avoid -- and that heuristic is
    what would put model prose inside the test suite's assertions (ADR-0002).

    Surrounding whitespace is stripped and nothing else is: a trailing newline
    is how a chat completion ends a line rather than the model saying something
    besides the sentinel, while a sentinel with prose beside it is a model that
    answered *and* hedged, which is an answer.
    """
    return text.strip() == BACKSTOP_SENTINEL


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
        # `content` is optional in the SDK -- a completion that produced no
        # text at all comes back as None. Coerced here rather than downstream
        # so this really does satisfy `Generator`, which the pipeline reads as
        # a str: an empty answer is an empty answer, neither a declared
        # Abstention nor an AttributeError out of `declares_abstention`.
        return response.choices[0].message.content or ""


class FakeGenerator:
    """Deterministic, offline stand-in for tests -- never calls an API.

    Answers rather than abstains: its text is not the sentinel, so a seam test
    using it gets an ordinary Answer (test_generator.py pins that).
    """

    def generate(self, prompt: str) -> str:
        return "FAKE ANSWER (offline test double; not asserted on generated text -- ADR-0002)"
