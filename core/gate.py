"""The distance gate: abstention layer 1 of ADR-0003.

Runs after retrieval and before the prompt is built, so a question whose
subject the corpus never covers costs zero generation tokens. It is the cheap,
deliberately permissive layer -- the near-miss questions it lets through are
layer 2's job (the abstention instruction in core/generator.py).
"""

from core.store import RetrievedChunk

ABSTENTION_TEXT = (
    "I don't know — the corpus has nothing close enough to this question to "
    "answer from."
)


def should_abstain(evidence: list[RetrievedChunk], threshold: float) -> bool:
    """True when not one retrieved Chunk lies within τ of the question.

    Distances are cosine and lower-is-closer (ADR-0003), so the gate reads
    `distance > τ` and a Chunk exactly at τ is inside. Evidence that came back
    empty -- an empty corpus -- abstains too: there is nothing to answer from.
    """
    if not evidence:
        return True
    return min(chunk.distance for chunk in evidence) > threshold
