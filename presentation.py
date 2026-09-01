"""The strings both surfaces show a reader, owned by neither of them.

The command line renders these to a terminal and the Week 5 page will render
them to widgets (#22, PLAN.md §第 5 週). Wording that lived in cli.py would
leave the page to re-say the same things in its own words, and two surfaces
describing one Evidence differently is a corpus the reader cannot check: a card
that names a Document one way here and another way there reads as two
Documents.

What is *not* here is how a surface frames these: the terminal's list bullet
and its parenthetical asides are terminal idiom, not wording, and a widget has
its own. Each surface composes; neither writes.

Written over `RetrievedChunk` and `ChunkFilter` -- values the ask seam already
returns (core/ask.py) -- so nothing here re-derives a decision the pipeline
made. In particular the provisional-τ notice is derived from the threshold it
is handed, never from a literal restated per surface.
"""

from config import DISTANCE_THRESHOLD
from core.store import ChunkFilter, RetrievedChunk


def evidence_card(chunk: RetrievedChunk) -> str:
    """One Evidence Chunk as the reader judges it: Document title, Source type,
    Locator.

    One card per Chunk of the Evidence and none besides, which is what makes
    the cards report the answer's real footing: a question scoped to one Domain
    or capped per Document has fewer Chunks to show, and showing anything the
    Evidence does not hold would credit the answer to a Document generation was
    never given (CONTEXT.md: Evidence is the only thing an answer's citations
    may point at).

    The Locator is printed as it arrived and never labelled by shape -- it is a
    heading path today and a page or an anchor as PLAN.md §五 lands, and a card
    that called it a "section" would go wrong the day the first paper is cited
    (CONTEXT.md).
    """
    return f"{chunk.title} ({chunk.source_type}) — {chunk.locator}"


def describe_filter(chunk_filter: ChunkFilter) -> str:
    """The restriction in the reader's words, for a message that has to say
    what the question was narrowed to."""
    return ", ".join(
        f"{label} {value}"
        for label, value in (
            ("domain", chunk_filter.domain),
            ("source type", chunk_filter.source_type),
        )
        if value is not None
    )


def provisional_threshold_notice() -> str | None:
    """What the gate ran on while τ is still hand-set, or None once it is not.

    Every word of it is read off the configured threshold -- the flag, the
    value, and the model it was set for -- rather than off any surface's idea
    of the project's calendar, so the Week 6 sweep flipping that flag in
    config.py is the only edit that removes this. A notice restated as a
    literal would outlive the condition it describes on whichever surface was
    forgotten (ADR-0003).

    Takes no threshold: neither surface has one of its own to pass, and a
    parameter only a test ever fills is a seam that misreports where the value
    comes from.

    None rather than "" for the derived case: a surface asks whether there is a
    notice, and an empty caption or a blank terminal line is a surface having
    rendered nothing without noticing.

    "tau" rather than "τ" is kept from the terminal it was written for, where
    the letter is not encodable on every console this runs on.
    """
    if not DISTANCE_THRESHOLD.provisional:
        return None
    return (
        f"The distance gate used tau={DISTANCE_THRESHOLD.value}, hand-set for "
        f"{DISTANCE_THRESHOLD.embedding_model} and not yet derived from the "
        "eval sweep — see ADR-0003."
    )
