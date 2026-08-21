"""Structured chunking for headed notes (PLAN.md §2.3, §4).

Splits on headings and never straddles one. An undersized section is merged
into a neighboring section instead of becoming its own tiny Chunk -- forward
into the next section normally, or backward into the previous one when it is
the last section in the Document. Either way the merged Chunk carries exactly
one Locator, so "never straddles a heading" holds at the Chunk level even
though the merged text originated under two headings. An oversized section is
split into multiple Chunks that all share that one Locator.

Sized and split by core/tokenization.py, so the rules above hold for a Chinese
note as they do for an English one. A Chunk's text is a verbatim span of the
Document -- nothing here reflows it.

Headings are Markdown's `#` in a `.md` note and a paragraph's style in a
`.docx` one, so what the chunking rules are stated over is a Section -- a
heading path and the body beneath it -- rather than a syntax. Parsing a format
into Sections and chunking them are two jobs: `parse_markdown_sections` does
the first for Markdown, other formats do it for themselves
(ingestion/extraction.py), and every one of them reaches the same
`chunk_sections`. One structured path, one place the rules live -- a note that
arrives as docx cannot chunk by rules that have drifted from the ones a note
that arrives as Markdown chunks by.
"""

import re
from dataclasses import dataclass

from core.tokenization import count_tokens, tokenize

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

LOCATOR_SEPARATOR = " › "


@dataclass(frozen=True)
class ChunkDraft:
    ordinal: int
    locator: str
    text: str


@dataclass(frozen=True)
class Section:
    """One heading's worth of a Document: where it sits, and what is under it.

    `heading_path` runs from the outermost heading down to the one that opened
    this section, so its length is that heading's level and its join is the
    Locator. Empty for text that precedes any heading, which is a Chunk with an
    empty Locator rather than a Chunk with none.

    `body` is the prose alone, headings excluded -- a heading is what a Chunk is
    cited by, not text the Chunk repeats.
    """

    heading_path: tuple[str, ...]
    body: str

    @property
    def locator(self) -> str:
        return LOCATOR_SEPARATOR.join(self.heading_path)


def parse_markdown_sections(text: str) -> list[Section]:
    """Markdown's ATX headings as Sections, in document order.

    A section whose body is entirely whitespace is dropped: a heading with
    nothing under it is a place in the Document, not a span of it, and it has
    no text to embed. The heading still shows up in the path of everything
    nested beneath it, since the path is built from the heading stack rather
    than from the sections that survived.
    """
    sections: list[tuple[tuple[str, ...], list[str]]] = [((), [])]
    stack: list[str] = []

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            stack[level - 1 :] = [match.group(2)]
            sections.append((tuple(stack), []))
        else:
            sections[-1][1].append(line)

    return [
        Section(heading_path=path, body=body)
        for path, lines in sections
        if (body := "\n".join(lines).strip())
    ]


def _split_oversized(text: str, max_tokens: int) -> list[str]:
    """Cut `text` into pieces of at most `max_tokens` tokens each.

    Cuts the source string at token boundaries rather than rejoining token
    strings, so the pieces concatenate back into exactly `text` -- whatever
    sits between the tokens, newlines included, stays where the author put it.
    """
    tokens = tokenize(text)
    if len(tokens) <= max_tokens:
        return [text]

    # The first piece starts at 0 and the last ends at the end of the text, so
    # that anything before the first token or after the last is carried too.
    cuts = [0, *(tokens[i].start for i in range(max_tokens, len(tokens), max_tokens))]
    return [
        text[start:end] for start, end in zip(cuts, [*cuts[1:], len(text)], strict=True)
    ]


def chunk_sections(
    sections: list[Section], min_tokens: int, max_tokens: int
) -> list[ChunkDraft]:
    """The merge-and-split rules of this module's docstring, over any format's
    Sections."""
    chunks: list[ChunkDraft] = []
    pending = ""

    for section in sections:
        combined = f"{pending}\n\n{section.body}".strip() if pending else section.body

        if count_tokens(combined) < min_tokens:
            pending = combined
            continue

        pending = ""
        for piece in _split_oversized(combined, max_tokens):
            chunks.append(
                ChunkDraft(ordinal=len(chunks), locator=section.locator, text=piece)
            )

    if pending:
        if chunks:
            last = chunks[-1]
            chunks[-1] = ChunkDraft(
                ordinal=last.ordinal,
                locator=last.locator,
                text=f"{last.text}\n\n{pending}".strip(),
            )
        else:
            chunks.append(
                ChunkDraft(ordinal=0, locator=sections[-1].locator, text=pending)
            )

    return chunks
