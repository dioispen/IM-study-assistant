"""The two chunking paths, and the rules each is stated over (PLAN.md §2.3, §4).

Which one a Document takes follows from what its format could tell a reader
about it, and the choice is made once, in ingestion/common.py (ADR-0007). A
Document whose format marks where its author changed subject arrives here as
Sections and is chunked by `chunk_sections`; one whose format offers only a
place to cite arrives as Extents and is windowed by `chunk_windows`. Both
paths live here so that the difference between them is one file's worth of
reading rather than an archaeology across two.

## The structured path (`chunk_sections`)

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

## The windowed path (`chunk_windows`)

Fixed-size windows with overlap, cut over the Document's whole text. A page
break is not a boundary the way a heading is -- it is where the printer ran out
of paper -- so it is straddled rather than respected, and each Chunk is cited
by the Extent it starts in. Sized by the same core/tokenization.py measure, on
which ADR-0004 insists by name.

The two paths are not alternatives to choose between per Document by taste.
Each is the only set of rules that can be stated over what its input offers:
merge-and-split needs headings to be about, and a window needs nothing at all,
which is exactly why it is what is left when a format has nothing to offer.
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


@dataclass(frozen=True)
class Extent:
    """A stretch of an unstructured Document that one Locator names.

    The windowed path's counterpart to a Section, and deliberately weaker than
    one. A Section is a boundary the chunker respects; an Extent is only what
    tells a window where in the Document it came from -- a PDF's page, a
    fetched article's body. Its `locator` is written by whichever reader
    produced it, in that format's own terms, so that this module needs to know
    what a page is no more than `chunk_sections` needs to know what a `#` is.
    """

    locator: str
    text: str


# What is written between two Extents when they are joined into the one text
# the windows are cut from. A newline rather than nothing, so that the last
# word of a page and the first word of the next are two tokens -- run together
# they would be one token that occurs nowhere in the Document, and a question
# using either word would miss the Chunk holding both.
EXTENT_SEPARATOR = "\n"


@dataclass(frozen=True)
class _PlacedExtent:
    """Where one Extent landed in the joined text, and what it is cited by.

    The pair exists because the windows are cut over the joined text while the
    Locators belong to the Extents that went into it -- so something has to
    remember which offsets came from where.
    """

    start: int
    locator: str


def _join_extents(extents: list[Extent]) -> tuple[str, list[_PlacedExtent]]:
    """`extents` as one text, with where each one landed in it.

    Extents with no text are dropped -- a blank page in a scan is a place in
    the Document rather than a span of it, exactly as a heading with nothing
    under it is dropped from the structured path. It has no text to embed, and
    keeping it would leave a Locator that cites a page holding nothing.

    Dropping one costs no numbering, because the number is already in the
    Locator its reader wrote: it is `ingestion/extraction.py` that must yield an
    Extent per page including the empty ones, since it is the only place that
    knows a page's number is its position in the file.
    """
    text_parts: list[str] = []
    placed: list[_PlacedExtent] = []
    offset = 0

    for extent in extents:
        if not extent.text.strip():
            continue
        if text_parts:
            offset += len(EXTENT_SEPARATOR)
            text_parts.append(EXTENT_SEPARATOR)
        placed.append(_PlacedExtent(start=offset, locator=extent.locator))
        text_parts.append(extent.text)
        offset += len(extent.text)

    return "".join(text_parts), placed


def _locator_at(offset: int, placed: list[_PlacedExtent]) -> str:
    """The Locator of the Extent that `offset` falls in.

    A linear scan rather than a bisect: a Document is a few hundred pages at
    most, and this runs once per Chunk.
    """
    locator = placed[0].locator
    for extent in placed:
        if extent.start > offset:
            break
        locator = extent.locator
    return locator


def chunk_windows(
    extents: list[Extent], window: int, overlap: int
) -> list[ChunkDraft]:
    """Fixed-size windows with overlap over a Document that has no structure.

    The windows are cut over the whole Document rather than page by page: a
    page break is where the printer ran out of paper, not where the author
    changed subject, so a paragraph that runs across one stays whole inside
    some window and the overlap keeps working across it. That is the difference
    from `chunk_sections`, whose boundaries are the author's own and therefore
    never straddled.

    Each Chunk is cited by the Extent its text *starts* in -- the page a
    reader following the citation turns to and reads forward from. A Chunk that
    straddles two pages is cited by the first, not by a range: a range would
    need this module to know how to count pages, or an article's anchors, or
    whatever the next unstructured Source offers, and it tells the reader
    nothing they do not already have from where to start.

    Sized by core/tokenization.py, which ADR-0004 requires of this path by
    name: a window sized in whitespace-words reads a whole Chinese page as one
    or two words, never reaches its own size, and hands back the page as a
    single Chunk -- the failure that collapsed Chinese notes on the structured
    path, arriving here by the same route.
    """
    if window < 1:
        raise ValueError(
            f"window={window} cuts no text into Chunks; a window is a number of "
            "tokens and has to be at least 1."
        )
    if not 0 <= overlap < window:
        # Not a bad chunking but no chunking: at overlap == window the window
        # never advances, so the Document becomes an unbounded number of copies
        # of its opening tokens.
        raise ValueError(
            f"overlap={overlap} must be at least 0 and less than window={window}; "
            "an overlap that reaches the window size leaves the window standing "
            "still, and the Document is cut into copies of its first tokens."
        )

    unnamed = [n for n, extent in enumerate(extents, start=1) if not extent.locator]
    if unnamed:
        # Enforced here rather than trusted of each reader, because this is the
        # path that makes the promise: an unstructured Document has nothing but
        # its Locator to be cited by, so a Chunk without one is a Chunk no
        # reader can follow back and no reader can tell is unfollowable -- it
        # renders as a source card ending in a dash. The structured path admits
        # an empty Locator on purpose (prose above a note's first heading is
        # genuinely nowhere in particular); here there is no such place, since
        # what the format offers *is* the place.
        raise ValueError(
            f"extent {unnamed[0]} of {len(extents)} has an empty locator, and "
            "the windowed path has nothing else to cite a Chunk by. Whichever "
            "reader produced these owes every Extent a Locator in its own "
            "format's terms -- a page, an anchor, a URL."
        )

    text, placed = _join_extents(extents)
    tokens = tokenize(text)
    if not tokens:
        return []

    stride = window - overlap
    # The final window is the first one to reach the end of the Document, and
    # it always holds more than the overlap it repeats: the window before it
    # did *not* reach the end, so there is at least one token past that
    # window's end for this one to carry. A tail shorter than that is already
    # inside the previous Chunk, and emitting it would spend a retrieval slot
    # on a duplicate.
    cuts = []
    cut = 0
    while True:
        cuts.append(cut)
        if cut + window >= len(tokens):
            break
        cut += stride

    return [
        ChunkDraft(
            ordinal=ordinal,
            locator=_locator_at(begin, placed),
            text=text[begin:end],
        )
        for ordinal, (begin, end) in enumerate(
            # From the first token's start, except for the opening window,
            # which starts at 0 so that anything before the first token is
            # carried; to the last token's end, except for the final window,
            # which runs to the end of the text for the same reason.
            (
                tokens[first].start if first else 0,
                len(text)
                if first + window >= len(tokens)
                else tokens[first + window - 1].end,
            )
            for first in cuts
        )
    ]
