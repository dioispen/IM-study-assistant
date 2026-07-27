"""Structured chunking for headed Markdown notes (PLAN.md §2.3, §4).

Splits on headings and never straddles one. An undersized section is merged
into a neighboring section instead of becoming its own tiny Chunk -- forward
into the next section normally, or backward into the previous one when it is
the last section in the Document. Either way the merged Chunk carries exactly
one Locator, so "never straddles a heading" holds at the Chunk level even
though the merged text originated under two headings. An oversized section is
split into multiple Chunks that all share that one Locator.
"""

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

LOCATOR_SEPARATOR = " › "


@dataclass(frozen=True)
class ChunkDraft:
    ordinal: int
    locator: str
    text: str


@dataclass
class _Section:
    heading_path: list[str]
    body_lines: list[str]

    @property
    def locator(self) -> str:
        return LOCATOR_SEPARATOR.join(self.heading_path)


def _approx_tokens(text: str) -> int:
    """Word count as a stand-in for token count -- see config.py."""
    return len(text.split())


def _parse_sections(text: str) -> list[_Section]:
    sections: list[_Section] = []
    stack: list[str] = []
    current = _Section(heading_path=[], body_lines=[])
    sections.append(current)

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            heading = match.group(2)
            stack[level - 1 :] = [heading]
            current = _Section(heading_path=list(stack), body_lines=[])
            sections.append(current)
        else:
            current.body_lines.append(line)

    return [s for s in sections if "\n".join(s.body_lines).strip()]


def _split_oversized(text: str, max_tokens: int) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)
    ]


def chunk_markdown(text: str, min_tokens: int, max_tokens: int) -> list[ChunkDraft]:
    sections = _parse_sections(text)

    chunks: list[ChunkDraft] = []
    pending = ""

    for section in sections:
        body = "\n".join(section.body_lines).strip()
        combined = f"{pending}\n\n{body}".strip() if pending else body

        if _approx_tokens(combined) < min_tokens:
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
