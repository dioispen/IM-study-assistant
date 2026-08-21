"""Builds .pdf fixtures at test time instead of committing them.

The reason tests/docx_fixture.py gives, and more so: docs/corpus-sources.md
asks of a committed fixture that it be deterministic and reviewable in a diff,
and a PDF is a binary container of compressed streams carrying a creation
timestamp. Written from the page list a test passes in, the fixture paper's
prose sits in the test that depends on it, where the diff shows it.

What this cannot cover is the two-column layout a real paper arrives in, since
it is PyMuPDF reading back what PyMuPDF wrote in one column. That is deliberate
rather than a gap to close later: reading a two-column page in the right order
is a property of the extractor and of the file it was handed, not something a
fixture written by the same library could establish (ADR-0007 says what is
attempted there and what is not). What the fixtures do establish is everything
that follows extraction -- that a page number reaches the Locator, that the
windowed path cuts what came back, that a broken file costs the run only
itself.
"""

from pathlib import Path

import pymupdf

# A base CJK font, so a Traditional Chinese fixture round-trips: the corpus is
# zh-tw (PLAN.md), and a paper fixture written in English would leave the
# windowed path exercised only against the one script that whitespace happens
# to work for -- which is how a Chinese note collapsing into a single Chunk
# survived a green suite once already (#11).
_FONT = "china-t"

# The text area of the page, inset from its edges the way a paper's is. Text
# that does not fit is what `write_pdf` refuses to write, so the margin matters
# only in that it makes a page hold a believable amount of prose.
_MARGIN = 72
_FONT_SIZE = 11


def write_pdf(path: Path, pages: list[str]) -> Path:
    """A PDF whose page *n* holds `pages[n - 1]`.

    An empty string is a page with no text layer at all -- what every page of a
    scanned paper is, and what the reader has to tell from a page that is
    simply blank.

    Text too long for its page raises rather than being silently truncated:
    PyMuPDF reports the overflow as a negative return and writes what fit, so a
    fixture whose last paragraph vanished would otherwise fail its test as a
    chunking bug that isn't one.
    """
    document = pymupdf.open()

    for number, body in enumerate(pages, start=1):
        page = document.new_page()
        if not body:
            continue
        rectangle = pymupdf.Rect(
            _MARGIN, _MARGIN, page.rect.width - _MARGIN, page.rect.height - _MARGIN
        )
        overflow = page.insert_textbox(
            rectangle, body, fontname=_FONT, fontsize=_FONT_SIZE
        )
        if overflow < 0:
            raise ValueError(
                f"page {number} of {path.name} does not fit on one page "
                f"(short by {-overflow:.0f} units). Shorten it, or split it "
                "across two pages -- a fixture that loses its tail would fail "
                "its test as a chunking bug rather than as a fixture bug."
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(path))
    document.close()
    return path


def write_encrypted_pdf(path: Path, pages: list[str], password: str = "secret") -> Path:
    """`pages` written to a PDF that cannot be read without `password`.

    Ordinary for a paper downloaded from a publisher, and the failure the
    reader must name rather than report as an empty Document: a corpus owner
    whose paper ingested as "no extractable text" would go looking for a
    scanner, not for the password they already have.
    """
    source = write_pdf(path.parent / f".{path.name}", pages)
    document = pymupdf.open(str(source))
    document.save(
        str(path),
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw=password,
        owner_pw=password,
    )
    document.close()
    source.unlink()
    return path


def truncate(source: Path, target: Path) -> Path:
    """`source` cut off half way, as a sync interrupted mid-copy leaves it.

    The file opens as far as its header and then has no cross-reference table
    to find its pages by -- a failure that surfaces inside whichever library is
    parsing it, which is what the reader must not let out of `ingest_folder`.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    target.write_bytes(data[: len(data) // 2])
    return target
