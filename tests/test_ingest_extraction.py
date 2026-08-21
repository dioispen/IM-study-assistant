"""Unit tests for recovering a note's Sections from the file it arrived as.

The routing is by file format and nothing else, so these are stated per format:
what a `.md` note yields, what a `.docx` note yields, and what happens to a file
neither reader can make sense of.
"""

import zipfile

import docx
import pytest

from core.chunking import Section
from ingestion.extraction import ExtractionError, extract_note
from tests.docx_fixture import BODY, TABLE, corrupt_a_part, resave_untouched, write_docx

ZH_OVERVIEW = "分層的目的是解耦，每一層只需要理解自己的職責。"
ZH_LAYERING = "應用層之下依序是傳輸層、網路層與連結層。"


def _write_markdown(tmp_path, text, name="note.md"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_a_markdown_note_yields_one_section_per_heading(tmp_path):
    path = _write_markdown(
        tmp_path,
        f"# OSI 參考模型\n\n## 概述\n\n{ZH_OVERVIEW}\n\n## 分層架構\n\n{ZH_LAYERING}\n",
    )

    note = extract_note(path)

    assert note.sections == [
        Section(heading_path=("OSI 參考模型", "概述"), body=ZH_OVERVIEW),
        Section(heading_path=("OSI 參考模型", "分層架構"), body=ZH_LAYERING),
    ]


def test_a_markdown_notes_extracted_text_is_the_file_verbatim(tmp_path):
    # What content_hash is taken over, and it must stay the file's own text:
    # every Markdown Document already in a registry was hashed this way, and a
    # different rendering here would re-embed the whole corpus on the next run
    # for a change nobody made to a note (ADR-0001).
    raw = f"# OSI 參考模型\n\n## 概述\n\n{ZH_OVERVIEW}\n"
    path = _write_markdown(tmp_path, raw)

    assert extract_note(path).text == raw


def test_a_docx_notes_heading_styles_become_the_heading_path(tmp_path):
    path = write_docx(
        tmp_path / "osi_model.docx",
        [
            ("Heading 1", "OSI 參考模型"),
            ("Heading 2", "概述"),
            (BODY, ZH_OVERVIEW),
            ("Heading 2", "分層架構"),
            (BODY, ZH_LAYERING),
        ],
    )

    note = extract_note(path)

    assert note.sections == [
        Section(heading_path=("OSI 參考模型", "概述"), body=ZH_OVERVIEW),
        Section(heading_path=("OSI 參考模型", "分層架構"), body=ZH_LAYERING),
    ]


def test_a_docx_heading_stack_resets_on_a_shallower_heading(tmp_path):
    # The rule Markdown's `#` levels already follow, read off paragraph styles
    # instead: a Heading 1 after a Heading 2 opens a new path rather than
    # nesting under the old one.
    path = write_docx(
        tmp_path / "note.docx",
        [
            ("Heading 1", "傳輸控制協定"),
            ("Heading 2", "三向交握"),
            (BODY, ZH_OVERVIEW),
            ("Heading 1", "OSI 參考模型"),
            ("Heading 2", "分層架構"),
            (BODY, ZH_LAYERING),
        ],
    )

    assert [s.locator for s in extract_note(path).sections] == [
        "傳輸控制協定 › 三向交握",
        "OSI 參考模型 › 分層架構",
    ]


def test_docx_body_text_beginning_with_a_hash_is_not_read_as_a_heading(tmp_path):
    # A study note quoting an include directive is prose, and it is prose the
    # docx itself marks as prose. Nothing routes docx through Markdown syntax,
    # so the line stays in the body it was written in rather than becoming the
    # heading the Chunk is then cited by.
    path = write_docx(
        tmp_path / "note.docx",
        [
            ("Heading 1", "C 語言"),
            ("Heading 2", "前置處理"),
            (BODY, "#include <stdio.h>"),
            (BODY, ZH_OVERVIEW),
        ],
    )

    [section] = extract_note(path).sections

    assert section.locator == "C 語言 › 前置處理"
    assert section.body == f"#include <stdio.h>\n{ZH_OVERVIEW}"


def test_docx_text_before_any_heading_becomes_a_section_with_an_empty_locator(tmp_path):
    path = write_docx(
        tmp_path / "note.docx",
        [(BODY, ZH_OVERVIEW), ("Heading 1", "分層架構"), (BODY, ZH_LAYERING)],
    )

    sections = extract_note(path).sections

    assert sections[0] == Section(heading_path=(), body=ZH_OVERVIEW)
    assert sections[0].locator == ""


def test_a_docx_table_keeps_its_text_under_the_heading_it_sits_beneath(tmp_path):
    # Word's paragraphs and its tables are different things in the package, and
    # a reader that walks only the paragraphs drops the table silently -- a
    # note whose comparison table is the part worth retrieving would answer
    # nothing, and the run would report an ordinary success.
    path = write_docx(
        tmp_path / "note.docx",
        [
            ("Heading 1", "OSI 參考模型"),
            ("Heading 2", "分層架構"),
            (TABLE, [["層", "職責"], ["傳輸層", "端點到端點的可靠傳遞"]]),
        ],
    )

    [section] = extract_note(path).sections

    assert section.locator == "OSI 參考模型 › 分層架構"
    assert "端點到端點的可靠傳遞" in section.body
    assert "傳輸層" in section.body


def test_an_empty_docx_heading_contributes_no_section(tmp_path):
    # An ideographic space is whitespace, so this note has headings and no
    # prose -- the same shape `.md` already drops, so that what reaches the
    # registry is never a Document contributing no Chunk to any answer.
    path = write_docx(
        tmp_path / "note.docx",
        [("Heading 1", "OSI 參考模型"), ("Heading 2", "概述"), (BODY, "　　")],
    )

    assert extract_note(path).sections == []


def test_a_docx_notes_extracted_text_carries_every_heading_and_body(tmp_path):
    # content_hash is taken over this, so it has to carry everything a Chunk
    # could be built from. It is the extracted text rather than the file's
    # bytes because a docx re-saved untouched is a new zip -- new timestamps,
    # reordered parts -- and hashing that re-embeds a Document nobody edited.
    path = write_docx(
        tmp_path / "note.docx",
        [("Heading 1", "OSI 參考模型"), ("Heading 2", "概述"), (BODY, ZH_OVERVIEW)],
    )

    text = extract_note(path).text

    assert "OSI 參考模型" in text
    assert "概述" in text
    assert ZH_OVERVIEW in text


def test_an_untouched_docx_resaved_extracts_to_the_same_text(tmp_path):
    original = write_docx(
        tmp_path / "original.docx",
        [("Heading 1", "OSI 參考模型"), ("Heading 2", "概述"), (BODY, ZH_OVERVIEW)],
    )

    resaved = resave_untouched(original, tmp_path / "resaved.docx")

    assert original.read_bytes() != resaved.read_bytes()
    assert extract_note(original).text == extract_note(resaved).text


def test_editing_a_docx_changes_its_extracted_text(tmp_path):
    # The other half of the skip: it is only safe if the text moves when the
    # note's words do.
    before = write_docx(
        tmp_path / "before.docx", [("Heading 1", "OSI"), (BODY, ZH_OVERVIEW)]
    )
    after = write_docx(
        tmp_path / "after.docx", [("Heading 1", "OSI"), (BODY, ZH_LAYERING)]
    )

    assert extract_note(before).text != extract_note(after).text


def test_a_docx_that_is_not_a_readable_package_raises_extraction_error(tmp_path):
    path = tmp_path / "corrupt.docx"
    path.write_bytes(b"this is not a zip, let alone a Word package")

    with pytest.raises(ExtractionError):
        extract_note(path)


def test_a_zip_that_is_not_a_word_package_raises_extraction_error(tmp_path):
    # Closer to the real accident than random bytes: a file that opens as a zip
    # and then holds nothing Word would recognise -- a truncated download, or a
    # .zip somebody renamed.
    path = tmp_path / "renamed.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("notes.txt", "not a Word package")

    with pytest.raises(ExtractionError):
        extract_note(path)


def test_a_markdown_note_that_is_not_utf_8_raises_extraction_error(tmp_path):
    path = tmp_path / "note.md"
    path.write_bytes("# 標題\n\n本文\n".encode("big5"))

    with pytest.raises(ExtractionError):
        extract_note(path)


def test_a_file_of_an_unreadable_format_raises_extraction_error(tmp_path):
    # Reached only if the walk hands over something it should have filtered, so
    # it is a bug rather than a bad note -- but ExtractionError all the same,
    # since costing the folder every note after this one is the worse outcome.
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-1.7")

    with pytest.raises(ExtractionError):
        extract_note(path)


def test_a_docx_whose_xml_does_not_parse_raises_extraction_error(tmp_path):
    # Not a subclass of anything a reader would think to name: it arrives from
    # the XML parser underneath python-docx. Letting it out of here does not
    # report one bad note -- it aborts the walk, so every note after this one
    # in the corpus goes unread and retirement never runs.
    good = write_docx(tmp_path / "good.docx", [("Heading 1", "OSI"), (BODY, ZH_OVERVIEW)])

    corrupt = corrupt_a_part(good, tmp_path / "corrupt.docx")

    with pytest.raises(ExtractionError):
        extract_note(corrupt)


def test_a_truncated_docx_raises_extraction_error(tmp_path):
    good = write_docx(tmp_path / "good.docx", [("Heading 1", "OSI"), (BODY, ZH_OVERVIEW)])
    truncated = tmp_path / "truncated.docx"
    truncated.write_bytes(good.read_bytes()[:400])

    with pytest.raises(ExtractionError):
        extract_note(truncated)


def test_a_cell_merged_across_columns_is_read_once(tmp_path):
    # `row.cells` yields one entry per grid column, so a merged header comes
    # back once per column it spans. Repeating it inflates the note's own
    # vocabulary in the Chunk a reader is shown, and merged headers are
    # ordinary in exactly the comparison tables tables are kept for.
    path = write_docx(
        tmp_path / "note.docx",
        [("Heading 1", "OSI 參考模型"), (TABLE, [["層", "職責"], ["傳輸層", "可靠傳遞"]])],
    )
    document = docx.Document(str(path))
    table = document.tables[0]
    table.rows[0].cells[0].merge(table.rows[0].cells[1]).text = "分層比較"
    document.save(path)

    [section] = extract_note(path).sections

    assert section.body.splitlines()[0] == "分層比較"
