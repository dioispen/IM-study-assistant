"""Unit tests for recovering a Document's structure from the file it arrived as.

The routing is by file format and nothing else, so these are stated per format:
what a `.md` note yields, what a `.docx` note yields, what a `.pdf` paper
yields, and what happens to a file no reader can make sense of.

The formats do not all yield the same thing, and that is the point of the
routing: `.md` and `.docx` have headings, so they yield Sections for the
structured path, while a `.pdf` has pages and yields Extents for the windowed
one (ADR-0007).
"""

import zipfile

import docx
import pytest

from core.chunking import Extent, Section
from ingestion.extraction import (
    ExtractionError,
    StructuredDocument,
    UnstructuredDocument,
    extract_document,
)
from tests.docx_fixture import BODY, TABLE, corrupt_a_part, resave_untouched, write_docx
from tests.pdf_fixture import truncate, write_encrypted_pdf, write_pdf

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

    note = extract_document(path)

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

    assert extract_document(path).text == raw


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

    note = extract_document(path)

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

    assert [s.locator for s in extract_document(path).sections] == [
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

    [section] = extract_document(path).sections

    assert section.locator == "C 語言 › 前置處理"
    assert section.body == f"#include <stdio.h>\n{ZH_OVERVIEW}"


def test_docx_text_before_any_heading_becomes_a_section_with_an_empty_locator(tmp_path):
    path = write_docx(
        tmp_path / "note.docx",
        [(BODY, ZH_OVERVIEW), ("Heading 1", "分層架構"), (BODY, ZH_LAYERING)],
    )

    sections = extract_document(path).sections

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

    [section] = extract_document(path).sections

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

    assert extract_document(path).sections == []


def test_a_docx_notes_extracted_text_carries_every_heading_and_body(tmp_path):
    # content_hash is taken over this, so it has to carry everything a Chunk
    # could be built from. It is the extracted text rather than the file's
    # bytes because a docx re-saved untouched is a new zip -- new timestamps,
    # reordered parts -- and hashing that re-embeds a Document nobody edited.
    path = write_docx(
        tmp_path / "note.docx",
        [("Heading 1", "OSI 參考模型"), ("Heading 2", "概述"), (BODY, ZH_OVERVIEW)],
    )

    text = extract_document(path).text

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
    assert extract_document(original).text == extract_document(resaved).text


def test_editing_a_docx_changes_its_extracted_text(tmp_path):
    # The other half of the skip: it is only safe if the text moves when the
    # note's words do.
    before = write_docx(
        tmp_path / "before.docx", [("Heading 1", "OSI"), (BODY, ZH_OVERVIEW)]
    )
    after = write_docx(
        tmp_path / "after.docx", [("Heading 1", "OSI"), (BODY, ZH_LAYERING)]
    )

    assert extract_document(before).text != extract_document(after).text


def test_a_docx_that_is_not_a_readable_package_raises_extraction_error(tmp_path):
    path = tmp_path / "corrupt.docx"
    path.write_bytes(b"this is not a zip, let alone a Word package")

    with pytest.raises(ExtractionError):
        extract_document(path)


def test_a_zip_that_is_not_a_word_package_raises_extraction_error(tmp_path):
    # Closer to the real accident than random bytes: a file that opens as a zip
    # and then holds nothing Word would recognise -- a truncated download, or a
    # .zip somebody renamed.
    path = tmp_path / "renamed.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("notes.txt", "not a Word package")

    with pytest.raises(ExtractionError):
        extract_document(path)


def test_a_markdown_note_that_is_not_utf_8_raises_extraction_error(tmp_path):
    path = tmp_path / "note.md"
    path.write_bytes("# 標題\n\n本文\n".encode("big5"))

    with pytest.raises(ExtractionError):
        extract_document(path)


def test_a_file_of_an_unreadable_format_raises_extraction_error(tmp_path):
    # Reached only if the walk hands over something it should have filtered, so
    # it is a bug rather than a bad Document -- but ExtractionError all the
    # same, since costing the folder every file after this one is the worse
    # outcome. `.epub` rather than `.pdf`: PDF has a reader of its own now, and
    # a suffix this test names has to be one no reader claims.
    path = tmp_path / "textbook.epub"
    path.write_bytes(b"PK an ebook nobody here can read")

    with pytest.raises(ExtractionError):
        extract_document(path)


def test_a_docx_whose_xml_does_not_parse_raises_extraction_error(tmp_path):
    # Not a subclass of anything a reader would think to name: it arrives from
    # the XML parser underneath python-docx. Letting it out of here does not
    # report one bad note -- it aborts the walk, so every note after this one
    # in the corpus goes unread and retirement never runs.
    good = write_docx(tmp_path / "good.docx", [("Heading 1", "OSI"), (BODY, ZH_OVERVIEW)])

    corrupt = corrupt_a_part(good, tmp_path / "corrupt.docx")

    with pytest.raises(ExtractionError):
        extract_document(corrupt)


def test_a_truncated_docx_raises_extraction_error(tmp_path):
    good = write_docx(tmp_path / "good.docx", [("Heading 1", "OSI"), (BODY, ZH_OVERVIEW)])
    truncated = tmp_path / "truncated.docx"
    truncated.write_bytes(good.read_bytes()[:400])

    with pytest.raises(ExtractionError):
        extract_document(truncated)


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

    [section] = extract_document(path).sections

    assert section.body.splitlines()[0] == "分層比較"


# PDF papers (#7). A paper has no headings a reader can trust -- and in the
# two-column layout most of them arrive in, none a reader can even find -- so
# what this format offers in place of structure is the page, and what it yields
# is Extents for the windowed path (ADR-0007).

ZH_PAPER_ABSTRACT = (
    "本文提出一套以資源配置圖為基礎的死結偵測方法，並在多核心環境下評估其成本。"
)
ZH_PAPER_METHOD = (
    "偵測程序週期性地掃描等待圖，將已完成的行程自圖中移除，再檢查剩餘節點是否成環。"
)
ZH_PAPER_RESULT = (
    "實驗顯示，當資源種類增加時，偵測的執行時間呈現次線性成長，記憶體用量則維持穩定。"
)


def test_a_pdf_yields_one_extent_per_page(tmp_path):
    path = write_pdf(
        tmp_path / "paper.pdf",
        [ZH_PAPER_ABSTRACT, ZH_PAPER_METHOD, ZH_PAPER_RESULT],
    )

    paper = extract_document(path)

    assert isinstance(paper, UnstructuredDocument)
    assert [extent.locator for extent in paper.extents] == ["p. 1", "p. 2", "p. 3"]


def test_a_pdf_extent_holds_the_text_of_its_own_page(tmp_path):
    # A page number that does not point at the page the words are on is worse
    # than no Locator: a reader follows it, does not find the sentence, and
    # stops trusting every citation the system prints.
    path = write_pdf(
        tmp_path / "paper.pdf",
        [ZH_PAPER_ABSTRACT, ZH_PAPER_METHOD, ZH_PAPER_RESULT],
    )

    by_locator = {p.locator: p.text for p in extract_document(path).extents}

    assert "資源配置圖" in by_locator["p. 1"]
    assert "等待圖" in by_locator["p. 2"]
    assert "次線性成長" in by_locator["p. 3"]
    assert "等待圖" not in by_locator["p. 1"]


def test_a_blank_page_keeps_its_place_in_the_numbering(tmp_path):
    # A Extent per page, empty ones included, because the page number has to
    # be the number printed on the page. Renumbering past a blank one -- a
    # divider, a page holding a full-page figure -- would shift every Locator
    # after it by one and cite the wrong page for the rest of the paper. The
    # empty Extent is dropped later, by the chunker, which cites nothing it
    # has no text for.
    path = write_pdf(tmp_path / "paper.pdf", [ZH_PAPER_ABSTRACT, "", ZH_PAPER_RESULT])

    paper = extract_document(path)

    assert [p.locator for p in paper.extents] == ["p. 1", "p. 2", "p. 3"]
    assert paper.extents[1].text.strip() == ""
    assert "次線性成長" in paper.extents[2].text


def test_a_pdf_is_not_read_as_a_structured_document(tmp_path):
    # The routing criterion, asserted as a type rather than inferred from the
    # Chunks: a PDF that came back with Sections would be chunked by rules
    # stated over headings it does not have, and the first heading a two-column
    # extractor invents becomes a Locator citing a place in the paper that is
    # not there.
    path = write_pdf(tmp_path / "paper.pdf", [ZH_PAPER_ABSTRACT])

    assert not isinstance(extract_document(path), StructuredDocument)


def test_a_pdf_with_no_text_layer_at_all_raises_extraction_error(tmp_path):
    # A scanned paper: every page an image, not a word between them. Named as
    # its own failure rather than reported as a Document that yielded no
    # Chunks, because the two ask for opposite things from the corpus owner --
    # one wants OCR, the other wants a file that is not empty.
    path = write_pdf(tmp_path / "scan.pdf", ["", "", ""])

    with pytest.raises(ExtractionError, match="text layer"):
        extract_document(path)


def test_an_encrypted_pdf_raises_extraction_error_naming_the_password(tmp_path):
    # Ordinary for a paper downloaded from a publisher. Reported as an empty
    # Document it would send its owner looking for a scanner, when what they
    # need is the password they already have.
    path = write_encrypted_pdf(tmp_path / "paper.pdf", [ZH_PAPER_ABSTRACT])

    with pytest.raises(ExtractionError, match="password"):
        extract_document(path)


def test_a_pdf_that_is_not_a_pdf_raises_extraction_error(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"this is not a PDF, whatever its name says")

    with pytest.raises(ExtractionError):
        extract_document(path)


def test_a_truncated_pdf_raises_extraction_error(tmp_path):
    # The failure a sync interrupted mid-copy leaves: a header that opens and a
    # cross-reference table that is not there. It arrives from inside PyMuPDF,
    # so letting it out would abort the walk rather than report one bad file.
    good = write_pdf(tmp_path / "good.pdf", [ZH_PAPER_ABSTRACT, ZH_PAPER_METHOD])

    with pytest.raises(ExtractionError):
        extract_document(truncate(good, tmp_path / "truncated.pdf"))


def test_a_pdfs_extracted_text_is_hashed_rather_than_its_bytes(tmp_path):
    # A PDF's bytes carry a creation timestamp and an object layout, so two
    # files holding the same paper are not the same bytes. Hashing what was
    # extracted asks the question the skip wants answered -- have the paper's
    # words changed? -- rather than a question about the file (ADR-0006).
    first = write_pdf(tmp_path / "first.pdf", [ZH_PAPER_ABSTRACT, ZH_PAPER_METHOD])
    second = write_pdf(tmp_path / "second.pdf", [ZH_PAPER_ABSTRACT, ZH_PAPER_METHOD])

    assert first.read_bytes() != second.read_bytes()
    assert extract_document(first).text == extract_document(second).text


def test_repagination_moves_a_pdfs_extracted_text(tmp_path):
    # The same words across a different number of pages is a changed Document,
    # because every Locator after the break has moved. A hash blind to that
    # would leave the corpus citing page numbers the file no longer has, with
    # no run able to detect it (ADR-0001).
    together = write_pdf(tmp_path / "together.pdf", [f"{ZH_PAPER_ABSTRACT}{ZH_PAPER_METHOD}"])
    apart = write_pdf(tmp_path / "apart.pdf", [ZH_PAPER_ABSTRACT, ZH_PAPER_METHOD])

    assert extract_document(together).text != extract_document(apart).text


def test_a_pdf_extent_carries_no_page_furniture_beyond_its_text(tmp_path):
    # The Extent is what gets embedded, so what it holds is the page's prose.
    # Nothing here reflows it -- the line breaks the layout put in stay where
    # they are, exactly as a Chunk of a note is a verbatim span of the note.
    path = write_pdf(tmp_path / "paper.pdf", [ZH_PAPER_ABSTRACT])

    [extent] = extract_document(path).extents

    assert isinstance(extent, Extent)
    assert extent.text.replace("\n", "").startswith("本文提出一套")
