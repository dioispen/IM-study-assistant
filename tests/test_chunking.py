import pytest

from config import MAX_SECTION_TOKENS, MIN_SECTION_TOKENS
from core.chunking import (
    Extent,
    chunk_sections,
    chunk_windows,
    parse_markdown_sections,
)
from core.tokenization import count_tokens


def chunk_markdown(text: str, min_tokens: int, max_tokens: int):
    """The two halves of the structured path, composed as this module reads it.

    Every rule below is about headed prose rather than about `#`, so each test
    states one over a Markdown document and reads the Chunks back. Production
    parses each format into Sections itself (ingestion/extraction.py) and
    nothing there needs this pairing, which is why it lives here rather than
    beside them.
    """
    return chunk_sections(
        parse_markdown_sections(text), min_tokens=min_tokens, max_tokens=max_tokens
    )


NORMAL = "word " * 20  # 20 tokens: within [10, 30] for most tests below


def test_splits_on_headings_without_straddling():
    text = f"# BST\n\n## Insertion\n\n{NORMAL}\n\n## Deletion\n\n{NORMAL}"

    chunks = chunk_markdown(text, min_tokens=10, max_tokens=30)

    assert [c.locator for c in chunks] == ["BST › Insertion", "BST › Deletion"]
    assert "Insertion" not in chunks[1].text
    assert "Deletion" not in chunks[0].text


def test_ordinals_are_sequential_from_zero():
    text = f"# BST\n\n## Insertion\n\n{NORMAL}\n\n## Deletion\n\n{NORMAL}"

    chunks = chunk_markdown(text, min_tokens=10, max_tokens=30)

    assert [c.ordinal for c in chunks] == [0, 1]


def test_oversized_section_splits_into_multiple_chunks_sharing_one_locator():
    big_section = "word " * 100
    text = f"# BST\n\n## Insertion\n\n{big_section}"

    chunks = chunk_markdown(text, min_tokens=10, max_tokens=30)

    assert len(chunks) > 1
    assert {c.locator for c in chunks} == {"BST › Insertion"}
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    # No text is dropped by the split.
    assert " ".join(c.text for c in chunks).split() == big_section.split()


def test_undersized_section_merges_forward_into_the_next_section():
    tiny = "word " * 3  # below min_tokens=10
    text = f"# BST\n\n## Overview\n\n{tiny}\n\n## Insertion\n\n{NORMAL}"

    chunks = chunk_markdown(text, min_tokens=10, max_tokens=30)

    assert len(chunks) == 1
    assert chunks[0].locator == "BST › Insertion"
    assert "word word word" in chunks[0].text
    assert NORMAL.strip() in chunks[0].text


def test_trailing_undersized_section_merges_backward_when_no_next_section_exists():
    tiny = "word " * 3
    text = f"# BST\n\n## Insertion\n\n{NORMAL}\n\n## Summary\n\n{tiny}"

    chunks = chunk_markdown(text, min_tokens=10, max_tokens=30)

    assert len(chunks) == 1
    assert chunks[0].locator == "BST › Insertion"
    assert "word word word" in chunks[0].text


def test_single_undersized_section_with_no_neighbors_stands_alone():
    tiny = "word " * 3
    text = f"# BST\n\n## Overview\n\n{tiny}"

    chunks = chunk_markdown(text, min_tokens=10, max_tokens=30)

    assert len(chunks) == 1
    assert chunks[0].locator == "BST › Overview"


def test_text_before_any_heading_uses_an_empty_locator():
    text = f"{NORMAL}\n\n## Insertion\n\n{NORMAL}"

    chunks = chunk_markdown(text, min_tokens=10, max_tokens=30)

    assert chunks[0].locator == ""


ZH_CONDITIONS = (
    "死結發生時，以下四個條件必定同時成立：互斥、佔有並等待、不可搶奪、循環等待。"
    "互斥指一項資源在同一時間只能由一個行程持有；佔有並等待指行程在持有資源的同時，"
    "又去請求另一項已被佔用的資源。"
)
ZH_GRAPH = (
    "資源配置圖以節點代表行程與資源，以有向邊代表請求邊與配置邊。若圖中不存在環路，"
    "系統必定沒有死結；若每一種資源都只有一個實例，環路的存在就是死結的充分且必要條件。"
)
ZH_STRATEGIES = (
    "處理死結的策略分為預防、避免、偵測與復原。預防的做法是讓四個必要條件之中至少一個"
    "無法成立；避免的做法則是在配置資源之前先檢查系統是否仍處於安全狀態，銀行家演算法"
    "即屬於此類。"
)


def test_chinese_sections_each_become_their_own_chunk_at_configured_thresholds():
    text = (
        f"# 死結\n\n## 四個必要條件\n\n{ZH_CONDITIONS}\n\n"
        f"## 資源配置圖\n\n{ZH_GRAPH}\n\n"
        f"## 處理策略\n\n{ZH_STRATEGIES}"
    )

    chunks = chunk_markdown(
        text, min_tokens=MIN_SECTION_TOKENS, max_tokens=MAX_SECTION_TOKENS
    )

    assert [c.locator for c in chunks] == [
        "死結 › 四個必要條件",
        "死結 › 資源配置圖",
        "死結 › 處理策略",
    ]
    assert [c.ordinal for c in chunks] == [0, 1, 2]


def test_oversized_chinese_section_splits_into_chunks_sharing_one_locator():
    big_section = f"{ZH_CONDITIONS}\n{ZH_GRAPH}\n{ZH_STRATEGIES}"
    text = f"# 死結\n\n## 四個必要條件\n\n{big_section}"

    chunks = chunk_markdown(text, min_tokens=10, max_tokens=30)

    assert len(chunks) > 1
    assert {c.locator for c in chunks} == {"死結 › 四個必要條件"}
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_oversized_split_reconstitutes_the_source_text_exactly():
    big_section = f"{ZH_CONDITIONS}\n{ZH_GRAPH}\n{ZH_STRATEGIES}"
    text = f"# 死結\n\n## 四個必要條件\n\n{big_section}"

    chunks = chunk_markdown(text, min_tokens=10, max_tokens=30)

    assert "".join(c.text for c in chunks) == big_section


def test_undersized_chinese_section_merges_forward_into_the_next_section():
    tiny = "死結的定義如下。"  # below min_tokens=10
    text = f"# 死結\n\n## 概述\n\n{tiny}\n\n## 資源配置圖\n\n{ZH_GRAPH}"

    chunks = chunk_markdown(text, min_tokens=10, max_tokens=300)

    assert len(chunks) == 1
    assert chunks[0].locator == "死結 › 資源配置圖"
    assert tiny in chunks[0].text
    assert ZH_GRAPH in chunks[0].text


def test_trailing_undersized_chinese_section_merges_backward():
    tiny = "死結的定義如下。"
    text = f"# 死結\n\n## 資源配置圖\n\n{ZH_GRAPH}\n\n## 小結\n\n{tiny}"

    chunks = chunk_markdown(text, min_tokens=10, max_tokens=300)

    assert len(chunks) == 1
    assert chunks[0].locator == "死結 › 資源配置圖"
    assert tiny in chunks[0].text


def test_mixed_language_section_is_sized_by_both_scripts():
    # Three whitespace-separated words, so a whitespace counter reads this as
    # undersized and merges it away; counting CJK codepoints reads it as a
    # section in its own right.
    mixed = (
        "死結偵測演算法會定期檢查資源配置圖是否存在環路，"
        "deadlock detection 的成本與檢查頻率成正比。"
    )
    text = f"# 死結\n\n## 偵測\n\n{mixed}\n\n## 資源配置圖\n\n{ZH_GRAPH}"

    chunks = chunk_markdown(text, min_tokens=10, max_tokens=300)

    assert [c.locator for c in chunks] == ["死結 › 偵測", "死結 › 資源配置圖"]
    assert chunks[0].text == mixed


def test_heading_stack_resets_on_shallower_heading():
    text = (
        f"# BST\n\n## Insertion\n\n{NORMAL}\n\n"
        f"# AVL\n\n## Insertion\n\n{NORMAL}"
    )

    chunks = chunk_markdown(text, min_tokens=10, max_tokens=30)

    assert [c.locator for c in chunks] == ["BST › Insertion", "AVL › Insertion"]


# The windowed path (#7). Its rules are about a Document with nothing to split
# on, so each test below states one over Extents -- a PDF's pages are what
# production hands it (ingestion/extraction.py), but nothing here is about a
# PDF. No test asserts a token count for a string, for ADR-0004's reason: the
# measure behind the counts is a placeholder due for replacement in Week 6.

PAGE_ONE = "alpha " * 30
PAGE_TWO = "bravo " * 30


def windows(extents, window, overlap):
    return chunk_windows(extents, window=window, overlap=overlap)


def test_a_window_shorter_than_the_document_yields_several_chunks():
    chunks = windows([Extent(locator="p. 1", text="word " * 100)], window=30, overlap=5)

    assert len(chunks) > 1
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_a_document_shorter_than_one_window_is_a_single_chunk():
    chunks = windows([Extent(locator="p. 1", text=PAGE_ONE)], window=100, overlap=20)

    assert len(chunks) == 1
    # Verbatim, trailing whitespace included: the only window runs from the
    # start of the text to the end of it, so nothing is reflowed away.
    assert chunks[0].text == PAGE_ONE


def test_consecutive_windows_repeat_the_overlap():
    # What the overlap is for: a sentence sitting on the seam between two
    # windows is whole in one of them rather than halved into both.
    text = " ".join(f"w{i}" for i in range(60))

    chunks = windows([Extent(locator="p. 1", text=text)], window=20, overlap=5)

    first, second = chunks[0], chunks[1]
    assert first.text.split()[-5:] == second.text.split()[:5]


def test_no_window_exceeds_the_configured_size():
    chunks = windows([Extent(locator="p. 1", text="word " * 250)], window=40, overlap=10)

    assert all(count_tokens(c.text) <= 40 for c in chunks)


def test_every_word_of_the_document_appears_in_some_window():
    text = " ".join(f"w{i}" for i in range(97))  # not a multiple of the stride

    chunks = windows([Extent(locator="p. 1", text=text)], window=20, overlap=5)

    covered = {word for chunk in chunks for word in chunk.text.split()}
    assert covered == set(text.split())


def test_the_final_window_carries_more_than_the_overlap_it_repeats():
    # Otherwise the tail of a Document that does not divide evenly comes back
    # as a Chunk that is entirely a copy of the one before it -- a duplicate in
    # the Evidence, spending a retrieval slot to say what the previous Chunk
    # already said.
    text = " ".join(f"w{i}" for i in range(83))

    chunks = windows([Extent(locator="p. 1", text=text)], window=20, overlap=5)

    last, previous = chunks[-1], chunks[-2]
    assert not set(last.text.split()) <= set(previous.text.split())


def test_a_chunk_is_cited_by_the_extent_it_starts_in():
    chunks = windows(
        [Extent(locator="p. 1", text=PAGE_ONE), Extent(locator="p. 2", text=PAGE_TWO)],
        window=30,
        overlap=0,
    )

    assert [c.locator for c in chunks] == ["p. 1", "p. 2"]


def test_a_window_straddling_two_extents_is_cited_by_the_one_it_starts_in():
    # The window is fixed and the extent boundary is not a boundary it
    # respects, so a Chunk really does hold text from two pages. It is cited by
    # the page a reader following the citation starts reading at.
    chunks = windows(
        [Extent(locator="p. 1", text=PAGE_ONE), Extent(locator="p. 2", text=PAGE_TWO)],
        window=40,
        overlap=0,
    )

    assert chunks[0].locator == "p. 1"
    assert "alpha" in chunks[0].text and "bravo" in chunks[0].text


def test_no_chunk_has_an_empty_locator():
    chunks = windows(
        [Extent(locator=f"p. {n}", text="word " * 40) for n in range(1, 4)],
        window=25,
        overlap=5,
    )

    assert chunks
    assert all(c.locator for c in chunks)


def test_an_extent_with_no_text_is_never_cited():
    # A blank page in a scanned paper is a place in the Document, not a span of
    # it -- dropped exactly as a heading with nothing under it is dropped from
    # the structured path.
    chunks = windows(
        [
            Extent(locator="p. 1", text=PAGE_ONE),
            Extent(locator="p. 2", text="   \n  "),
            Extent(locator="p. 3", text=PAGE_TWO),
        ],
        window=30,
        overlap=0,
    )

    assert "p. 2" not in {c.locator for c in chunks}


def test_a_document_with_no_text_at_all_yields_no_chunks():
    assert windows([Extent(locator="p. 1", text="  \n ")], window=30, overlap=5) == []


def test_an_overlap_at_least_as_large_as_the_window_is_refused():
    # It is not a bad chunking, it is no chunking: the window never advances,
    # so the Document is cut into an unbounded number of copies of its opening.
    with pytest.raises(ValueError, match="overlap"):
        windows([Extent(locator="p. 1", text=PAGE_ONE)], window=20, overlap=20)


def test_a_window_of_no_tokens_is_refused():
    with pytest.raises(ValueError, match="window"):
        windows([Extent(locator="p. 1", text=PAGE_ONE)], window=0, overlap=0)


def test_an_extent_with_no_locator_is_refused():
    # The path's one promise to the reader, enforced where it is made rather
    # than trusted of each reader in turn. An unstructured Document has nothing
    # but its Locator to be cited by, so a Chunk without one renders as a
    # source card ending in a dash: unfollowable, and not visibly so.
    with pytest.raises(ValueError, match="locator"):
        windows([Extent(locator="", text=PAGE_ONE)], window=30, overlap=5)


def test_one_unnamed_extent_refuses_the_whole_document():
    # Not "chunk the named ones and drop the rest": a Document half of whose
    # Chunks are missing is a corpus gap nobody sees, whereas a run that fails
    # names the file in its report (#4).
    with pytest.raises(ValueError, match="locator"):
        windows(
            [Extent(locator="p. 1", text=PAGE_ONE), Extent(locator="", text=PAGE_TWO)],
            window=30,
            overlap=5,
        )


ZH_PAPER_PAGE = (
    "本文提出一套以資源配置圖為基礎的死結偵測方法，並在多核心環境下評估其成本。"
    "實驗顯示，當資源種類增加時，偵測的執行時間呈現次線性成長。"
)


def test_a_chinese_document_is_windowed_by_the_same_token_measure():
    # ADR-0004's obligation on this path, and the failure it names: a window
    # sized in whitespace-words reads a whole Chinese page as one or two words,
    # never reaches its own size, and returns the page as a single Chunk.
    page = ZH_PAPER_PAGE * 4

    chunks = windows([Extent(locator="p. 1", text=page)], window=40, overlap=8)

    assert len(chunks) > 1
    assert all(count_tokens(c.text) <= 40 for c in chunks)
    assert all(c.locator == "p. 1" for c in chunks)


def test_a_chinese_window_is_a_verbatim_span_of_its_document():
    # The oversized splitter's property, on this path: the source string is cut
    # between token offsets rather than rejoined from token strings, so a
    # Chunk's text is text the Document contains.
    page = ZH_PAPER_PAGE * 4

    chunks = windows([Extent(locator="p. 1", text=page)], window=40, overlap=8)

    assert all(chunk.text in page for chunk in chunks)
