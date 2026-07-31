from config import MAX_SECTION_TOKENS, MIN_SECTION_TOKENS
from core.chunking import chunk_markdown

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
