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


def test_heading_stack_resets_on_shallower_heading():
    text = (
        f"# BST\n\n## Insertion\n\n{NORMAL}\n\n"
        f"# AVL\n\n## Insertion\n\n{NORMAL}"
    )

    chunks = chunk_markdown(text, min_tokens=10, max_tokens=30)

    assert [c.locator for c in chunks] == ["BST › Insertion", "AVL › Insertion"]
