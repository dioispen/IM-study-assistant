"""The one place that decides what a token is.

A placeholder measure, not a real tokenizer: one CJK codepoint is one token,
and a run of non-CJK word characters is one token. Picking a real tokenizer is
deferred to Week 6 tuning (PLAN.md), together with the threshold sweep that
choice would invalidate.

The heuristic is chosen so that the section thresholds in config.py keep
meaning roughly what they mean for English -- `text-embedding-3-small` splits
Chinese into about one to two tokens per character -- while giving Chinese a
count that is not simply one. Whitespace-delimited counting is not an option:
Chinese does not delimit words with whitespace, so a whole paragraph counts as
one word and every section reads as undersized.

Tokens carry their span in the source text so that a caller wanting to cut the
text can cut the original string rather than rejoin token strings, which would
destroy whatever sits between the tokens.
"""

import re
from dataclasses import dataclass

# Scripts written without whitespace between words: CJK ideographs (unified,
# extensions A and B, and the compatibility block) and Japanese kana.
# Everything else falls to the word-run branch.
_CJK = (
    "぀-ヿ"  # hiragana and katakana
    "㐀-䶿"  # CJK unified ideographs extension A
    "一-鿿"  # CJK unified ideographs
    "豈-﫿"  # CJK compatibility ideographs
    "\U00020000-\U0002a6df"  # CJK unified ideographs extension B
)

# Order matters: a CJK codepoint is tried first, so a word run stops at one --
# `\w` matches ideographs too, so an unguarded run would swallow them.
# Punctuation bounds a word run, as it already does for the offline test
# embedder, so `co-operate` is two tokens.
_TOKEN_RE = re.compile(rf"[{_CJK}]|(?:(?![{_CJK}])[\w'])+")


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


def tokenize(text: str) -> list[Token]:
    return [
        Token(text=match.group(), start=match.start(), end=match.end())
        for match in _TOKEN_RE.finditer(text)
    ]


def count_tokens(text: str) -> int:
    return sum(1 for _ in _TOKEN_RE.finditer(text))
