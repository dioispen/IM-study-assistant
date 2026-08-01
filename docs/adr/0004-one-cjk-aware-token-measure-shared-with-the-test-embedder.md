# One CJK-aware token measure, shared with the test embedder

Section size for structured chunking is measured by a heuristic that counts one
CJK codepoint as one token and a run of non-CJK word characters as one token.
It lives in `core/tokenization.py`, which is the only place in the codebase
that decides what a token is, and it exposes each token's span in the source
text as well as its string. Both production chunking and the offline
`FakeEmbedder` count with it — chunking measures the length of the token list,
the test double hashes the token strings.

Whitespace-split words, the measure this replaces, cannot work for the corpus
this system exists to serve. Chinese does not delimit words with whitespace, so
a whole Chinese section counted as one or two tokens, read as undersized, and
merged into its neighbour — a three-section note collapsed into a single Chunk
carrying only the last section's Locator.

Adopting `tiktoken` instead was considered and rejected. It shifts English
counts as well as Chinese ones, which would force `MIN_SECTION_TOKENS` and
`MAX_SECTION_TOKENS` to be re-picked, and PLAN.md defers picking a real
tokenizer to Week 5, where it belongs beside the threshold sweep that choice
invalidates. The heuristic keeps English counts essentially unchanged and puts
Chinese in the right order of magnitude — `text-embedding-3-small` splits
Chinese into roughly one to two tokens per character — so one pair of numbers
keeps meaning roughly the same thing in both languages.

The windowed chunking path for unstructured Documents does not exist yet. When
it lands, its `window` and `overlap` are counted with this same function rather
than a second measure of its own: a window sized in whitespace-words would
swallow a whole Chinese page exactly as section thresholds swallowed a whole
Chinese section, and two measures would leave Week 5 sweeping two sets of
numbers against two definitions of a token.

## Consequences

This is a placeholder, and stays one until Week 5 replaces it. Whoever does
inherits every call site of `core/tokenization.py` and the obligation to
re-sweep the thresholds behind them, because a new measure moves the counts
those numbers were picked against. Tests therefore pin chunking behaviour — how many Chunks a note became, which Locator each
carries — and never assert a token count for a string, so that replacing the
measure does not mean rewriting the suite.

Production code and a test double sharing one function is deliberate coupling,
not an oversight to clean up. A Chinese seam test is only meaningful if the
fixture's Chunks and the test's question are tokenized alike; two tokenizers
would drift, and the drift would show up as retrieval geometry that proves
nothing.

Because tokens carry spans, the oversize splitter cuts the source string
between offsets rather than rejoining token strings. A Chunk's text is now a
verbatim span of its Document: long Chinese sections became splittable at all,
and Chunk text in both languages stopped being reflowed with its line breaks
removed.

Counts move very slightly for English where punctuation splits a word —
`co-operate` is two tokens, not one. That is within the noise of a placeholder
measure and changes no existing behaviour.
