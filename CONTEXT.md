# IM Study Assistant

A retrieval-augmented question-answering system over a personal Information
Management study corpus. Answers cite where they came from, and say so when the
corpus does not contain an answer.

## Language

**Document**:
One source text that entered the corpus as a unit — a single note, one
Wikipedia article, one web article, one paper. Has an identity that outlives
any particular chunking of it.
_Avoid_: file, source, entry

**Chunk**:
A contiguous span of one Document's text, sized to be embedded and retrieved on
its own. Belongs to exactly one Document, knows its position within it, and —
where the Document has structure — does not straddle a heading.
_Avoid_: passage, segment, fragment

**Locator**:
The human-readable pointer to where in its Document a Chunk came from, written
for a reader who wants to go look. Always present; its shape depends on what
the Document offers — a heading path, a page, an anchor.
_Avoid_: section, position, offset, citation

**Extent**:
The stretch of a Document that one Locator names, in a Document whose format
offers somewhere to cite but nowhere to split — a page, an article's body. What
a Chunk is cut _from_ and cited _by_ on the windowed path, and never a Chunk
itself: several Chunks can start in one Extent, and one Chunk can run across
two.
_Avoid_: passage, page, block, unit

**Evidence**:
The Chunks retrieved for one question and passed to generation, after any
diversity limits are applied. Everything the answer may rest on, and the only
thing its citations may point at.
_Avoid_: context, passages, results, hits

**Retirement**:
A Document leaving the corpus, because an ingestion run looked where it was and
did not find it — deleted, moved, or moved out of reach. Takes its Chunks with
it, and is named in the run's report rather than done silently.
_Avoid_: removal, purge, cleanup, tombstone

## Evaluation

**Gold Document**:
A Document known to contain the answer to a particular eval question. Named at
Document granularity so that labels survive re-chunking.
_Avoid_: ground truth, relevant document, expected source

**Trap question**:
An eval question deliberately chosen to have no Gold Document, asked to find
out whether the system admits it doesn't know. Two kinds, always scored apart:
an _out-of-corpus_ trap asks about a subject the corpus never covers; a
_near-miss_ trap asks for a specific fact the corpus omits from a subject it
does cover.
_Avoid_: negative example, adversarial question

**Abstention**:
The system declining to answer because the corpus does not support one. A
correct outcome on a Trap question and a failure on any other — both rates are
tracked, because either one alone can be gamed.
_Avoid_: refusal, fallback, no-answer

**Domain**:
The subject area a Document belongs to — the unit by which the corpus grows and
by which retrieval is narrowed. Every Document has exactly one.
_Avoid_: topic, subject, category, field

**Source type**:
Who authored a Document — the reader's answer to "whose words am I reading?".
An open set that grows as new kinds of material arrive; it says nothing about
how the text was extracted, nor whose claim wins in a disagreement.
_Avoid_: format, file type, origin
