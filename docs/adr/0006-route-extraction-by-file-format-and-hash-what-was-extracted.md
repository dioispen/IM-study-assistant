# Route extraction by file format, and hash what was extracted

Ingestion picks a reader by the file's format alone — `.md` and `.docx` today —
and never by a flag on the invocation or by the Document's Source type. Every
reader produces the same two things: a list of Sections (a heading path and the
prose beneath it) and the text that Document's `content_hash` is taken over.
The chunking rules are stated once, over Sections, and no reader has an opinion
about them.

The text a reader hands back for hashing is per-format on purpose. Markdown's
is the file's own text. A docx's is the rendering of what was extracted from it.

We did this because a student's notes are not one file format, and which format
a note happens to be saved in says nothing about what is in it. A run has to
walk the folder the notes are actually in — `.md` beside `.docx` — and a note
is a note whichever program wrote it. The alternative, a run per format, makes
the corpus owner sort their own notes by extension before ingesting them, and
gets the retirement prefix wrong the moment they forget: a `.md`-only run over
a mixed folder does not find the `.docx` notes beneath it, which under
[0005](./0005-retire-a-document-absent-from-a-walk-that-covers-it.md) is exactly
the evidence that retires them.

Routing on Source type instead was the other candidate and is worse than merely
inconvenient. Source type answers "whose words am I reading?" (`CONTEXT.md`) —
a note the student wrote and a textbook chapter are different Source types
whether or not they are the same format, and a note stays a note whether it was
saved as Markdown or as Word. Folding the two together would make
`--source-type note` a claim about a file extension, and would leave no way to
say "this docx is a textbook chapter".

Hashing is split because the two formats' bytes mean different things. A `.md`
file's text *is* the note, and it is already what every Markdown Document in a
registry was hashed as — changing that would re-embed the whole corpus on the
next run for an edit nobody made. A `.docx` file's bytes are a zip: opening the
note in Word and closing it rewrites timestamps and can reorder parts, so a
hash over the package would report a changed Document every time its owner read
one of their own notes. Hashing what was extracted asks the question the skip
in [0001](./0001-document-registry-alongside-the-vector-store.md) actually wants
answered — have the note's words changed? — rather than a question about the
file.

## Considered Options

**Convert docx to Markdown text and reuse the Markdown parser.** The smallest
change, and rejected on the Locators it produces. A docx marks a heading with a
paragraph style, and body text with the absence of one; rendering both to a
string and parsing the string back throws that distinction away and re-derives
it from syntax. A study note whose body line reads `#include <stdio.h>` then
becomes a heading, and every Chunk under it is cited by a Locator the note does
not contain. Escaping the body on the way out would fix the parse and put a
backslash into the text a reader is shown. Sections skip the round trip.

**A separate chunking path per format.** Rejected: the rules — never straddle a
heading, merge undersized forward, split oversized under one Locator — are
about headed prose, not about syntax, and PLAN.md §4 has exactly one structured
path. Two copies drift, and the drift shows up as a Word note chunking
differently from the same note saved as Markdown, which is a difference nobody
can see from the answer.

**Hash the file's bytes for every format.** Rejected for docx for the reason
above. Uniformity here would cost a full re-embed of every Word note on every
run in which its owner opened it, and the run would report it as ordinary
re-ingestion.

**Hash the extracted text for every format, Markdown included.** Rejected only
because of what it costs today: it would change the `content_hash` of every
Markdown Document already in a registry and re-embed the whole corpus once, for
no behavioural gain — a Markdown file's text is already exactly what would be
hashed.

## Consequences

A docx's `content_hash` moves only when the extracted text moves, so a change
this reader cannot see is a change ingestion cannot see. Bold, colour, a
footnote, an image: all invisible to the hash and all absent from the Chunk
text, which is consistent but means widening the reader later — to pick up, say,
a footnote's prose — silently re-ingests every Word note in the corpus. That is
the right outcome and worth expecting rather than being surprised by.

The rendering that is hashed writes each Section under its whole Locator, not
under the single heading that opened it. An outer heading with no prose directly
beneath it contributes no Section of its own, so rendering only the innermost
heading would leave that outer heading out of the hash — and renaming it would
move every Locator in the note while `content_hash` stayed where it was, which
is a stale generation no later run could detect. The rendering is never parsed
back; it exists to be hashed.

Word's heading styles are read strictly: `Heading 1` through `Heading 9`, by
style name or style id, and nothing else. `Title`, `Subtitle` and a user's own
styles are body text. A note whose structure lives entirely in bold paragraphs
therefore chunks as one long Section rather than badly — visible immediately as
a single Locator covering the whole note, rather than as heading paths quietly
invented from formatting.

A Word heading below level 6 nests at level 6, because Markdown's `#` runs out
there and a Locator has to mean the same thing whichever format it came from.

The walk now matches suffixes case-insensitively, which `rglob` did not do
uniformly: `rglob("*.md")` case-folds on Windows and does not on POSIX, so a
note saved as `NOTES.DOCX` would otherwise be a Document that exists in one
machine's corpus and not another's — and, under 0005, one that a run on the
other machine could retire.

`~$`-prefixed files are skipped rather than failed. Word leaves one beside every
note it has open; it carries the note's name and the `.docx` suffix and is not a
Word package, so reading it would put a warning in every run for as long as the
note is open. A per-file warning the corpus owner learns to ignore is worse than
no warning.

Formats with no headings to split on are still out. PDF needs the windowed path,
deferred with the other unstructured Sources (PLAN.md §五) — the split is by
chunking path, not by the word "note". When it arrives it adds a reader here and
a second path beside `chunk_sections`, not a change to either.

> **Superseded in part by
> [0007](./0007-window-across-the-whole-document-and-cite-where-a-chunk-starts.md).**
> PDF has landed, and the paragraph above held: it added `_extract_pdf` here and
> `chunk_windows` beside `chunk_sections`, and changed neither. What 0007 does
> change is the shape a reader may hand back — Sections or Extents, rather than
> Sections alone — because the structure a format could recover is what picks
> the chunking path.
