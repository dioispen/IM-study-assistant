# Window across the whole Document, and cite where a Chunk starts

A Document whose format offers no headings is chunked by fixed windows with
overlap, cut over the Document's whole text rather than page by page. Each
Chunk is cited by the Extent its text starts in — a PDF's page number — so a
window that runs across a page break is cited by the page a reader turns to
first. The window and the overlap are `WINDOW_TOKENS` and
`WINDOW_OVERLAP_TOKENS` in `config.py`, counted with `core/tokenization.py`,
and due for the Week 6 sweep with the section thresholds beside them.

Which path a Document takes is decided by what its reader could recover, in one
place: `ingestion/common.py` matches on a `StructuredDocument` (Sections, from
`.md` and `.docx`) or an `UnstructuredDocument` (Extents, from `.pdf`). A
format decides which structure it can offer and the structure decides the path,
which is [0006](./0006-route-extraction-by-file-format-and-hash-what-was-extracted.md)'s
routing carried one step further rather than a second routing rule beside it.

Heading detection inside a PDF is not attempted at all. Papers arrive
two-column, and both of the things a reader might do about that are worse than
doing nothing: sorting text blocks by position interleaves the two columns line
by line, and inferring a heading from font size invents a Locator citing a
place in the paper that is not there. Pages are what a PDF can be trusted to
say, so pages are what it cites. Extraction quality on a real paper is checked
by hand before the paper is trusted (PLAN.md §5.3), which is the check this
decision leans on instead.

## Considered Options

**Windows bounded by the page, never straddling one.** The obvious mirror of
"a Chunk never straddles a heading", and rejected because the two boundaries
are not alike. A heading is the author saying "the subject changes here", so
respecting it is respecting the Document. A page break is where the printer ran
out of paper — it falls mid-sentence as often as not — so respecting it cuts a
paragraph the author wrote as one and forbids the overlap from repairing the
seam, which is the one place a fixed window most needs repairing. It also makes
the window not fixed: a page's tail becomes a short Chunk whose embedding is
dominated by whatever few sentences happened to fall there, and a page holding
a figure caption becomes a Chunk of one line.

**A page-range Locator for a straddling Chunk — "pp. 4–5".** Rejected on what
it costs to say. `chunk_windows` would have to know that its Extents are pages
and that pages have consecutive numbers, and the next unstructured Source has
neither — a fetched article has one body and an anchor, and there is no range
between two anchors. The range would then have to be a per-format callback
threaded through the chunker, which is a lot of machinery for a Locator that
tells the reader nothing the starting page does not: they turn to that page and
read forward, and the Chunk's text runs on from there.

**Chunk on paragraph or sentence boundaries instead of a fixed window.** The
better chunking, and not the one that was deferred. PLAN.md §5.1 names "固定長
度 + 重疊" specifically and §5.3 puts `window` / `overlap` on the Week 6 sweep
list, so a path with no such parameters would arrive with nothing for that
sweep to tune. It is also not free: a two-column extraction produces paragraph
breaks that are an artifact of the column, so the boundaries would be as
invented as the headings above.

**A separate `scripts/ingest_papers.py`, as PLAN.md §5.1 sketches.** Rejected
under 0006: a run per format makes the corpus owner sort their own material by
extension before ingesting it, and gets the retirement prefix wrong the moment
they forget. `python cli.py ingest <folder> --domain os --source-type paper` is
the whole of it, and a folder holding papers beside notes ingests in one walk.

## Consequences

A Chunk can hold text from two pages while citing one. That is the honest
report of what the Chunk is — it starts there — but it means a reader who
follows a citation and searches only that page may not find the sentence on it.
The alternative was a range Locator, priced above; what makes this acceptable
is that the reader is reading forward from the page anyway.

A page's number is its position in the file, and only the reader knows that
position — `_pdf_extents` numbers with `enumerate(document, start=1)`. So a
reader that skipped a blank page would renumber every page after it and cite
the wrong one for the rest of the paper. That is why `ingestion/extraction.py`
yields an Extent for every page including the empty ones, and why
`chunk_windows` is the one that drops them: by then the number is already
written into the Locator, so dropping costs no numbering at all.

The Document's `content_hash` is taken over the extracted pages rendered under
their own Locators, so the same words repaginated is a changed Document. It has
to be: every Locator after the break has moved, and a hash blind to that leaves
the corpus citing page numbers the file no longer has, with no later run able
to detect it ([0001](./0001-document-registry-alongside-the-vector-store.md)).

`WINDOW_TOKENS` matches `MAX_SECTION_TOKENS` deliberately. Chunks from both
paths compete in one ranking, and a systematically longer Chunk from one of
them would dilute its embedding across more subjects while a shorter one won
every short match — a retrieval bias with no meaning behind it, and one that
would read as "the corpus prefers papers". Whoever sweeps these in Week 6
inherits that pairing as something to check rather than something to preserve.

A scanned PDF and a password-protected one are named as their own failures
rather than arriving as a Document that yielded no Chunks. The three ask
opposite things of the corpus owner — one wants OCR, one wants a password, one
wants a file that is not empty — and a single message covering all three sends
them looking in the wrong place two times out of three.

`pymupdf` joins the dependencies, as PLAN.md §3 picked. It is the one library
here whose failures are not a documented exception type but whatever its C
layer raises, so `_pdf_extents` catches every exception for the reason
`_docx_sections` does: guessing the list short does not report one bad file, it
aborts the walk and takes the retirement sweep with it.
