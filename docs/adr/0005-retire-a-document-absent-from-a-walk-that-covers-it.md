# Retire a Document absent from a walk that covers it

A Document is retired — its Chunks deleted from the store and its registry row
deleted after them — when an ingestion run walks a folder whose path below the
corpus root is a prefix of that Document's `source_path`, and the walk does not
find it. Nothing outside the walked prefix is retired, however much of the
corpus the run happened not to look at. Every retirement is named by
`source_path` in the `IngestReport`.

We did this because `doc_id` is derived from a Document's path below the corpus
root, which makes a moved note a *new* Document while the old one lingers. Both
generations stay retrievable, so one question draws Evidence from two copies of
the same prose, and the older copy cites a `source_path` that no longer resolves
to a file — the traceability failure the corpus root was introduced to close,
reached by a different route. [0001](./0001-document-registry-alongside-the-vector-store.md)
already forbids two generations of Chunks under one `doc_id`; this extends the
same rule across the identity change a move causes, where the stale generation
hides under a `doc_id` no longer derivable from anything on disk.

Chunks are deleted before the registry row, so a run interrupted between the
two leaves a Document the next run retires again. The other order would leave
Chunks that nothing in the registry names, and retirement finds its candidates
by reading the registry — so no later run could reach them.

The prefix is what makes absence into evidence. Within the walked folder the
walk is exhaustive, so a Document it did not reach is genuinely not there —
deleted, moved elsewhere, or moved into a dot-prefixed directory, which is what
deleting a note in a vault does. Outside it the run has seen nothing and so
asserts nothing.

## Considered Options

**Retire every Document the run did not see.** Rejected: a run pointed at one
subfolder sees almost nothing of the corpus, so the first such run would empty
it. The failure is total, silent and indistinguishable from a successful
ingest.

**Retire a Document whose `source_path` no longer resolves to a file.**
Rejected because it stats the whole registry on every run to answer a question
the walk has already answered for the part of the corpus the run actually
covered — and it asks that question of Documents that have no file to stat, so
it needs the exemption described under Consequences before it can be correct
at all, where absence from a walk needs it only once a second kind of Source
exists.

**Mark retired Documents rather than deleting them.** Rejected for now:
retrieval filters nothing on such a mark today, so a marked Document is a live
one with an extra column, and the failure this ADR closes stays open. The
option remains open if an undo is ever wanted; deletion is what the report
names loudly precisely because it cannot be undone by re-running.

## Consequences

A note moved *out of* the walked folder — `os/` to `dsa/`, when the run was
pointed at `os/` — is ingested at its new path while the old Document survives.
It is retired by the next run that covers both, which pointing ingestion at the
corpus root always does. Retirement is therefore a property of what a run
looked at, and the whole-corpus run is the one that leaves nothing behind.

Every Document in the corpus today comes from a file below the corpus root, and
the sweep assumes it. A Source identified by something other than such a path —
config.py names the case, a Wikipedia Document keyed on its article URL — has a
`source_path` that no walk can ever reach, and at the corpus root the covering
prefix is empty, so a root run would retire it on its first pass. Whichever
ticket adds the second kind of Source owes retirement a way to tell "beneath the
walked folder" from "not a path below the root at all"; until then no such
Document exists to be lost.

A run pointed at a folder that is not there is refused rather than walked
(`FolderNotFound`). An empty walk of a missing folder is the same evidence as a
folder whose notes are all gone, and taking it at face value would retire
everything beneath it — the total, silent failure the first rejected option was
rejected for, reached by accident instead of by design.

A Document whose text could not be extracted counts as found, not as gone: it
is recorded in the report as failed and keeps whatever generation of Chunks it
last had. Retiring on a failed extraction would let one locked or garbled file
delete the only copy of a note the corpus holds — the same reason ingestion
already declines to clear a failed Document's Chunks.
