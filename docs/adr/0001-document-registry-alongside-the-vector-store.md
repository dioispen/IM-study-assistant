# Document registry alongside the vector store

ChromaDB holds Chunks, but a Document's identity lives in a separate small
registry keyed by a stable `doc_id`, recording its source path, title, domain,
source type, language, and a content hash. Chunk IDs are derived from
`doc_id` + ordinal rather than being random, so the same Document re-chunked
under different parameters produces comparable IDs.

We did this because the project's central activity is re-ingesting the *same*
corpus under varying chunk parameters and comparing the results. With chunks as
the only entity and random chunk IDs, two experiment runs share no identifiers,
a changed source file leaves undetectable stale chunks behind, and evaluation
labels cannot outlive a chunking change.

## Considered Options

Chunks as the only entity, with `source_path` as the de facto Document
identity and re-ingest implemented as delete-where-source_path. Rejected:
simpler and needs no second store, but makes experiment runs undiffable and
stale-chunk detection manual.

## Consequences

Evaluation labels name Gold Documents rather than gold chunks — see
[0002](./0002-evaluate-retrieval-not-generated-answers.md). That is not a
convenience; it is the property that lets a labelled eval set survive the
chunk-size experiments it exists to serve.
