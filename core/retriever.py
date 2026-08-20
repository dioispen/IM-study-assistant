"""Retrieval: embed the question, query the Chunk store, return Evidence.

Evidence is "the Chunks retrieved for one question and passed to generation,
after any diversity limits are applied" (CONTEXT.md), so the per-Document cap
belongs here rather than downstream of here: what this returns is already
Evidence, and nothing between it and generation may drop a Chunk from it.
"""

import math
from collections import Counter

from core.embedder import Embedder
from core.store import ChunkFilter, RetrievedChunk, VectorStore


def retrieve(
    question: str,
    embedder: Embedder,
    store: VectorStore,
    top_k: int,
    chunk_filter: ChunkFilter | None = None,
    max_chunks_per_document: int | None = None,
) -> list[RetrievedChunk]:
    """Up to `top_k` Chunks for `question`, inside `chunk_filter` and capped.

    `max_chunks_per_document=None` is no cap, which is what the store returns on
    its own -- kept expressible because it is the baseline the cap is measured
    against, both in the seam tests and in the Week 7 experiment that has to
    show the cap did not cost Recall@k (PLAN.md §第 7 週).

    Capping the `top_k` nearest Chunks would shrink the Evidence instead of
    diversifying it: five slots capped at two per Document leaves two Chunks
    when the five nearest all came from one Document. The slots a crowding
    Document gives up have to go to Documents further down the ranking, so the
    store is asked for a larger pool of candidates, which the cap then thins
    back to `top_k`.

    That pool cannot be a fixed multiple of `top_k`, because a single Document
    with more Chunks than the pool holds fills it entirely -- the very case the
    cap exists for, arriving as a silently short answer. So it grows until
    either `top_k` Chunks survive the cap or the store returns fewer candidates
    than were asked for, which is the only honest evidence that there is
    nothing further down to reach for. Evidence still comes back short of
    `top_k` when the corpus genuinely holds too few distinct Documents near the
    question; that is the cap doing its job rather than failing at it.

    Growing costs repeated queries against a local HNSW index and nothing else:
    the question is embedded once, and the surplus candidates are discarded
    before the prompt is built, so no extra embedding call and no extra
    generation tokens.
    """
    if max_chunks_per_document is not None and max_chunks_per_document < 1:
        # A cap of zero admits no Chunk of any Document, so every question
        # abstains on empty Evidence and the system reports that the corpus
        # covers nothing. Refused rather than served, because that abstention
        # is indistinguishable from the real one (ADR-0003).
        raise ValueError(
            f"max_chunks_per_document={max_chunks_per_document} admits no Chunk "
            "into the Evidence, so every question would abstain as though the "
            "corpus were empty. Pass at least 1, or None for no cap."
        )

    [query_embedding] = embedder.embed([question])
    pool_size = _initial_pool_size(top_k, max_chunks_per_document)

    while True:
        candidates = store.query(
            embedding=query_embedding, top_k=pool_size, chunk_filter=chunk_filter
        )
        evidence = _cap_per_document(candidates, top_k, max_chunks_per_document)
        if len(evidence) == top_k or len(candidates) < pool_size:
            return evidence
        pool_size *= 2


def _initial_pool_size(top_k: int, max_chunks_per_document: int | None) -> int:
    """How many candidates to ask for on the first try.

    Filling `top_k` under a cap of `c` needs Chunks from at least ceil(top_k/c)
    distinct Documents, so the first pool gives each of those Documents a full
    `top_k`-sized stretch of the ranking to be found in -- enough that the usual
    case is answered in one query and the growth loop never runs a second.

    An uncapped question, or one whose cap is loose enough never to bind, asks
    for exactly `top_k`, exactly as retrieval did before there was a cap.
    """
    if max_chunks_per_document is None or max_chunks_per_document >= top_k:
        return top_k
    return top_k * math.ceil(top_k / max_chunks_per_document)


def _cap_per_document(
    candidates: list[RetrievedChunk], top_k: int, max_chunks_per_document: int | None
) -> list[RetrievedChunk]:
    """The nearest `top_k` of `candidates`, no more than the cap per Document.

    Walks nearest-first and keeps what is still under the cap, so each Document
    contributes its own nearest Chunks and the nearest Chunk overall is always
    kept -- which is what leaves the distance gate reading the same number
    whether or not a cap was applied (ADR-0003: the gate is the abstention rule,
    and a diversity limit must not quietly become a second one).
    """
    kept: list[RetrievedChunk] = []
    per_document: Counter[str] = Counter()

    for chunk in candidates:
        if len(kept) == top_k:
            break
        if (
            max_chunks_per_document is not None
            and per_document[chunk.doc_id] >= max_chunks_per_document
        ):
            continue
        kept.append(chunk)
        per_document[chunk.doc_id] += 1

    return kept
