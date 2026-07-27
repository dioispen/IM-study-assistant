# Two-layer abstention with a derived threshold

Abstention has two layers that look redundant and are not. A distance gate
abstains before the LLM is called when the nearest Chunk is farther than τ; the
generation prompt independently instructs the model to abstain when the
Evidence does not support an answer. τ is never hand-tuned — it is swept
against the eval set and read off by a stated rule: the largest τ whose false
abstention rate on answerable questions stays at or below 5%.

Keep both layers. They catch different failures. The gate handles questions
whose subject the corpus never covers, where the nearest Chunks are far away.
The prompt backstop handles near-miss questions — a specific fact missing from a
subject the corpus does cover — where the retrieved Chunks are topically
adjacent and therefore close, so no τ tight enough to gate them is loose enough
to keep genuine questions working. Deleting either layer leaves one of those two
failure modes uncovered.

## Consequences

The gate is deliberately set conservatively, and the rule that picks τ
prioritises not refusing answerable questions over catching hopeless ones. The
two layers fail with different recoverability: when the gate abstains wrongly
the LLM is never called and the error is final, whereas when the gate passes
weak Evidence the backstop can still abstain. Asymmetric error costs argue for
a permissive cheap layer and a smart second one, not for tightening both.

τ is a property of the embedding model, not of the system. It is re-derived
whenever the embedding model changes; a τ carried across models would silently
confound a model comparison with a mis-set threshold.

The vector collection must be created with cosine distance
(`hnsw:space: "cosine"`) — ChromaDB defaults to L2, and switching afterwards
means recreating the collection and re-embedding the corpus. Distances are
returned lower-is-closer, so the gate reads `distance > τ`.
