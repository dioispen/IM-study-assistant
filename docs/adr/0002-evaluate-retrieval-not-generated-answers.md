# Evaluate retrieval, not generated answers

The routine evaluation loop scores retrieval — Recall@k and MRR against Gold
Documents, plus abstention rates on Trap questions — and does not grade the
generated answer at all. Generated answers are read by hand only at weekly
milestones.

Every tuning parameter in this project (chunk window, overlap, section split and
merge thresholds, top-k, embedding model) is a retrieval-stage parameter.
Grading generated text to measure them reads the signal through a noisy
amplifier: phrasing varies run to run, so a real effect needs many samples to
emerge that retrieval metrics show directly and deterministically. It is also
the only version that can actually be re-run on every change — grading three
chunk sizes against three top-k values across the question set by hand is a
multi-hour round, so it would simply stop happening.

## Consequences

Abstention is reported as three separate rates that are never summed: false
abstention on answerable questions, and abstention on out-of-corpus and
near-miss Traps respectively. A single aggregate hides the case that matters —
a system can abstain perfectly on unrelated subjects while confidently
inventing answers to near-miss questions.

Every result row is keyed by embedding model, because the embedding model is a
swept parameter rather than a fixed platform choice, and scores from different
models are not comparable.

The eval set is deliberately weighted toward Traps (roughly 20 answerable to 30
Traps) rather than split evenly across the four question categories. Answerable
questions cost a Gold-Document labelling pass; Traps cost one sentence each. At
an even split there would be too few Traps for their rate to carry a usable
signal.

Nothing here measures conflict detection. That feature is demonstrated rather
than validated, and the README says so.
