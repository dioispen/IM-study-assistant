# The prompt backstop declares itself with a sentinel

The generation instructions carry a contract: when the Evidence does not
support an answer, the model replies with one exact sentinel line and nothing
else. The generation module both asks for that line and recognises it, and the
ask seam maps a recognised line onto the prompt-backstop Abstention before the
answer leaves the pipeline. The sentinel is the only thing read out of
generated text — no keyword sniffing, no classifier, no "does this sound like
a refusal" heuristic.

Abstention layer 2 of ADR-0003 previously emitted a paragraph of prose, which
meant no code could tell it from an ordinary answer. Every surface therefore
displayed the system's most important behaviour as though it had not happened:
a model that had correctly declined was rendered with an answer's confidence.
A layer nobody can observe is a layer nobody can tell is broken, which is
precisely what ADR-0003 keeps two layers in order to avoid.

Recognising an Abstention in the model's own words, instead of asking it to
declare one, was the obvious alternative and is rejected. It makes the pipeline's behaviour a function of
the shape of model prose, so a rephrasing between model versions silently
changes which questions abstain — and it puts prose quality inside the test
suite's assertions, which is exactly what ADR-0002 keeps out. Against a
declared signal the suite asserts on our handling of a contract we wrote.

## Consequences

**A model that ignores the sentinel degrades to today's behaviour, not to
something worse.** Prose saying "I don't know" without the sentinel is an
ordinary answer, displayed with its Evidence cards, exactly as before this
decision. This is a known limitation, accepted rather than defended against:
the defence would be the heuristic above, and it would cost more than the
failure does. The failure is visible — a reader sees an answer that admits it
has nothing, beside the Chunks it had — whereas a misfiring heuristic silently
converts answers into Abstentions.

A sentinel with prose beside it is an answer, and its text reaches the reader
with the token still in it. That is the same limitation seen from the other
side: the contract is the line alone, so a half-honoured contract is not one,
and stripping the token out of prose would be editing a generated answer on a
guess about what the model meant by it. Rare enough to accept, and visible when
it happens.

Layer 2's abstention rate, when the Week 6 eval measures it, counts declared
abstentions only and therefore undercounts. The number is a floor, and reading
it as one is the reason it can be reported at all: before the sentinel there
was no automated measurement of layer 2 to under- or over-count.

The prompt and the reading of the reply live in one module, because a contract
split across two is one that can drift apart while both halves still pass their
own tests. Changing the sentinel string changes the prompt in the same edit.

The sentinel never reaches a reader. What leaves the pipeline is the Abstention
in words, and those words are the backstop's own rather than the distance
gate's: the gate found nothing close enough to answer from, while the backstop
was given Evidence, handed it to the model and had it judged insufficient. That
Evidence stays attached to the Abstention — it is exactly the material the
model judged, and a reader needs it in order to disagree — which is what makes
the two layers render differently on every surface: a gate Abstention cites
nothing, a backstop Abstention keeps its cards.
