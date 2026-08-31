from core.generator import (
    BACKSTOP_SENTINEL,
    FakeGenerator,
    build_prompt,
    declares_abstention,
)
from core.store import RetrievedChunk

EVIDENCE = [
    RetrievedChunk(
        chunk_id="doc1:000",
        doc_id="doc1",
        locator="BST › Insertion",
        domain="dsa",
        source_type="note",
        title="Red-Black Trees",
        text="Insert by walking left or right from the root.",
        distance=0.1,
    ),
    RetrievedChunk(
        chunk_id="doc2:003",
        doc_id="doc2",
        locator="Process Scheduling › Round Robin",
        domain="os",
        source_type="note",
        title="Scheduling Notes",
        text="Round robin gives each process a fixed time slice.",
        distance=0.3,
    ),
]


def test_prompt_instructs_answer_only_from_evidence():
    prompt = build_prompt("What is round robin?", EVIDENCE)

    assert "only" in prompt.lower()
    assert "evidence" in prompt.lower()


def test_prompt_instructs_citation():
    prompt = build_prompt("What is round robin?", EVIDENCE)

    assert "cite" in prompt.lower()


def test_prompt_declares_the_sentinel_the_backstop_abstains_with():
    # The contract the model is held to, and the only thing read back out of
    # its reply. A prompt that stopped naming the sentinel would leave
    # `declares_abstention` waiting for a line nothing was ever asked for.
    prompt = build_prompt("What is round robin?", EVIDENCE)

    assert BACKSTOP_SENTINEL in prompt


def test_prompt_requires_the_sentinel_alone_rather_than_alongside_prose():
    # "Exactly this line and nothing else" is what makes the signal parseable:
    # a sentinel wrapped in an apology is not the declared signal, and
    # `declares_abstention` is right to read it as an ordinary answer.
    prompt = build_prompt("What is round robin?", EVIDENCE)

    assert "nothing else" in prompt.lower()


def test_prompt_includes_the_question():
    prompt = build_prompt("What is round robin?", EVIDENCE)

    assert "What is round robin?" in prompt


def test_prompt_attributes_each_chunk_with_title_source_type_and_locator():
    prompt = build_prompt("What is round robin?", EVIDENCE)

    assert "Red-Black Trees" in prompt
    assert "BST › Insertion" in prompt
    assert "note" in prompt
    assert "Scheduling Notes" in prompt
    assert "Process Scheduling › Round Robin" in prompt


def test_prompt_with_no_evidence_still_carries_the_abstention_instruction():
    prompt = build_prompt("What is round robin?", [])

    assert BACKSTOP_SENTINEL in prompt


def test_fake_generator_never_makes_network_calls_and_is_deterministic():
    generator = FakeGenerator()

    first = generator.generate("some prompt")
    second = generator.generate("some prompt")

    assert first == second
    assert isinstance(first, str)


def test_the_sentinel_alone_declares_an_abstention():
    assert declares_abstention(BACKSTOP_SENTINEL)


def test_surrounding_whitespace_does_not_stop_the_sentinel_declaring():
    # Whitespace is not content: a trailing newline is how a chat completion
    # ends a line, not the model saying something besides the sentinel.
    assert declares_abstention(f"\n  {BACKSTOP_SENTINEL}  \n")


def test_the_sentinel_with_anything_beside_it_is_an_ordinary_answer():
    # The contract is the line alone. A reply that argues its case alongside
    # the sentinel is prose, and reading a declaration out of it would be the
    # heuristic this design exists to avoid (ADR-0008).
    assert not declares_abstention(f"{BACKSTOP_SENTINEL} — the Evidence covers only insertion.")
    assert not declares_abstention(f"Sorry.\n{BACKSTOP_SENTINEL}")


def test_prose_that_reads_like_a_refusal_declares_nothing():
    # The known limitation of ADR-0008, pinned as behaviour: a model that gives
    # up in its own words is answered as an ordinary answer. Asserted against
    # wording this module chose, never against model prose (ADR-0002).
    assert not declares_abstention("I don't know — the Evidence doesn't cover that.")


def test_an_ordinary_answer_declares_nothing():
    assert not declares_abstention("Round robin gives each process a fixed time slice.")


def test_the_fake_generator_never_declares_an_abstention():
    # The offline double stands in for a model that answers, so a change to its
    # text must not silently turn every seam test's answer into an Abstention.
    assert not declares_abstention(FakeGenerator().generate("some prompt"))
