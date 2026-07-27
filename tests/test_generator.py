from core.generator import FakeGenerator, build_prompt
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


def test_prompt_instructs_abstention_when_evidence_does_not_support_an_answer():
    prompt = build_prompt("What is round robin?", EVIDENCE)

    assert "don't know" in prompt.lower() or "do not know" in prompt.lower()


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

    assert "don't know" in prompt.lower() or "do not know" in prompt.lower()


def test_fake_generator_never_makes_network_calls_and_is_deterministic():
    generator = FakeGenerator()

    first = generator.generate("some prompt")
    second = generator.generate("some prompt")

    assert first == second
    assert isinstance(first, str)
