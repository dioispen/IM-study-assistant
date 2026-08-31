"""The command-line entry point's own surface: the arguments a scoped question
is asked with, and the source cards its Evidence is rendered as.

Nothing here opens a store or reaches a model -- `cmd_ask` builds the real
OpenAI clients, so what is testable offline is the parsing in front of it and
the rendering behind it. That the cards really do reflect filtered, capped
Evidence is asserted at the seam, in test_filtered_retrieval_seam.py, where a
store exists to filter.
"""

import pytest

from cli import build_parser, describe_filter, print_answer, source_card
from config import DISTANCE_THRESHOLD, MAX_CHUNKS_PER_DOCUMENT
from core.ask import Abstention, Answer
from core.gate import ABSTENTION_TEXT
from core.generator import BACKSTOP_ABSTENTION_TEXT, BACKSTOP_SENTINEL
from core.store import ChunkFilter, RetrievedChunk


def parse(*argv):
    return build_parser().parse_args(list(argv))


def make_chunk(**overrides):
    fields = dict(
        chunk_id="doc1:000",
        doc_id="doc1",
        locator="Hash Table › Buckets",
        domain="dsa",
        source_type="note",
        title="hash table",
        text="A hash table stores each key in a bucket.",
        distance=0.2,
    )
    return RetrievedChunk(**{**fields, **overrides})


def test_a_question_can_be_asked_unrestricted():
    args = parse("ask", "What is a hash collision?")

    assert ChunkFilter(domain=args.domain, source_type=args.source_type) == ChunkFilter()


def test_a_question_can_be_restricted_to_one_domain():
    args = parse("ask", "What is a hash collision?", "--domain", "dsa")

    assert args.domain == "dsa"
    assert args.source_type is None


def test_a_question_can_be_restricted_to_one_source_type():
    args = parse("ask", "What is a hash collision?", "--source-type", "note")

    assert args.source_type == "note"
    assert args.domain is None


def test_a_question_can_be_restricted_to_both_at_once():
    args = parse(
        "ask", "What is a hash collision?", "--domain", "dsa", "--source-type", "note"
    )

    assert ChunkFilter(domain=args.domain, source_type=args.source_type) == ChunkFilter(
        domain="dsa", source_type="note"
    )


def test_a_domain_that_does_not_exist_is_refused_rather_than_matching_nothing():
    # A misspelled Domain matches no Chunk, and the abstention that follows
    # looks exactly like a corpus that covers nothing -- so it is refused at
    # the argument rather than answered as if it were a fact about the corpus.
    with pytest.raises(SystemExit):
        parse("ask", "What is a hash collision?", "--domain", "dsaa")


def test_the_cap_defaults_to_the_configured_one_and_can_be_overridden():
    assert parse("ask", "q").max_per_document == MAX_CHUNKS_PER_DOCUMENT
    assert parse("ask", "q", "--max-per-document", "1").max_per_document == 1


def test_the_cap_can_be_turned_off_from_the_command_line():
    # Uncapped retrieval is the baseline the Week 7 diversity experiment reads
    # the cap against, so it has to be reachable without editing config.py --
    # and None, not a number, is what says "no cap" to `ask`.
    assert parse("ask", "q", "--max-per-document", "off").max_per_document is None


@pytest.mark.parametrize("value", ["0", "-1"])
def test_a_cap_that_admits_no_chunk_is_refused_at_the_argument(value):
    # A cap below one admits nothing, so every question would abstain as though
    # the corpus were empty -- refused before a store is opened rather than
    # answered as if it were a fact about the corpus.
    with pytest.raises(SystemExit):
        parse("ask", "q", "--max-per-document", value)


def test_a_cap_that_is_neither_a_number_nor_off_is_refused():
    with pytest.raises(SystemExit):
        parse("ask", "q", "--max-per-document", "none")


def test_a_source_card_names_the_title_source_type_and_locator():
    card = source_card(make_chunk())

    assert "hash table" in card
    assert "(note)" in card
    assert "Hash Table › Buckets" in card


def test_a_filter_is_described_by_whichever_axes_it_restricts():
    assert describe_filter(ChunkFilter(domain="dsa")) == "domain dsa"
    assert describe_filter(ChunkFilter(source_type="note")) == "source type note"
    assert (
        describe_filter(ChunkFilter(domain="dsa", source_type="note"))
        == "domain dsa, source type note"
    )
    assert describe_filter(ChunkFilter()) == ""


def gate_abstention(chunk_filter=ChunkFilter()):
    return Answer(
        text=ABSTENTION_TEXT,
        evidence=[],
        abstention=Abstention.DISTANCE_GATE,
        chunk_filter=chunk_filter,
    )


def test_a_gate_abstention_prints_its_text_and_cites_nothing(capsys):
    # Which layer declined is read off the Answer, so the terminal never has to
    # recognise an abstention by the words it is written in.
    print_answer(gate_abstention())

    out = capsys.readouterr().out
    assert ABSTENTION_TEXT in out
    assert "Sources:" not in out


def test_a_gate_abstention_names_the_restriction_it_was_asked_under(capsys):
    # The abstention text speaks of "the corpus", so a scoped question says
    # what it was scoped to -- read off the Answer's own filter rather than off
    # whatever scope happens to be in force at display time.
    print_answer(gate_abstention(ChunkFilter(domain="dsa", source_type="note")))

    out = capsys.readouterr().out
    assert "restricted to domain dsa, source type note" in out


def test_an_unrestricted_abstention_names_no_restriction(capsys):
    print_answer(gate_abstention())

    assert "restricted to" not in capsys.readouterr().out


def test_an_abstention_admits_an_underived_tau_for_exactly_as_long_as_it_is_one(capsys):
    # An abstention is the one moment a provisional tau visibly costs the
    # reader an answer. Tied to the threshold's own flag, so Week 6 deriving it
    # removes the notice with no edit here.
    print_answer(gate_abstention())

    out = capsys.readouterr().out
    assert (f"tau={DISTANCE_THRESHOLD.value}" in out) == DISTANCE_THRESHOLD.provisional


def test_an_ordinary_answer_prints_a_source_card_per_evidence_chunk(capsys):
    answer = Answer(
        text="A collision is two keys landing in one bucket.",
        evidence=[make_chunk(), make_chunk(chunk_id="doc2:000", doc_id="doc2", title="hashing")],
        abstention=Abstention.NONE,
        chunk_filter=ChunkFilter(),
    )

    print_answer(answer)

    out = capsys.readouterr().out
    assert "Sources:" in out
    assert out.count("- ") == 2
    assert source_card(answer.evidence[0]) in out


def backstop_abstention(evidence=None):
    return Answer(
        text=BACKSTOP_ABSTENTION_TEXT,
        evidence=[make_chunk()] if evidence is None else evidence,
        abstention=Abstention.PROMPT_BACKSTOP,
        chunk_filter=ChunkFilter(),
    )


def test_a_backstop_abstention_keeps_the_cards_it_was_judged_from(capsys):
    # The backstop fired over Evidence that was assembled and handed to the
    # model, so the cards are exactly the material it judged insufficient --
    # and the reader needs them in order to disagree (#21).
    print_answer(backstop_abstention())

    out = capsys.readouterr().out
    assert BACKSTOP_ABSTENTION_TEXT in out
    assert "Sources:" in out
    assert source_card(make_chunk()) in out


def test_a_backstop_abstention_shows_no_contract_token_to_the_reader(capsys):
    print_answer(backstop_abstention())

    assert BACKSTOP_SENTINEL not in capsys.readouterr().out


def test_the_two_abstentions_are_told_apart_by_which_cards_they_print(capsys):
    # The pair that makes ADR-0003's two layers visible on the terminal: same
    # branch, opposite outcomes, and neither read off the answer's text.
    print_answer(gate_abstention())
    gated = capsys.readouterr().out
    print_answer(backstop_abstention())
    backstopped = capsys.readouterr().out

    assert "Sources:" not in gated
    assert "Sources:" in backstopped
