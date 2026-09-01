"""The presentation strings both surfaces render, tested where they now live.

Not the CLI's tests: the same strings reach the terminal through `print_answer`
and the Week 5 page through widgets (#22), so what a card *says* is asserted
here and how a surface *frames* it is asserted in that surface's own tests --
the terminal's list bullet in test_cli.py, and nothing here.
"""

from dataclasses import replace

import presentation
from config import DISTANCE_THRESHOLD
from core.store import ChunkFilter
from presentation import describe_filter, evidence_card, provisional_threshold_notice
from tests.chunk_fixture import make_chunk


def configured_threshold(monkeypatch, **overrides):
    """The gate's threshold as config.py will one day set it.

    Patched where the notice reads it rather than handed to the notice: it
    takes no threshold, because neither surface has one of its own to give it,
    and Week 6 edits exactly this constant.
    """
    monkeypatch.setattr(
        presentation, "DISTANCE_THRESHOLD", replace(DISTANCE_THRESHOLD, **overrides)
    )


def test_an_evidence_card_names_the_document_title_source_type_and_locator():
    # The three fields a reader judges a Chunk by before trusting it, and the
    # whole card: no bullet, no widget, nothing that belongs to one surface.
    assert evidence_card(make_chunk()) == "hash table (note) — Hash Table › Buckets"


def test_an_evidence_card_names_whatever_shape_the_locator_took():
    # A card is written over the Locator and knows nothing about its shape, so
    # a page reaches the reader through the same card a heading path does
    # (CONTEXT.md -- a Locator is never named for any one shape).
    assert evidence_card(make_chunk(locator="p. 3")) == "hash table (note) — p. 3"


def test_a_filter_is_described_by_whichever_axes_it_restricts():
    assert describe_filter(ChunkFilter(domain="dsa")) == "domain dsa"
    assert describe_filter(ChunkFilter(source_type="note")) == "source type note"
    assert (
        describe_filter(ChunkFilter(domain="dsa", source_type="note"))
        == "domain dsa, source type note"
    )
    assert describe_filter(ChunkFilter()) == ""


def test_a_provisional_threshold_admits_its_tau_the_model_and_where_to_read_why(
    monkeypatch,
):
    # Read off a threshold that is nobody's configured value, so a notice that
    # restated today's tau as a literal of its own would fail here rather than
    # agree with config.py by coincidence.
    configured_threshold(
        monkeypatch, value=0.42, embedding_model="some-other-model", provisional=True
    )

    notice = provisional_threshold_notice()

    assert notice is not None
    assert "tau=0.42" in notice
    assert "some-other-model" in notice
    assert "ADR-0003" in notice


def test_a_derived_threshold_leaves_no_notice_at_all(monkeypatch):
    # Week 6 flipping the flag has to be the only edit (#22): no surface may
    # restate the provisional condition as a literal of its own, so the notice
    # has to vanish here rather than be suppressed by each surface separately.
    configured_threshold(monkeypatch, provisional=False)

    assert provisional_threshold_notice() is None


def test_the_shipped_notice_follows_the_configured_threshold_and_not_a_constant():
    # Holds on either side of the Week 6 sweep, which is the point of writing
    # it against the flag rather than against today's value of it.
    assert (provisional_threshold_notice() is not None) == DISTANCE_THRESHOLD.provisional
