"""Unit tests for the distance gate -- abstention layer 1 of ADR-0003."""

import pytest

from config import DISTANCE_THRESHOLD, EMBEDDING_MODEL
from core.gate import ABSTENTION_TEXT, should_abstain
from core.store import RetrievedChunk


def chunk_at(distance: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="doc1:000",
        doc_id="doc1",
        locator="Process Scheduling › Round Robin",
        domain="os",
        source_type="note",
        title="Scheduling Notes",
        text="Round robin gives each process a fixed time slice.",
        distance=distance,
    )


def test_nearest_chunk_farther_than_tau_abstains():
    assert should_abstain([chunk_at(0.9), chunk_at(0.95)], threshold=0.85)


def test_nearest_chunk_within_tau_does_not_abstain():
    assert not should_abstain([chunk_at(0.4), chunk_at(0.95)], threshold=0.85)


def test_only_the_nearest_chunk_decides_not_the_farther_ones():
    # Evidence arrives nearest-first, but the gate must not depend on ordering.
    assert not should_abstain([chunk_at(0.95), chunk_at(0.4)], threshold=0.85)


def test_a_chunk_exactly_at_tau_passes_the_gate():
    # ADR-0003 reads the gate as `distance > tau`, so tau itself is inside.
    assert not should_abstain([chunk_at(0.85)], threshold=0.85)


def test_no_evidence_at_all_abstains():
    assert should_abstain([], threshold=0.85)


def test_abstention_text_says_it_does_not_know():
    assert "don't know" in ABSTENTION_TEXT.lower()


def test_shipped_tau_is_paired_with_the_embedding_model_in_use():
    # ADR-0003: tau is a property of the embedding model. A tau carried across
    # models would silently confound a model comparison with a mis-set gate.
    assert DISTANCE_THRESHOLD.embedding_model == EMBEDDING_MODEL


def test_shipped_tau_is_marked_provisional_until_the_week_5_sweep_derives_it():
    assert DISTANCE_THRESHOLD.provisional


def test_shipped_tau_is_permissive():
    # ADR-0003's asymmetric-cost argument: a wrong gate abstention is final,
    # while wrongly passed weak Evidence still meets the prompt backstop. On
    # cosine distances (0 = identical, 1 = orthogonal) a permissive placeholder
    # sits near the orthogonal end.
    assert DISTANCE_THRESHOLD.value >= 0.8


def test_using_tau_with_the_model_it_was_set_for_returns_it():
    assert DISTANCE_THRESHOLD.for_model(DISTANCE_THRESHOLD.embedding_model) == (
        DISTANCE_THRESHOLD.value
    )


def test_using_tau_with_a_different_embedding_model_fails_loudly():
    with pytest.raises(RuntimeError, match="(?i)re-derive tau"):
        DISTANCE_THRESHOLD.for_model("text-embedding-3-large")
