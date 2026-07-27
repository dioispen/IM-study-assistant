import math

from core.embedder import FakeEmbedder


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def test_embed_is_deterministic():
    embedder = FakeEmbedder()

    first = embedder.embed(["red-black tree insertion"])
    second = embedder.embed(["red-black tree insertion"])

    assert first == second


def test_embed_returns_one_vector_per_text_of_configured_dimension():
    embedder = FakeEmbedder(dim=32)

    vectors = embedder.embed(["hello world", "goodbye"])

    assert len(vectors) == 2
    assert all(len(v) == 32 for v in vectors)


def test_shared_vocabulary_makes_texts_more_similar_than_unrelated_ones():
    embedder = FakeEmbedder()

    query = embedder.embed(["round robin scheduling time slice"])[0]
    related = embedder.embed(["round robin scheduling gives each process a time slice"])[0]
    unrelated = embedder.embed(["binary search tree insertion deletes rebalances"])[0]

    assert cosine(query, related) > cosine(query, unrelated)
