import hashlib
import math
import re

from core.embedder import FakeEmbedder
from core.tokenization import tokenize

DIM = 64


# The hashing trick, isolated from the tokenization that feeds it, so the tests
# below can hold the hashing constant and vary only how the text was cut into
# tokens. It duplicates FakeEmbedder._embed_one's arithmetic on purpose;
# test_the_embedder_hashes_the_tokens_the_chunker_counts is what ties the two
# together, so a change to the real bucket or norm rule fails there rather than
# leaving this helper quietly testing something the pipeline no longer does.
def hash_embed(tokens):
    vector = [0.0] * DIM
    for token in tokens:
        vector[int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


# The tokenization FakeEmbedder used before it shared the chunker's: a run of
# word characters, whatever script they are written in. Kept as the historical
# baseline the two migration guards below measure against -- not a second live
# measure, which ADR-0004 reserves to core/tokenization.py alone.
LEGACY_WORD_RE = re.compile(r"[\w']+")


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


def test_the_embedder_hashes_the_tokens_the_chunker_counts():
    # ADR-0004: production chunking and the test double must cut text the same
    # way, or a fixture's Chunks and the question meant to retrieve them are
    # split by two different rules and the geometry proves nothing.
    embedder = FakeEmbedder(dim=DIM)
    text = "死結偵測 deadlock detection 的成本"

    [vector] = embedder.embed([text])

    assert vector == hash_embed(t.text for t in tokenize(text.lower()))


def test_english_embeddings_are_unchanged_by_the_shared_tokenizer():
    # The gate tests pin distances measured before the swap; punctuation
    # already bounded a word for both rules, so English must not move at all.
    #
    # A migration guard for this one swap, not a standing invariant. ADR-0004
    # says tests must not pin the measure, because Week 5 replaces it with a
    # real tokenizer -- and records that doing so "shifts English counts as
    # well as Chinese ones". So this test is expected to fail at that swap, and
    # the correct response then is to delete it along with LEGACY_WORD_RE and
    # re-derive the gate geometry, never to repair it.
    embedder = FakeEmbedder(dim=DIM)
    texts = [
        "round robin scheduling time quantum",
        "baroque counterpoint fugue harpsichord ornamentation",
        "Deleting a node with two children requires finding its in-order successor.",
        "What are the four conditions required for deadlock to occur?",
    ]

    vectors = embedder.embed(texts)

    assert vectors == [hash_embed(LEGACY_WORD_RE.findall(t.lower())) for t in texts]


def test_a_chinese_clause_embeds_as_its_characters_not_as_one_bucket():
    # The whole point of the swap: Chinese has no whitespace between words, so
    # the legacy rule hashed a punctuation-bounded clause into a single bucket
    # and every Chinese text sat at distance 1.0 from every other.
    embedder = FakeEmbedder(dim=DIM)
    clause = "資源配置圖中不存在環路"

    [vector] = embedder.embed([clause])

    assert sum(1 for v in vector if v) > 1
    assert sum(1 for v in hash_embed(LEGACY_WORD_RE.findall(clause)) if v) == 1


def test_shared_chinese_vocabulary_makes_texts_more_similar_than_unrelated_ones():
    embedder = FakeEmbedder()

    query = embedder.embed(["三向交握的第三個步驟"])[0]
    related = embedder.embed(["三向交握的第三個步驟由發起端送出確認封包"])[0]
    unrelated = embedder.embed(["應用層與傳輸層之間由通訊埠號辨別不同服務"])[0]

    assert cosine(query, related) > cosine(query, unrelated)
