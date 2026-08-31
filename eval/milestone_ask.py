"""Week 4 milestone (issue #10): hand-read sample + both abstention probes.

Not the Week 6 eval harness (that is eval/run_eval.py, still to come, scored
against gold_doc_ids). This is a one-off qualitative read: a fixed list of
questions run through the real pipeline so a human can read the generated prose
(ADR-0002 keeps generated text out of the automated suite; a milestone is when
someone reads it).

Prints, per question: the nearest retrieval distance, whether the distance gate
fired, which Abstention the pipeline reported -- layer 2 declares itself since
ADR-0008, so it is read off the Answer rather than out of the prose -- the
answer, and the Evidence, both what retrieval returned pre-gate and what the
answer cited. `retrieve` is called separately from `ask` only to expose
the pre-gate distances, which `ask` does not return.

    python eval/milestone_ask.py > docs/milestones/2026-w4-corpus-load/ask_results.txt

Needs OPENAI_API_KEY (env or .env beside cli.py) and an ingested corpus. Each
question costs one text-embedding-3-small call plus, unless it abstains, one
gpt-4o-mini call. Answers are not deterministic -- a re-run can differ on the
borderline cases.
"""

import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from config import (
    DISTANCE_THRESHOLD,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    MAX_CHUNKS_PER_DOCUMENT,
    TOP_K,
)
from core.ask import ask
from core.embedder import OpenAIEmbedder
from core.generator import OpenAIGenerator
from core.retriever import retrieve
from core.store import ChunkFilter, VectorStore

tau = DISTANCE_THRESHOLD.for_model(EMBEDDING_MODEL)
store = VectorStore()
embedder = OpenAIEmbedder(model=EMBEDDING_MODEL)
generator = OpenAIGenerator(model=GENERATION_MODEL)

# The hand-read sample: (label, question, --domain scope or None). Chosen after
# reading the corpus, spread across single-fact / comparison / cross-document,
# with one Domain-scoped question. No gold_doc_ids -- this is a qualitative read.
SAMPLE = [
    ("intro-cs", "什麼是二補數（2's complement）?它相較於一補數有什麼優點?", None),
    ("intro-cs", "IEEE 754 單精度浮點數的 32 個位元是怎麼分配的?", None),
    ("intro-cs", "機器指令週期分成哪幾個階段?每個階段做什麼?", None),
    ("intro-cs", "漢明碼和單純的同位位元在錯誤處理能力上有什麼差別?", None),
    ("intro-cs", "為什麼storage hierarchy能有效運作?", None),
    ("dsa", "quicksort 和 merge sort 的取捨是什麼?", None),
    ("dsa", "雜湊衝突有哪些解決方法?", None),
    ("dsa", "AVL 樹是如何維持平衡的?", None),
    ("dsa", "什麼是二元搜尋樹?它的查詢時間複雜度為何?", None),
    ("dsa", "圖的 BFS 和 DFS 走訪有什麼不同?", None),
    ("dsa (scoped)", "堆積（heap）的插入操作是怎麼運作的?", "dsa"),
]

# ADR-0003's two abstention layers, probed one at a time and scored apart. An
# out-of-corpus trap should be stopped by the distance gate (layer 1); a
# near-miss trap should be stopped by the prompt backstop (layer 2).
PROBES = [
    ("LAYER 1 — out-of-corpus trap", "什麼是 CRISPR-Cas9 基因編輯技術?", None),
    ("LAYER 2 — near-miss trap", "紅黑樹（red-black tree）的節點插入會做哪些變色與旋轉操作?", None),
    ("LAYER 2 — near-miss trap (intro-cs)", "格雷碼（Gray code）要如何做兩數相加的運算?", None),
]


def run(question: str, domain: str | None) -> None:
    chunk_filter = ChunkFilter(domain=domain) if domain else None
    evidence = retrieve(
        question,
        embedder,
        store,
        top_k=TOP_K,
        chunk_filter=chunk_filter,
        max_chunks_per_document=MAX_CHUNKS_PER_DOCUMENT,
    )
    nearest = min((c.distance for c in evidence), default=None)
    answer = ask(
        question,
        embedder,
        store,
        generator,
        top_k=TOP_K,
        distance_threshold=tau,
        chunk_filter=chunk_filter,
        max_chunks_per_document=MAX_CHUNKS_PER_DOCUMENT,
    )
    print(f"nearest_distance = {nearest!r}  (tau = {tau})")
    gate = "GATE FIRED (layer 1)" if (not evidence or nearest > tau) else "gate passed"
    print(f"gate: {gate}")
    print(f"abstention: {answer.abstention.value}")
    print("retrieval (pre-gate) Evidence:")
    for chunk in evidence:
        print(f"  d={chunk.distance:.4f}  [{chunk.domain}] {chunk.title}  ›  {chunk.locator}")
    print("--- answer ---")
    print(answer.text)
    # Whatever the Answer cites, an abstention included: layer 2 judged
    # Evidence that was assembled and handed over, and reading whether it gave
    # up too early is what a milestone hand-read is for (ADR-0008). Read off
    # the Evidence rather than off the reason, so this cannot come to disagree
    # with which reason carries cards.
    if answer.evidence:
        print("--- cited Evidence cards ---")
        for chunk in answer.evidence:
            print(f"  - {chunk.title} ({chunk.source_type}) — {chunk.locator}")
    print()


print("#" * 78)
print("# HAND-READ SAMPLE")
print("#" * 78)
for label, question, scope in SAMPLE:
    print("=" * 78)
    print(f"[{label}] {question}")
    print("=" * 78)
    run(question, scope)

print("#" * 78)
print("# ABSTENTION PROBES  (ADR-0003 — layers scored separately, never summed)")
print("#" * 78)
for label, question, scope in PROBES:
    print("=" * 78)
    print(f"{label}\nQ: {question}")
    print("=" * 78)
    run(question, scope)
