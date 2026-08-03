"""Project-wide configuration: domains, storage paths, and pipeline parameters.

A parameter that must not be read on its own carries its own type here rather
than in the module that consumes it -- DistanceThreshold lives beside the value
it guards because core/gate.py already imports core/store.py, which imports
this module, so owning the type there would close an import cycle.
"""

from dataclasses import dataclass
from pathlib import Path

DOMAINS = [
    "intro-cs",
    "dsa",
    "os",
    "network",
    "security",
    "ai",
    "mis",
    "design-pattern",
]

DATA_DIR = Path(__file__).resolve().parent / "data"
REGISTRY_PATH = DATA_DIR / "documents.sqlite"
CHROMA_PATH = DATA_DIR / "chroma"
CHUNK_COLLECTION_NAME = "chunks"

# The one root every Document's source_path -- and so its doc_id -- is derived
# relative to. Ingestion is pointed at a folder beneath it, and a file's
# identity is the path from here down rather than the path from whichever
# folder the run was handed. Two consequences worth the setting living here
# rather than being passed per run: the same note keeps one doc_id however
# ingestion is invoked, and two same-named notes in different folders beneath
# the root cannot collide onto one Document. Nothing absolute enters the
# derivation, so the same corpus on another machine derives the same ids --
# which is what ADR-0001's registry is built on. To ingest notes kept
# elsewhere, point this at the root they already share.
CORPUS_ROOT = DATA_DIR / "corpus"

# Section token thresholds for structured chunking. "Tokens" here means what
# core/tokenization.py counts: one CJK codepoint is one token, and a run of
# non-CJK word characters is one token. That heuristic is a placeholder --
# close enough to a real tokenizer in both languages for these numbers to
# mean roughly the same thing in each -- until a real tokenizer is picked
# (deferred to Week 5 tuning per PLAN.md).
MIN_SECTION_TOKENS = 40
MAX_SECTION_TOKENS = 300

EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o-mini"

TOP_K = 5


@dataclass(frozen=True)
class DistanceThreshold:
    """τ for the distance gate, carried together with the embedding model it
    was set for (ADR-0003: τ is a property of the embedding model, not of the
    system) and with whether it has actually been derived yet.
    """

    value: float
    embedding_model: str
    provisional: bool

    def for_model(self, embedding_model: str) -> float:
        """The τ to gate with, or a loud failure if it belongs to another model."""
        if embedding_model != self.embedding_model:
            raise RuntimeError(
                # Spelled "tau" rather than "τ": this reaches a console via an
                # uncaught traceback on stderr, which nothing reconfigures.
                f"Distance threshold tau={self.value} was set for embedding model "
                f"{self.embedding_model!r}, but {embedding_model!r} is in use. "
                "Re-derive tau for the new model -- carrying one across models "
                "silently confounds a model comparison with a mis-set gate."
            )
        return self.value


# Abstention layer 1 of ADR-0003. PROVISIONAL: not derived, hand-picked to be
# permissive until the Week 5 sweep reads the real value off the eval set (the
# largest τ whose false abstention rate on answerable questions stays ≤ 5%).
# The asymmetry justifies erring loose: a wrong gate abstention never reaches
# the LLM and is unrecoverable, whereas weak Evidence let through still meets
# the prompt backstop. Distances are cosine and lower-is-closer, so the gate
# reads `distance > τ` and a larger τ is the more permissive one.
DISTANCE_THRESHOLD = DistanceThreshold(
    value=0.85,
    embedding_model="text-embedding-3-small",
    provisional=True,
)

DEFAULT_LANGUAGE = "zh-tw"
