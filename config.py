"""Project-wide configuration: domains, storage paths, and pipeline parameters."""

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

# Section token thresholds for structured chunking. "Tokens" here means
# whitespace-split words (see core/chunking.py) -- a placeholder measure
# until a real tokenizer is picked (deferred to Week 5 tuning per PLAN.md).
MIN_SECTION_TOKENS = 40
MAX_SECTION_TOKENS = 300

EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o-mini"

TOP_K = 5

DEFAULT_LANGUAGE = "zh-tw"
