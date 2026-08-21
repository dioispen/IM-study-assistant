"""Command-line entry point: ingest a folder of notes, or ask a question and
get a cited answer with source cards."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from config import (
    DEFAULT_LANGUAGE,
    DISTANCE_THRESHOLD,
    DOMAINS,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    MAX_CHUNKS_PER_DOCUMENT,
    REGISTRY_PATH,
    TOP_K,
)
from core.ask import ask
from core.embedder import OpenAIEmbedder
from core.generator import OpenAIGenerator
from core.registry import Registry
from core.store import ChunkFilter, RetrievedChunk, VectorStore
from ingestion.common import FolderNotFound, OutsideCorpusRoot, ingest_folder


NO_CAP = "off"


def cap_argument(value: str) -> int | None:
    """`--max-per-document` as the cap `ask` takes: a positive int, or None.

    A parser-level type rather than a check inside cmd_ask, so a bad value is
    refused before a store is opened or a key is needed, and named on the usage
    line the reader is already looking at.

    "off" exists because no number means no cap: zero admits nothing and every
    question would abstain as though the corpus were empty. Uncapped retrieval
    is the baseline the Week 7 diversity experiment reads the cap against
    (PLAN.md §第 7 週), so it needs to be reachable from the command line
    without editing config.py.
    """
    if value == NO_CAP:
        return None
    try:
        cap = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value!r} is neither a whole number nor {NO_CAP!r}"
        ) from None
    if cap < 1:
        raise argparse.ArgumentTypeError(
            f"a cap of {cap} admits no Chunk into the Evidence, so every question "
            f"would abstain as though the corpus were empty; pass at least 1, or "
            f"{NO_CAP!r} for no cap"
        )
    return cap


def cmd_ingest(args: argparse.Namespace) -> None:
    registry = Registry(REGISTRY_PATH)
    store = VectorStore()
    embedder = OpenAIEmbedder(model=EMBEDDING_MODEL)

    try:
        report = ingest_folder(
            folder=args.folder,
            domain=args.domain,
            source_type=args.source_type,
            registry=registry,
            store=store,
            embedder=embedder,
            language=args.language,
        )
    except (FolderNotFound, OutsideCorpusRoot) as error:
        # A mistyped folder is the likeliest way to reach either, and both
        # messages already say what to do about it -- so it reads as an error
        # rather than as a traceback the reader has to interpret. Raised before
        # the walk, so no Document was read, no Chunk written and nothing
        # retired.
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print(report.summary())
    # Retirements are named one by one rather than only counted. A run that
    # deletes Chunks is the one outcome the reader cannot undo by re-running,
    # and when it is a surprise -- a folder moved out of the corpus root, a
    # vault half-synced -- the source_path is what tells them which notes to
    # put back. Not a warning: reorganising notes is the ordinary case, and the
    # retirement is exactly what was asked for.
    for retired in report.retired:
        print(retired.notice())
    # Failures go to stderr and name the file: a garbled note is a run the
    # reader has to act on, not a silent gap in the corpus.
    for failure in report.failed:
        print(failure.warning(), file=sys.stderr)


def source_card(chunk: RetrievedChunk) -> str:
    """One Evidence Chunk as the line a reader follows back to the note.

    One card per Chunk of the Evidence and none besides, which is what makes
    the cards report the answer's real footing: a question scoped to one Domain
    or capped per Document has fewer Chunks to show, and showing anything the
    Evidence does not hold would credit the answer to a Document generation was
    never given (CONTEXT.md: Evidence is the only thing an answer's citations
    may point at).
    """
    return f"- {chunk.title} ({chunk.source_type}) — {chunk.locator}"


def describe_filter(chunk_filter: ChunkFilter) -> str:
    """The restriction in the reader's words, for a message that has to say
    what the question was narrowed to."""
    return ", ".join(
        f"{label} {value}"
        for label, value in (
            ("domain", chunk_filter.domain),
            ("source type", chunk_filter.source_type),
        )
        if value is not None
    )


def cmd_ask(args: argparse.Namespace) -> None:
    # Asserted before the store is opened or a key is needed: a tau belonging
    # to another embedding model is a misconfiguration, not a bad answer.
    threshold = DISTANCE_THRESHOLD.for_model(EMBEDDING_MODEL)

    store = VectorStore()
    embedder = OpenAIEmbedder(model=EMBEDDING_MODEL)
    generator = OpenAIGenerator(model=GENERATION_MODEL)

    chunk_filter = ChunkFilter(domain=args.domain, source_type=args.source_type)

    answer = ask(
        args.question,
        embedder,
        store,
        generator,
        top_k=args.top_k,
        distance_threshold=threshold,
        chunk_filter=chunk_filter,
        max_chunks_per_document=args.max_per_document,
    )

    print(answer.text)
    if answer.abstained:
        # A scoped question abstains on the corpus it was scoped to, and the
        # abstention text speaks of "the corpus" -- so the restriction is named
        # here rather than leaving the reader to read it as "your notes do not
        # cover this" when the truth may be "not under that Domain".
        restriction = describe_filter(chunk_filter)
        if restriction:
            print()
            print(f"(The question was restricted to {restriction}.)")
        # An abstention is the one moment an underived tau visibly costs the
        # reader an answer, so that is where it admits it is underived.
        if DISTANCE_THRESHOLD.provisional:
            print()
            print(
                f"(The distance gate used tau={DISTANCE_THRESHOLD.value}, hand-set for "
                f"{DISTANCE_THRESHOLD.embedding_model} and not yet derived from the "
                "eval sweep — see ADR-0003.)"
            )
        return

    print()
    print("Sources:")
    for chunk in answer.evidence:
        print(source_card(chunk))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="im-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest a folder of .md and .docx notes"
    )
    # No --corpus-root override, for the reason --distance-threshold has none:
    # a Document is identified by its path below one root, and a per-run root
    # would put that identity back in the hands of the invocation, which is
    # exactly what the root exists to take it out of. Notes kept elsewhere move
    # CORPUS_ROOT in config.py.
    ingest_parser.add_argument(
        "folder",
        help=(
            "Folder of .md and .docx notes beneath the corpus root, including any "
            "in its subfolders. Both formats ingest in one run, routed by the "
            "file's own format"
        ),
    )
    ingest_parser.add_argument("--domain", required=True, choices=DOMAINS)
    ingest_parser.add_argument("--source-type", default="note")
    ingest_parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    ingest_parser.set_defaults(func=cmd_ingest)

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=TOP_K)
    # The scope of the question. Constrained to DOMAINS for the reason ingest
    # is: a Domain that does not exist matches no Chunk, and the resulting
    # abstention is indistinguishable from a corpus that genuinely covers
    # nothing -- so a typo would read as an answer about the corpus. Source
    # type takes no choices because it is an open set (CONTEXT.md) with no list
    # to check against.
    ask_parser.add_argument(
        "--domain", choices=DOMAINS, help="Restrict the question to one Domain"
    )
    ask_parser.add_argument(
        "--source-type", help="Restrict the question to one Source type"
    )
    # Unlike --distance-threshold, this one exists: the cap is a diversity
    # preference to be swept in Week 7, not a calibrated constant paired with an
    # embedding model, and turning it off for one question is how the sweep
    # reads the baseline.
    ask_parser.add_argument(
        "--max-per-document",
        type=cap_argument,
        default=MAX_CHUNKS_PER_DOCUMENT,
        metavar=f"{{N,{NO_CAP}}}",
        help=(
            "Most Chunks one Document may contribute to the Evidence, "
            f"or {NO_CAP!r} for no cap"
        ),
    )
    # No --distance-threshold override: ADR-0003 has tau swept and read off by
    # a stated rule, never hand-tuned per invocation, and a per-call float
    # would slip past the embedding-model pairing that DistanceThreshold exists
    # to enforce.
    ask_parser.set_defaults(func=cmd_ask)

    return parser


def main(argv: list[str] | None = None) -> None:
    # Titles, Locators and source paths may contain non-ASCII (notes
    # are zh-tw by default; Locators use "›"). Windows consoles often default
    # to a legacy codepage that can't encode either, so force UTF-8 on both
    # streams -- ingest failures name their file on stderr.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    # OPENAI_API_KEY reaches the OpenAI client through the environment, which
    # is what lets core/openai_client.py stay zero-argument. Loading .env here
    # rather than at import time preserves that module's laziness: importing
    # config or anything under core still costs nothing and needs no key, so
    # the offline tests are untouched by this.
    #
    # Pinned to the file beside this one, not searched for from the working
    # directory, so `im-rag` run from elsewhere reads the same key. A real
    # OPENAI_API_KEY already in the environment wins -- load_dotenv does not
    # override, so a shell export or a CI secret beats a stale .env in a
    # checkout, and .env is the fallback rather than the authority.
    load_dotenv(Path(__file__).resolve().parent / ".env")

    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
