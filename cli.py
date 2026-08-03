"""Command-line entry point: ingest a folder of Markdown notes, or ask a
question and get a cited answer with source cards."""

import argparse
import sys

from config import (
    DEFAULT_LANGUAGE,
    DISTANCE_THRESHOLD,
    DOMAINS,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    REGISTRY_PATH,
    TOP_K,
)
from core.ask import ask
from core.embedder import OpenAIEmbedder
from core.generator import OpenAIGenerator
from core.registry import Registry
from core.store import VectorStore
from ingestion.common import ingest_folder


def cmd_ingest(args: argparse.Namespace) -> None:
    registry = Registry(REGISTRY_PATH)
    store = VectorStore()
    embedder = OpenAIEmbedder(model=EMBEDDING_MODEL)

    report = ingest_folder(
        folder=args.folder,
        domain=args.domain,
        source_type=args.source_type,
        registry=registry,
        store=store,
        embedder=embedder,
        language=args.language,
    )
    print(report.summary())
    # Failures go to stderr and name the file: a garbled note is a run the
    # reader has to act on, not a silent gap in the corpus.
    for failure in report.failed:
        print(failure.warning(), file=sys.stderr)


def cmd_ask(args: argparse.Namespace) -> None:
    # Asserted before the store is opened or a key is needed: a tau belonging
    # to another embedding model is a misconfiguration, not a bad answer.
    threshold = DISTANCE_THRESHOLD.for_model(EMBEDDING_MODEL)

    store = VectorStore()
    embedder = OpenAIEmbedder(model=EMBEDDING_MODEL)
    generator = OpenAIGenerator(model=GENERATION_MODEL)

    answer = ask(
        args.question,
        embedder,
        store,
        generator,
        top_k=args.top_k,
        distance_threshold=threshold,
    )

    print(answer.text)
    if answer.abstained:
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
        print(f"- {chunk.title} ({chunk.source_type}) — {chunk.locator}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="im-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest a folder of Markdown notes"
    )
    ingest_parser.add_argument(
        "folder", help="Folder of .md notes, including any in its subfolders"
    )
    ingest_parser.add_argument("--domain", required=True, choices=DOMAINS)
    ingest_parser.add_argument("--source-type", default="note")
    ingest_parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    ingest_parser.set_defaults(func=cmd_ingest)

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=TOP_K)
    # No --distance-threshold override: ADR-0003 has tau swept and read off by
    # a stated rule, never hand-tuned per invocation, and a per-call float
    # would slip past the embedding-model pairing that DistanceThreshold exists
    # to enforce.
    ask_parser.set_defaults(func=cmd_ask)

    return parser


def main(argv: list[str] | None = None) -> None:
    # Titles, Locators and source paths may contain non-ASCII (Markdown notes
    # are zh-tw by default; Locators use "›"). Windows consoles often default
    # to a legacy codepage that can't encode either, so force UTF-8 on both
    # streams -- ingest failures name their file on stderr.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
