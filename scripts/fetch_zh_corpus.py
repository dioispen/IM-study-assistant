"""Fetches the third-party Chinese corpora into `data/corpus/`, which is
gitignored -- these files are deliberately not vendored into the repository.

hello-algo is CC BY-NC-SA 4.0. Its NonCommercial and ShareAlike terms cannot be
satisfied by a repository licensed MIT, and both terms are triggered only by
*sharing* the material. Fetching it onto one machine to read and index is not
sharing, so the whole conflict disappears as long as the files never enter a
commit. That is the only reason this script exists rather than a `corpus/`
directory -- see docs/corpus-sources.md.

Each source is pinned to a commit, so the corpus a distance sweep was measured
against can be reconstructed later. Re-running is cheap and idempotent: a
checkout already at the pinned commit is left alone.
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Derived from __file__ rather than imported from config.py: this script is run
# as `python scripts/fetch_zh_corpus.py`, which puts scripts/ on sys.path
# instead of the repo root, so `import config` would not resolve.
CORPUS_DIR = REPO_ROOT / "data" / "corpus"


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    # Pinned, never a branch name: an unpinned corpus silently changes under a
    # tau that was swept against the old one (ADR-0003).
    commit: str
    # Cone-mode sparse-checkout paths. Narrowing to the Chinese docs keeps the
    # checkout to the material actually ingested rather than the whole book,
    # including its code samples in twelve languages.
    paths: tuple[str, ...]
    license: str


SOURCES = (
    Source(
        name="hello-algo",
        url="https://github.com/krahets/hello-algo.git",
        commit="69932aed1891a7b7f6a0de88cd116d3fe13e7032",
        paths=("zh-hant/docs",),
        license="CC BY-NC-SA 4.0 — do not commit, do not redistribute",
    ),
)


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({result.returncode}):\n"
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def _current_commit(dest: Path) -> str | None:
    """The commit `dest` is checked out at, or None if it is not a checkout yet."""
    if not (dest / ".git").exists():
        return None
    try:
        return _git("rev-parse", "HEAD", cwd=dest)
    except RuntimeError:
        # A half-finished fetch from an interrupted earlier run. Reporting "not
        # a checkout" makes the next fetch repair it in place.
        return None


def fetch(source: Source, force: bool = False) -> Path:
    dest = CORPUS_DIR / source.name

    if not force and _current_commit(dest) == source.commit:
        print(f"{source.name}: already at {source.commit[:8]}, nothing to do")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        _git("init", "--quiet", str(dest))
        _git("remote", "add", "origin", source.url, cwd=dest)

    _git("sparse-checkout", "init", "--cone", cwd=dest)
    _git("sparse-checkout", "set", *source.paths, cwd=dest)

    print(f"{source.name}: fetching {source.commit[:8]} from {source.url}")
    # --filter=blob:none with --depth 1 pulls one commit's trees and only the
    # blobs the sparse paths need; the rest of the book's history never lands.
    _git("fetch", "--depth", "1", "--filter=blob:none", "origin", source.commit, cwd=dest)
    _git("checkout", "--quiet", "--force", source.commit, cwd=dest)

    files = sum(1 for _ in dest.rglob("*.md"))
    print(f"{source.name}: {files} Markdown files at {dest}")
    print(f"{source.name}: {source.license}")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fetch_zh_corpus",
        description="Fetch pinned third-party Chinese corpora into data/corpus/ (gitignored).",
    )
    parser.add_argument(
        "--source",
        choices=[s.name for s in SOURCES],
        help="Fetch only this source (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if the checkout is already at the pinned commit.",
    )
    args = parser.parse_args(argv)

    # Source paths and progress lines are ASCII, but git's stderr on a failure
    # may not be, and a legacy Windows console codepage cannot encode it.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    selected = [s for s in SOURCES if args.source in (None, s.name)]
    for source in selected:
        try:
            fetch(source, force=args.force)
        except RuntimeError as error:
            print(f"ERROR: {source.name}: {error}", file=sys.stderr)
            return 1
        except FileNotFoundError:
            print("ERROR: git is not on PATH", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
