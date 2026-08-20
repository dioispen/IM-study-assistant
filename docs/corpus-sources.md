# Corpus sources

Third-party material used as ingestion input. **None of it is committed.**
`scripts/fetch_zh_corpus.py` pulls it into `data/corpus/`, which `/data/` in
`.gitignore` already covers.

## Why fetched instead of vendored

This project is MIT. Some good Chinese study material is not, and the mismatch
is not fixable by changing this project's license:

- **CC BY-NC-SA 4.0** (hello-algo) carries a NonCommercial term. MIT promises
  recipients unrestricted use including commercial use, which cannot be
  delivered for NC content. There is also no ShareAlike compatibility path for
  the NC variants — CC's compatible-license list exists only for plain BY-SA.
  Relicensing this repo to CC BY-NC-SA to "match" would make the code itself
  non-free, for the sake of test input.
- Both the NC and ShareAlike obligations are triggered by **sharing** the
  material. Downloading it, reading it, chunking it and embedding it locally is
  not sharing. Keeping the files out of every commit sidesteps the conflict
  entirely, at the cost of one fetch step.

The same reasoning covers the derived artifacts: chunked text and an embedding
index built from BY-NC-SA material are plausibly adapted material, so `data/`
(registry, Chroma store, corpus) stays uncommitted as a whole.

The rule for adding a source: if it cannot be committed, pin it here and fetch
it. If it can (public domain, CC0, CC BY, government open data), it may go in
the repo — but see the fixtures note below before putting it under `tests/`.

## Sources

### hello-algo

| | |
|---|---|
| Upstream | https://github.com/krahets/hello-algo |
| Author | krahets |
| License | CC BY-NC-SA 4.0 |
| Pinned commit | `69932aed1891a7b7f6a0de88cd116d3fe13e7032` |
| Sparse paths | `zh-hant/docs` |
| Lands at | `data/corpus/hello-algo/zh-hant/docs/` |
| Domain | `dsa` |

Traditional Chinese data-structures and algorithms text. Its value here is
being real zh-hant technical prose at volume — headings, tables, code fences,
mixed CJK/Latin runs — which is what `core/chunking.py` and the CJK token
measure of ADR-0004 need to be exercised against. Hand-written fixtures cannot
produce that volume or that mess.

```
python scripts/fetch_zh_corpus.py --source hello-algo
```

Re-running is idempotent; a checkout already at the pinned commit is skipped.
Bumping the corpus means editing `commit` in the script, and any tau swept
against the old commit has to be re-derived (ADR-0003).

**Ingesting it:** one run over the whole tree. `ingest_folder` walks nested
folders (#15, landed), so pointing it at `zh-hant/docs` covers every chapter
directory beneath it:

```
python cli.py ingest data/corpus/hello-algo/zh-hant/docs --domain dsa
```

The same walk is what retires: a Document ingested from beneath this folder and
no longer found there leaves the corpus with its Chunks, named in the run's
report (ADR-0005). So bumping the pinned commit and re-running is how the
corpus follows upstream — deletions included — rather than a manual cleanup.

## Why none of this belongs in `tests/fixtures/`

Beyond the license: `tests/test_ingest_ask_seam.py` pins a specific retrieval
geometry, and `_ingest_with_chinese` documents why the Chinese fixtures are
kept out of the store the gate tests read distances off. `FakeEmbedder` hashes
into 64 buckets, so one Chinese Chunk sized for the configured thresholds fills
at least 47 of them and the out-of-corpus trap collapses from 0.82 to 0.64 by
collision alone. Real corpus volume makes that far worse.

Committed fixtures also have to be deterministic and reviewable in a diff. A
fetched corpus is neither. Its place is manual evaluation and chunking stress
tests, guarded by a `skipif` on the path existing — never a test that fails on
a machine that has not run the fetch.
