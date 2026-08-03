# IM-study-assistant

A RAG question-answering system over Information Management study notes and
literature. See `PLAN.md` for the full eight-week project plan.

## Output language

Write in Traditional Chinese (繁體中文), keeping domain vocabulary in English, for:

- **Commit messages** — both the subject line and the body.
- **The closing explanation after `/implement`** — the wrap-up written once the work is committed.

**Domain vocabulary stays in English.** That means the terms `CONTEXT.md` defines
(Document, Chunk, Locator, Evidence, Domain, Source, abstention, …), identifiers and
symbol names (`chunk_markdown`, `doc_id`, `MIN_SECTION_TOKENS`), file paths, ADR
numbers, issue references, and anything quoted from code or from an English document.
Do not translate them and do not coin Chinese equivalents — a glossary term appears in
Chinese prose in its English form, so that the ubiquitous language stays one language.

```
中文 fixture 在真實門檻下觸發三條 chunking 分支

oversize 分支原本只在 seam test 覆寫的 MIN_TOKENS / MAX_TOKENS 下執行,
在 config.py 的 MAX_SECTION_TOKENS = 300 之下從未被走過。
```

Everything else in the repo stays in English: code, comments, docstrings, `CONTEXT.md`,
`PLAN.md`, ADRs, and GitHub issues — issues match the existing tracker and are picked
up by AFK agents.

## Agent skills

### Issue tracker

Issues and PRDs live as GitHub issues in `dioispen/IM-study-assistant`, managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, each label string equal to its name. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
