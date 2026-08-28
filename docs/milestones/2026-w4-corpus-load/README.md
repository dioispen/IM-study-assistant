# 第 4 週 — P1 corpus load 驗證素材

issue #10。這個資料夾是 [根 README](../../../README.md) 「里程碑記錄 › 第 4 週」
那一節背後的原始輸出——已消化的表格在根 README,未消化的 transcript 在這裡。

| 檔案 | 內容 | 怎麼重跑 |
|---|---|---|
| `ingest_runs.txt` | 四次 `cli.py ingest` 的逐字輸出(intro-cs、dsa 各跑兩次),對應根 README §1、§2 | `python cli.py ingest <folder> --domain <domain>` |
| `ask_results.txt` | 11 題 hand-read + 3 個 abstention probe 的完整 transcript:每題的最近距離、gate 判定、`abstained` flag、pre-gate 的 Evidence 清單、答案全文、引用的 Evidence 卡片 | `python eval/milestone_ask.py > docs/milestones/2026-w4-corpus-load/ask_results.txt` |

`eval/milestone_ask.py` 是產生 `ask_results.txt` 的 driver,題目寫死在裡面。它
**不是**第 6 週的 eval harness(那支是 `eval/run_eval.py`,對 `gold_doc_ids`
評分,尚未存在)——這裡是質化閱讀,沒有評分規則。

`ask_results.txt` 是「當時實際問了什麼、答了什麼」的準據。driver 的題目清單日後
若被改動,以 transcript 裡印出的問題為準,不以 driver 現在的內容為準。

## 判讀是誰做的

ADR-0002 讓生成的散文由人讀、不進自動化套件。這一輪讀 `ask_results.txt`、寫出
根 README 那張表(判讀 + 成因)的是 agent,不是 repo 擁有者。`ask_results.txt`
留在版本控制裡,就是要讓判讀能被覆核——特別是標「正確」的那幾題。

## 為什麼放在 `docs/` 而不是 `data/`

`data/` 整個 gitignore(語料授權 + registry / Chroma 是衍生物,見
[docs/corpus-sources.md](../../corpus-sources.md))。里程碑的證據要能被別人看到,
所以走 `docs/`。答案本身沒有授權疑慮:問題是 agent 出的,回答是 gpt-4o-mini
就 Evidence 生成的。
