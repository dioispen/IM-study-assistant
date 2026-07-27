# 資管知識庫 RAG 問答系統 — 兩個月專案計劃書

> 目標:從零打造一個以「資管相關筆記與文獻」為知識庫的 RAG(Retrieval-Augmented Generation)問答系統,
> 過程中徹底理解 embedding、向量檢索、chunking、prompt 設計等核心技術,而非只會呼叫框架。

---

## 一、專案概述

### 1.1 一句話描述
一個能用自然語言提問的「資管學習助手」:整合個人筆記、維基百科、網路文章與論文,
回答計概、DSA、OS、網路、資安、AI、MIS、Design Pattern 等領域的問題,並標明答案來源。

### 1.2 專案目標
- **學習目標(主要)**:親手實作 RAG 完整流程,理解每個環節的原理與取捨。
- **產品目標(次要)**:做出自己期中期末真的會拿來用的學習工具。
- **產出目標**:一個結構清晰的 GitHub repo + 詳細 README(含架構圖、實驗記錄、踩坑心得)。

### 1.3 設計原則
1. **第一版不用 LangChain**:純 Python 手刻 pipeline,先懂原理,行有餘力最後再用框架重寫比較。
2. **主題聚焦、來源多元**:先做熟 2–3 個領域,pipeline 穩定後再水平擴充其他領域。
3. **領域可擴充**:領域(domain)只是 metadata 的一個欄位,新增領域 = 丟新文件 + 標新標籤,
   不需要改動系統架構。
4. **誠實面對極限**:系統答不出來就說不知道;README 記錄失敗案例與原因分析。

---

## 二、知識庫設計

### 2.1 領域規劃(可持續新增)

| 領域代號 | 領域名稱 | 優先序 | 備註 |
|---|---|---|---|
| `intro-cs` | 計算機概論 | P1 | 基礎,各來源材料最好找 |
| `dsa` | 資料結構與演算法 | P1 | 筆記量通常最多 |
| `os` | 作業系統 | P2 | |
| `network` | 計算機網路 | P2 | |
| `security` | 資訊安全 | P2 | |
| `ai` | 人工智慧 / 機器學習 | P2 | 與本專案本身高度相關 |
| `mis` | 管理資訊系統 | P3 | 質化內容多,適合 RAG |
| `design-pattern` | 設計模式 | P3 | |
| `(未來新增)` | — | — | 只需新增 domain 標籤即可 |

> **建議**:第 3–5 週只做 P1 的兩個領域,把 pipeline 與檢索品質做穩;
> 第 6 週起再批次匯入 P2、P3。領域擴充應該是「一個下午就能完成」的例行操作,
> 如果做不到,代表 ingestion 流程要重構。

### 2.2 資料來源與對應工具

| 來源類型 | 格式 | 抽取工具 | 注意事項 |
|---|---|---|---|
| 個人筆記 | md / docx / PDF / OneNote 匯出 | 原生讀取 / `python-docx` / `pymupdf` | 格式最亂但價值最高;手寫掃描檔第一版先跳過 |
| 維基百科 | API | `wikipedia` 套件 | 不要用爬的;先列好各領域條目清單(每領域 10–30 條) |
| 網路文章 | HTML | `trafilatura` | 抽正文去雜訊;記得保存原始 URL 當出處 |
| 論文 | PDF(常見雙欄) | `pymupdf` | 每領域挑 3–10 篇經典即可;抽取後務必人工抽查品質 |

### 2.3 Metadata Schema(本專案的靈魂)

Document 是第一級實體,擁有跨越任何一次 chunking 的穩定身分;Chunk 從屬於它。
兩者分開存放,理由見 [ADR-0001](docs/adr/0001-document-registry-alongside-the-vector-store.md)。
名詞定義見 [CONTEXT.md](CONTEXT.md)。

**Document registry**(SQLite 或 JSON manifest):

```python
{
    "doc_id": "a3f9c2e1b004",              # sha256(source_path)[:12],穩定不變
    "title": "紅黑樹筆記",
    "domain": "dsa",                       # 領域,每份文件恰好一個
    "source_type": "note",                 # 出處 = 誰寫的,開放式列舉
    "source_path": "notes/dsa/rbtree.md",  # 或 URL
    "language": "zh-tw",                   # 只給評測切片用,不做檢索過濾
    "content_hash": "sha256(...)",         # 沒變就跳過重新匯入
    "ingested_at": "2026-08-01"
}
```

**Chunk metadata**(存進 ChromaDB):

```python
{
    "chunk_id": "a3f9c2e1b004:007",   # doc_id:ordinal,可重現,不是隨機 uuid
    "doc_id": "a3f9c2e1b004",
    "ordinal": 7,
    "locator": "紅黑樹 › 插入操作",     # 「文件中的哪裡」,一定有值
    # 以下為了 metadata 過濾,從 Document 反正規化過來
    "domain": "dsa",
    "source_type": "note",
    "title": "紅黑樹筆記"
}
```

四個關鍵約定:

- **`domain` 每份文件恰好一個。** 跨領域文件(B-tree 條目算 `dsa` 還是資料庫?)就選主要的。
  這個決定很好反悔——embedding 不依賴 metadata,日後要加多領域標記只需 `collection.update()`,
  不必重新 embedding。
- **`source_type` 只回答「這是誰寫的」。** 不編碼抽取管線,也不編碼權威性。
  講義投影片、教科書章節出現時就新增值,跟新增領域一樣便宜。
- **`locator` 取代原本的 `section`。** 引用要指的是「文件中的哪裡」:筆記是標題路徑、
  PDF 是頁碼、維基是錨點——同一個概念,三種寫法。若沿用 `section`,論文(最需要精確引用的來源)
  會直接是 null。
- **`chunk_id` 不用 uuid。** 隨機 ID 會讓兩次實驗的 chunk 完全對不起來,而反覆對同一批語料
  重新 chunking 正是第 5 週的主要工作。

這份 schema 支撐後續三個關鍵功能:
1. **來源標註**:回答時顯示「此段來自你的 OS 筆記(插入操作)」vs「來自維基百科」。
2. **過濾檢索**:「只查我的筆記」「只查資安領域」。
3. **衝突偵測**:不同來源說法不一致時指出差異——但範圍已縮小,見第 6 週。

> **已知盲點**:系統分不出「獨立來源」與「改寫自另一份來源」。如果某份筆記是照著維基百科寫的,
> 兩者「說法一致」並不構成佐證,而是同一個來源被數了兩次。README 的已知限制要寫明這點。

---

## 三、技術架構

### 3.1 技術選型

| 元件 | 選擇 | 理由 |
|---|---|---|
| 語言 | Python 3.11+ | |
| LLM(生成) | OpenAI 或 Anthropic API | 小專案總花費通常 < 數百元台幣 |
| Embedding | OpenAI `text-embedding-3-small` 起步 | **Anthropic 沒有 embeddings endpoint**——生成與 embedding 是兩個獨立的選擇。第 5 週把本地 BGE-M3 拉進來當實驗項 |
| 向量資料庫 | ChromaDB | pip 裝了就能用,免架伺服器,原生支援 metadata 過濾。建 collection 時要指定 `hnsw:space="cosine"`(預設是 L2,事後想改必須重建 collection 並重新 embedding) |
| 前端介面 | Streamlit | 聊天介面約 50 行;可部署 Streamlit Cloud(免費) |
| PDF 抽取 | pymupdf | 中文支援優於 pypdf |
| 正文抽取 | trafilatura | 網路文章去雜訊 |
| 版本控制 | Git + GitHub | 從第一天就 commit |

### 3.2 系統流程

```
[各類文件] → ingestion 腳本(每種來源一支)
    → 統一格式(純文字 + metadata)
    → 寫入 Document registry(content_hash 沒變就整份跳過)
    → chunking(結構化來源依標題切;論文/文章退回固定長度 + 重疊)
    → embedding(批次呼叫 API)
    → ChromaDB 入庫
                                ┌─ 過濾條件(領域/來源)
[使用者提問] → 問題 embedding ──┴→ 向量檢索 top-k
    → 同一份 Document 最多取 N 段(來源多樣性)
    → 距離閘門:最近的 chunk 都比 τ 遠 → 直接回「不知道」,不呼叫 LLM
    → 組 prompt(Evidence + 出處 + 防幻覺指示)
    → LLM 生成回答(附來源);Evidence 撐不住時由 prompt backstop 自行說不知道
    → Streamlit 顯示
```

### 3.3 專案目錄結構(建議)

```
mis-rag/
├── ingestion/
│   ├── ingest_notes.py
│   ├── ingest_wiki.py
│   ├── ingest_articles.py
│   ├── ingest_papers.py
│   └── common.py          # 共用:chunking(兩條路徑)、locator、入庫
├── core/
│   ├── registry.py        # Document registry 讀寫;doc_id、content_hash
│   ├── embedder.py
│   ├── retriever.py       # 檢索 + metadata 過濾 + 每份文件取段上限
│   ├── gate.py            # 距離閘門;τ 由 sweep 推導,不寫死
│   └── generator.py       # prompt 組裝 + LLM 呼叫
├── eval/
│   ├── questions.json     # 20 題可答(附 gold_doc_ids)+ 30 題陷阱
│   ├── run_eval.py        # Recall@k、MRR、三種 abstention rate
│   └── sweep_tau.py       # 掃 τ 並套用選點規則
├── data/
│   ├── documents.sqlite   # Document registry
│   └── chroma/            # 向量庫
├── app.py                 # Streamlit 介面
├── config.py              # 領域清單、chunk 參數、embedding 模型
└── README.md
```

> 新增領域時,只需要:準備文件 → 在 `config.py` 加一行領域定義 → 跑對應 ingestion 腳本。

---

## 四、八週時程規劃

### 第 1 週:原理打底 — Embedding 與相似度
- [ ] 申請 API key,設定好開發環境與 Git repo
- [ ] 寫小程式:把 10 個句子轉 embedding,計算 cosine similarity
- [ ] 親眼驗證「語意相近 → 向量相近」
- [ ] **跨語言檢查(本週最重要的一項)**:準備幾組互為翻譯的中英句子,看中文句子是否真的
      靠近對應的英文句子。是 → 語料是一個池子;否 → 中文問題永遠碰不到英文論文,
      而且產出的答案看起來仍然合理,評測抓不到。這是整個專案最容易無聲失敗的地方,
      現在十個句子就測得完,等到第 5 週才發現就是重跑整條 pipeline
- [ ] 練習 LLM API 基本呼叫與 prompt 撰寫
- **里程碑**:能解釋 embedding 是什麼,並對「這批語料是一個池子還是兩個」給出有證據的答案

### 第 2 週:迷你 RAG 原型
- [ ] 手刻迷你版:10–20 段文字 → embedding → 提問 → 找最相關段落 → LLM 依據段落回答
- [ ] 加入「檢索不到相關內容就說不知道」的 prompt 設計
- [ ] 初次體驗:故意問知識庫沒有的問題,觀察行為
- **里程碑**:200 行以內、能跑的命令列迷你 RAG

### 第 3 週:Ingestion — 多來源資料進場
- [ ] 確定 P1 領域(建議 `intro-cs` + `dsa`)
- [ ] 完成四支 ingestion 腳本(筆記 / wiki / 文章 / 論文)
- [ ] 每種來源人工抽查抽取品質,記錄問題(斷行、亂碼、雜訊)
- [ ] 定案 metadata schema
- **里程碑**:P1 兩領域的四種來源都能轉成「乾淨文字 + metadata」

### 第 4 週:Chunking 與入庫
- [ ] 實作兩條 chunking 路徑,依 `source_type` 分流:
  - **結構化**(筆記、維基):依標題切,**不跨標題**;過小的段落往後合併,
    過大的段落切成多段但共用同一個 locator
  - **非結構化**(論文、文章):固定長度 + 重疊。兩欄 PDF 的標題偵測是另一個專案,
    不要在這週開這個坑(§5 已列為風險)
- [ ] 產生 locator:標題路徑 / 頁碼 / 錨點,三種來源都要有值
- [ ] 建立 Document registry,實作「content_hash 沒變就整份跳過」
- [ ] 批次 embedding 並寫入 ChromaDB(**建 collection 時記得 `hnsw:space="cosine"`**)
- [ ] 實作 metadata 過濾檢索(依領域、依來源類型)
- [ ] 串起完整問答流程(命令列版)
- **里程碑**:完整 pipeline 跑通,能對 P1 領域問答並附出處;重跑同一批文件不會產生重複 chunk

### 第 5 週:建立評測 + 第一輪優化

評測是**自動**的:不評分生成的答案,只評檢索。理由見
[ADR-0002](docs/adr/0002-evaluate-retrieval-not-generated-answers.md)——要調的參數
全部在檢索階段,而人工評分 3 種 chunk 大小 × 3 種 top-k × 全部題目是一輪好幾個小時,
「每次改動都重跑」根本不會發生。

- [ ] 出測試集,**刻意偏向陷阱題**。可答題要獵 gold document,很貴;陷阱題一句話一題,很便宜——
      平均分配等於用同樣的預算買兩種單價差很多的東西:
  - **可答題 20 題**,每題標 `gold_doc_ids`(標在 Document 層級,換 chunk 大小標籤才不會作廢)
    - 單領域事實題 8(「什麼是 TLB?」)
    - 理解/比較題 6(「quicksort 和 mergesort 的取捨?」)
    - 跨來源題 6(筆記和維基都提到的主題)
  - **陷阱題 30 題,分兩種、分開計分**:
    - 場外題 10:語料完全沒碰過的主題(「什麼是 CRISPR?」)→ 測距離閘門,幾乎必過
    - 擦邊題 20:主題有、那個特定事實沒有(有 B-tree 筆記但沒寫刪除操作)→ 測 prompt backstop。
      **這才是會產生自信幻覺的那一類**;和場外題合併計分就會被平均掉,兩層設計也就白做了
- [ ] 評測腳本輸出:
  - `Recall@k`、`MRR`(對 20 題可答題)
  - false abstention rate(可答題卻說不知道)
  - abstention rate ×2(場外題、擦邊題各一,**永不相加**)
  - 每一列都要帶 embedding 模型欄位——不同模型的分數不能互比
- [ ] 掃 τ 並套用選點規則:**在 false abstention ≤ 5%(20 題最多錯 1 題)的前提下取最大的 τ**
      ([ADR-0003](docs/adr/0003-two-layer-abstention-with-a-derived-threshold.md))
- [ ] 實驗 → **記錄成表**。chunking 分成兩條路徑後,「chunk 大小」已經不是單一參數,
      要分開掃、分開切片,否則會得出「chunk 大小影響不大」這種被設計本身製造出來的假結論
      (你的筆記是最大宗來源,絕大多數 gold document 走結構化那條路,掃 `window` 動不了它):

  | 參數 | 影響哪條路徑 | 切片方式 |
  |---|---|---|
  | `window` / `overlap` | 非結構化 | 只看 gold document 是論文/文章的題目 |
  | `min_section_tokens` / `max_section_tokens` | 結構化 | 只看 gold document 是筆記/維基的題目 |
  | `top-k` | 兩條都影響 | 全部題目 |
  | embedding 模型(3-small vs 本地 BGE-M3) | 兩條都影響 | 全部題目,**τ 要各自重掃** |

- **里程碑**:有數據支撐的第一輪優化報告 + τ 掃描曲線(兩者都是 README 素材)

### 第 6 週:進階優化 + 領域擴充
- [ ] 匯入 P2 領域(os / network / security / ai),驗證「新增領域一個下午完成」
- [ ] 進階實驗(擇 2–3 項,以下四項都能用第 5 週的儀器量):
  - 回答附精確出處(locator)
  - **來源多樣性**:每份 Document 取段上限或 MMR → 用 Recall@k 驗證沒變差。
    順手解掉「top-5 全是同一個章節切出來的五段」這個浪費 context 的問題
  - 多領域混合提問的檢索品質觀察(檢索結果是否偏向某類來源?)
  - Query rewriting:先讓 LLM 改寫問題再檢索
- [ ] 衝突偵測(**範圍已縮小**):在來源多樣性之上加一層 prompt——當 Evidence 橫跨
      ≥2 份 Document 時,請模型指出說法不一致處。這一項**只做展示,不宣稱驗證過**:
      現有儀器量不到它,而且系統分不出獨立來源與改寫來源(見 §2.3 的已知盲點)。
      README 要照 §1.3.4 老實寫明——把這條原則套在自己的 README 上,才算真的遵守它
- **里程碑**:6+ 領域上線,至少完成兩項**可量測**的進階實驗並記錄

### 第 7 週:Streamlit 介面
- [ ] 聊天式問答介面(含對話歷史)
- [ ] 側邊欄:領域篩選、來源類型篩選
- [ ] 回答下方顯示引用來源卡片
- [ ] (選配)上傳新文件即時入庫
- **里程碑**:可以 demo 的網頁版

### 第 8 週:收尾與發布
- [ ] 匯入 P3 領域(mis / design-pattern)
- [ ] README:架構圖、安裝步驟、實驗數據表、踩坑記錄、已知限制
- [ ] 部署到 Streamlit Cloud(注意 API key 用 secrets 管理)
- [ ] (選配挑戰)用 LangChain 重寫一版,寫一篇兩者比較心得
- **里程碑**:公開的 GitHub repo + 可線上試用的 demo

---

## 五、風險與對策

| 風險 | 徵兆 | 對策 |
|---|---|---|
| 中文 PDF 抽取亂碼 | 第 3 週論文/筆記抽出來不能看 | 換 pymupdf;仍不行則該文件先跳過,不要卡死 |
| 範圍膨脹 | 想一次匯入所有領域 | 嚴守 P1→P2→P3 順序;pipeline 沒穩不加資料 |
| 檢索品質差 | 答非所問 | 回到第 5 週評測集,依 chunking 路徑切片後逐項實驗,不要瞎調 |
| 幻覺(編造答案) | **擦邊題**答錯——場外題全過不代表沒事 | 兩層防線:距離閘門擋場外題、prompt backstop 擋擦邊題;兩種 abstention rate 永遠分開看 |
| 中文問題檢索不到英文材料 | 答案永遠只引用中文來源,但看起來仍然合理,分數也不一定難看 | 第 1 週就用十個句子測跨語言相似度;評測時依「提問語言 × gold document 語言」切片看 Recall |
| 評測標籤失效 | 換了 chunk 大小,標好的答案位置全對不上 | gold 標在 Document 層級而非 chunk;chunk_id 由 doc_id 推導而非隨機 |
| API 花費焦慮 | — | embedding 便宜;開發期用小模型,demo 再換好模型;設用量上限 |
| 時間不夠 | 第 6 週還沒完成第 4 週里程碑 | 砍選配項目;Streamlit 介面極簡化;P3 領域延後 |

---

## 六、學習產出清單(做完你會的東西)

- Embedding 與向量相似度的原理與實作
- 向量資料庫(ChromaDB)操作與 metadata 過濾檢索
- Chunking 策略設計與 A/B 比較方法
- 防幻覺設計:機械式閾值 + prompt 判斷的分工,以及兩者失誤代價不對稱時該偏向哪邊
- 異質資料源(筆記/wiki/文章/論文)的 ingestion 工程
- 建立評測集、用數據驅動優化的工作方法
- **評測設計本身**:挑對量測工具(量檢索而不是量文字)、把閾值從掃描推導出來而不是手調、
  以及怎麼看出一個數字是被實驗設計製造出來的
- Streamlit 快速打造 AI 應用介面與雲端部署
- (選配)框架 vs 手刻的取捨判斷

---

## 七、下一步

1. ~~建立 GitHub repo,把這份計劃書放進去當 `PLAN.md`~~ ✅
2. 申請 API key
3. 開始第 1 週第一個任務:10 個句子的 embedding 相似度實驗——**其中的跨語言那一組先做**,
   因為它會決定「這批語料是一個池子還是兩個」,而這件事越晚發現越貴

> 名詞定義見 [CONTEXT.md](CONTEXT.md);幾個不好反悔的決定與理由見 [docs/adr/](docs/adr/)。

> 提醒:每週結束花 15 分鐘寫「本週學到什麼、卡在哪」的短記錄,
> 第 8 週寫 README 時它們會是最好的素材。
