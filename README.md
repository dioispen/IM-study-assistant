# IM Study Assistant

一套建立在個人資管學習語料上的 retrieval-augmented 問答系統。回答會標明出處,
語料裡沒有答案時也會直說。

- **名詞定義與難以反悔的決定**:[CONTEXT.md](CONTEXT.md)(ubiquitous language)、
  [docs/adr/](docs/adr/)(ADR)。
- **八週計劃**:[PLAN.md](PLAN.md)。
- **語料出處**:[docs/corpus-sources.md](docs/corpus-sources.md)。

## 執行方式

把 ingestion 指向 `data/corpus/` 底下的資料夾(corpus root 定在 `config.py`);
其下每個 `.md`、`.docx`、`.pdf` 會在一次執行裡一起匯入,依檔案格式分流。

```
python cli.py ingest data/corpus/intro-cs-fundamentals --domain intro-cs
python cli.py ingest data/corpus/hello-algo/zh-hant/docs --domain dsa
python cli.py ask "AVL 樹是如何維持平衡的?"
python cli.py ask "只查我的 dsa 筆記" --domain dsa
```

重跑 ingestion 是常態:沒變的 Document 不花任何成本,變了的 Document 整批換掉
Chunk,原路徑上找不到的 Document 則被 retire(ADR-0005)。`ingest` 與 `ask`
都需要環境變數 `OPENAI_API_KEY`,或放在 `cli.py` 旁的 `.env`。

## 里程碑記錄

每個週里程碑寫下的「哪裡通了、哪裡壞了、為什麼」的短記錄——第 8 週寫 README
時的素材(PLAN.md §八)。生成的回答由人來讀,不進測試套件(ADR-0002):自動化
套件只斷言 retrieval,而里程碑正是有人去讀它產出的散文的時候。

### 第 4 週 — P1 corpus load(2026-08-28)

issue #10。整條 pipeline,跑在真實的 `intro-cs` + `dsa` 語料上,由人閱讀。範圍
只有筆記——Markdown 與 docx(PLAN.md §五);windowed / PDF 路徑存在(ADR-0007)
但沒有任何 P1 筆記走它。

#### 1. Corpus 匯入

| Domain | 來源 | Document 數 | Chunk 數 | `ingested_at` |
|---|---|---|---|---|
| `intro-cs` | `data/corpus/intro-cs-fundamentals/`(5 份 Markdown,自有的 LLM 改寫筆記——見 [docs/corpus-sources.md](docs/corpus-sources.md)) | 5 | 22 | 2026-08-28 |
| `dsa` | `data/corpus/hello-algo/zh-hant/docs/`(hello-algo,pin 在 `69932aed`,CC BY-NC-SA,fetch 而非 vendor) | 119 | 764 | 2026-08-04 |
| **合計** | | **124** | **786** | |

每份資料夾底下磁碟上的 `.md` 都在 registry 裡——雙向都查過,沒有漏、沒有多餘。
P1 語料結果全是 Markdown;docx 路徑(結構化,共用同一個 `chunk_sections`)有
fixture 覆蓋,但還沒有真實的 P1 筆記走過它。這次執行只有一種 `source_type`
(全 124 份都是 `note`),所以無法看出 retrieval 是否偏好某一種 Source(這個
觀察在 PLAN.md §5.3 的接回檢查清單上,不在本里程碑)。它能看出、也確實看出了
Domain 的不平衡——見 §3。

`dsa` 這半邊維持 2026-08-04 的 `ingested_at`:之後的 chunking commit(docx 分流、
windowed 路徑、`Section` 重構)沒有動到 `chunk_sections` 的 merge-and-split 規則
與 token 計數,所以 `content_hash`(取自抽取後的文字)不變,每一份 hello-algo
Document 都 skip。store 裡的 Chunk 與今天的 code 產物 byte 相同。用 `core/chunking.py`
在該區間的 diff 核對過。

#### 2. 重跑的 idempotence

兩個 ingest 立即各重跑一次:

```
intro-cs, run 2:  Ingested 0, skipped 5 unchanged, retired 0, failed 0.
dsa,      run 2:  Ingested 0, skipped 119 unchanged, retired 0, failed 0.
```

每一份 Document 都 skip,沒有 retire、沒有重新 embedding。acceptance criteria
1 與 2:達成。

#### 3. 人工閱讀樣本

11 題,gpt-4o-mini 生成答案,`top_k=5`、`MAX_CHUNKS_PER_DOCUMENT=2`、暫定 `τ=0.85`。

逐題的原始輸出(最近距離、gate 判定、pre-gate Evidence、答案全文、引用卡片)
在 [`docs/milestones/2026-w4-corpus-load/ask_results.txt`](docs/milestones/2026-w4-corpus-load/);
產生它的 driver 是 [`eval/milestone_ask.py`](eval/milestone_ask.py)。

| # | Domain | 問題(節略) | 判讀 | 原因 |
|---|---|---|---|---|
| 1 | intro-cs | 二補數相較一補數的優點 | **部分** | 用 `dsa` 的 `number encoding` 加 hello-algo `小結 › Q&A` 回答;專門的 `03-complement-systems` 筆記從沒進前五。答案不算錯,但漏掉筆記的框架(「把減法轉成加法,只需一套加法器」)。跨 Domain 滲漏。 |
| 2 | intro-cs | IEEE 754 單精度位元配置 | **正確** | s=1、e=8、f=23;逐字取自 `04-floating-point › 4-3`,引用的 Chunk 正確。 |
| 3 | intro-cs | 機器指令週期的階段 | **正確** | 五個階段加暫存器,逐字取自筆記的表格。 |
| 4 | intro-cs | 漢明碼 vs parity check | **正確** | 正確(Hamming 更正單一位元錯誤,parity 只能偵測)。偏薄——漏了同一節的 SECDED / 最小 Hamming 距離,但問題本身問得窄。 |
| 5 | intro-cs | storage hierarchy 為何有效 | **正確** | 時間 / 空間 locality,熱資料放上層;逐字。 |
| 6 | dsa | quicksort vs merge sort 的取捨 | **false abstention** | retrieval 拿到的是 `merge_sort.md` / `quick_sort.md` 的 section-intro Chunk——大半是 mkdocs 圖片 tab 標記(`=== "<1>" ![…]`,見 §4)。真正談「演算法特性」的段落排更後面;backstop 在 Evidence 撐不住時正確 abstain。 |
| 7 | dsa | 雜湊衝突的解決方法 | **部分 / 誤導** | 回傳的是筆記的策略前言(「改良結構讓它在衝突下仍能運作」「只在嚴重時才擴容」),而非「鏈式位址 / 開放定址」。命名的 Chunk(`雜湊衝突 › 鏈式位址`)前五名輸給 `小結`、`hash algorithm` 和一個 DP Chunk。 |
| 8 | dsa | AVL 樹如何維持平衡 | **正確** | \|balance factor\| > 1 時旋轉,插入 / 刪除後由底向頂,中序序列不變。好。 |
| 9 | dsa | BST 與查詢複雜度 | **正確** | O(log n),退化成鏈結串列時劣化為 O(n)。內容正確;模型引用了 `小結` 而非排第一的 BST 專屬 Chunk——citation 品質的小瑕疵。 |
| 10 | dsa | 圖的 BFS vs DFS | **false abstention** | 五個 Chunk 全是 BFS(`graph_traversal › 廣度優先走訪` ×2、一個練習、兩個二元樹走訪 Chunk)。沒有任何 DFS Chunk 進 Evidence,所以「比較」本來就撐不起來;backstop 正確 abstain。 |
| 11 | dsa(`--domain dsa`) | heap 的插入操作 | **正確** | 加到底部,再 sift-up / heapify 到 root。Domain scope 正常運作。 |

**計分:** 7 正確、2 部分、2 false abstention。

反覆出現的失敗模式,由重到輕:

1. **比較題會餓死 backstop**(#6、#10)。retrieval 把 `top_k` 五格全填給語意較
   近的那一側,另一側從沒進 Evidence,於是對語料「答得出來」的題目 abstain。
   這是最清楚的模式,也是第 6 週 `top_k` sweep 與任何 query rewriting 實驗該
   瞄準的目標。
2. **hello-algo 的 `小結 › 重點回顧 › Q & A` 與 section-0 Chunk 洗版**(#1、#7、
   #9)。每章的小結頁和圖片 tab 很多的 section-0 Chunk 又短、關鍵詞又密,把真正
   含答案的段落擠出前五。
3. **跨 Domain 滲漏**(#1)。一題 `intro-cs` 的問題被 `dsa` 的 Chunk 回答掉,
   因為 hello-algo 同主題(`number encoding`)量大、embed 更近,勝過單一份
   intro-cs 筆記。未 scoped 的問題沒有防線;`--domain` 能擋(#11 就很乾淨)。

#### 4. 抽取品質抽查

直接從 store 拉 Chunk 文字用眼睛看。沒有亂碼、沒有壞掉的編碼、沒有 mojibake——
標題變成 `›` 串起來的 Locator 路徑,兩份語料的表格與 code fence 都完整保留。
問題全是**從 hello-algo 的 mkdocs 原始檔帶進來的 noise**,沒有一個是致命的:

- **mkdocs content tab 變成內文。** `=== "Python"`、`=== "JS"`、`=== "<1>"`
  不是 Markdown 標題,`parse_markdown_sections` 不會在它們上面切,於是逐字落進
  Chunk 內。
- **多語言 code block 撐大 Chunk。** `avl_tree.md` 的「節點高度」一節是單一個
  約 2400 字的 Chunk,內容幾乎全是同一段 `TreeNode` 類別重複約 12 種語言。
  `docs/corpus-sources.md` 特地把 sparse-checkout 收窄以避開「十二種語言的
  code sample」,但它們是透過 tab 內嵌在 zh-hant 的 `.md` 裡,收窄路徑並沒有
  拿掉它們。
- **`![alt](x.assets/y.png)` 圖片參照留著。** `quick_sort.md` 的第一個 Chunk
  帶了九行連續的 `![…](pivot_division_stepN.png)`。
- **`` ```src [file]{quick_sort}-[class]{…} `` include 指令留著**,是佔位字串——
  真正的 code 是 mkdocs build 時從 `codes/` 拉進去的,在語料裡根本不存在。跟
  tab block 不一致,後者是**有**內嵌的。
- **約 14 個章節 `index.md` 「封面」頁變成 Document**——每個一個約 50–140 字的
  Chunk,內容是封面圖參照加一句詩意的 `!!! abstract` 題辭(「向日葵朝著太陽
  轉動…」)。零事實內容;124 份 Document 裡有 20 份是單 Chunk,大多是這些。
- **oversize-split 的 Chunk 從句子中間切斷。** 超過 `MAX_SECTION_TOKENS` 的一節
  會切成共用一個 Locator 的多個 Chunk(設計如此),但切點不看句子——`quick_sort`
  Chunk 0 結在「原陣列被劃分成三部分:左子陣列、」,Chunk 1 開頭是「基準數、
  右子陣列…」。
- **per-Document cap 可能被同一節用光。** `MAX_CHUNKS_PER_DOCUMENT=2` 之下,
  同一節 oversize-split 出來的兩半(同一個 Locator)能把一份 Document 的兩格都
  佔掉——在 #1、#8、#10 的 Evidence 裡看得到。第 7 週的 diversity 實驗(MMR vs
  cap)就是量這個的地方。

以上都沒有擋下 ingestion(全程 `failed 0`)。第 6 週 chunking 的待辦:embedding
前把 content-tab / 純圖片的行拿掉,並考慮排除 `index.md` 封面頁。

#### 5. 兩層 abstention,分開驗(ADR-0003)

兩層永遠不合併成一個數字。各自單獨驗。

**Layer 1 — distance gate(場外題 / out-of-corpus trap)。** 問題:「什麼是
CRISPR-Cas9 基因編輯技術?」——語料完全沒碰過的主題。

- 最近的 retrieval 距離 **0.7221**,對暫定 `τ=0.85` → **gate 沒有觸發**。最近的
  Chunk 是 `dsa › 編輯距離問題`(「edit distance」在 embedding 空間撞上「基因
  編輯」),接著是 `insertion sort`,再來是泛用的中文技術散文。
- 這題還是被 abstain 掉——由 **layer 2**,prompt backstop(「我不知道」)。
- **發現:** 在 `τ=0.85` 下 distance gate 沒發揮作用——一個全然陌生的主題就落
  在它裡面。這正是第 6 週 sweep 的用途:在答得出來的題目上 false abstention
  rate ≤ 5% 的前提下取最大的 `τ`。語料是密集的中文技術散文,那個 `τ` 大概會
  落在低很多的地方(約 0.5–0.6)。記為負結果:layer 1 驗過,且明確沒有作用。

**Layer 2 — prompt backstop(擦邊題 / near-miss trap)。** 兩題,問語料**有**
碰但缺該事實的主題:

- 「紅黑樹的節點插入會做哪些變色與旋轉操作?」——語料有平衡 BST 材料(AVL)但
  沒有紅黑樹。gate 放行(最近 0.372,AVL 旋轉 Chunk);backstop **正確 abstain**
  ——沒有拿 AVL 旋轉或通識來頂。
- 「格雷碼要如何做兩數相加的運算?」——筆記定義了格雷碼的性質但沒講在它上面
  做運算。gate 放行(最近 0.395);backstop **正確 abstain**,沒有掰一套程序。

**Layer 2 也「過度觸發」**在兩題**答得出來**的問題上(#6、#10),因為 retrieval
餓死了比較的其中一側——見 §3。所以 backstop 照設計運作;要處理的是上游的
retrieval,那是第 6 週的事。

#### 為第 6 週埋的待辦

- Sweep `τ`——暫定的 0.85 gate 是空轉的(§5)。
- 比較題是主要的失敗模式(§3);讓 `top_k` 與 query rewriting 瞄準它。
- Chunking:embedding 前去掉 content-tab / 純圖片 / `index.md` 封面的 noise(§4)。
- 第 5 週的介面必須把兩種「我不知道」顯示成不同的東西(PLAN.md §第 5 週)——
  這次執行確認了它們的成因確實不同。
