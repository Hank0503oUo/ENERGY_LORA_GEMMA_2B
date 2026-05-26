# Router-Strict v0.3 Data Pipeline

讓上一輪卡在 **val accuracy 21–24%** 的 router LoRA 突破到 60%+ 的所有資料處理工具。

## 為什麼上一輪失敗

| 問題 | 證據 | 影響 |
|---|---|---|
| system prompt 沒列合法 tool 清單 | Cell 7 渲染輸出第一段沒有 `query_energy_records` 等字樣 | 模型不知道輸出要從固定 vocab 選 → 中文 hallucinate |
| 14 個 tool ≤ 5 樣本（最少 1 樣本） | manifest.tool_distribution | 罕見 tool val 必錯 |
| `__refusal__` 占訓練資料 28%（93/335） | manifest.tool_distribution | 模型 over-trained 成 refusal-loving |
| epoch=8 訓到 train_loss 0.007 但 val_loss 卡 3.14 | training log | 純背書，val 完全沒泛化 |
| `save_total_limit=2` 把 best epoch (3) 刪掉 | training log | 留下來的是最 overfit 的 |

## 修法總覽

```
00_config.py                 ← 唯一的真理：tool 清單 + canonical system prompt
01_apply_tool_list_system.py ← Step 1：把清單塞進三個 jsonl 的 system 訊息
02_dedupe_train.py           ← 移除 6 筆 duplicate user prompt
03_augment_with_teacher.py   ← Step 2：請 teacher LLM 對冷門 tool 補資料
04_merge_and_rebuild_manifest.py  ← 把生成資料併進 train + 重生 manifest
05_check_distribution.py     ← 任何階段都可以跑來看現況
```

訓練腳本 `colab_train_router_strict_lora.py` 也已加 `load_best_model_at_end`、
`metric_for_best_model="eval_loss"`、`save_total_limit=5`，自動回滾到 val 最佳的 checkpoint。

---

## 順序

### Step 0 — 看現況

```bash
cd /content/drive/MyDrive/energy_lora_router_v02
python scripts/05_check_distribution.py
```

預期看到：
- ✗ canonical system prompt NOT applied yet
- 多個 tool train < 10
- `__refusal__` 28% of train

### Step 1 — 改 system prompt（5 秒）

```bash
python scripts/01_apply_tool_list_system.py
```

備份會放在 `data/_backup/`。

### Step 2 — 去重複（5 秒）

```bash
python scripts/02_dedupe_train.py
```

### Step 3 — augment（API 或 BYO-CLI、約 10–30 分鐘）

挑一個 teacher LLM。**推薦：DeepSeek V4 Flash via YuhuanStudio**（最便宜、中文最強）。

#### Provider 比較

| Provider | 推薦度 | 中文 | 價格 | 速度 | 設定 |
|---|---|---|---|---|---|
| **deepseek (V4 Flash)** | ⭐⭐⭐ | 極強 | 極低 | 快 | YuhuanStudio aggregator |
| gemini (2.0 Flash) | ⭐⭐ | 強 | 免費 1500 RPM | 中 | Google AI Studio |
| openai (4o-mini) | ⭐⭐ | 中 | 中 | 中 | OpenAI |
| anthropic (Haiku 4.5) | ⭐ | 中 | 中 | 快 | Anthropic |
| modelscope | ⭐⭐ | 強 | 免費（限額） | 中 | ModelScope（中國） |

#### A. DeepSeek V4 Flash via YuhuanStudio（推薦）

```bash
pip install openai
export YUHUAN_API_KEY=...   # 從 https://api.yuhuanstudio.com 拿

# dry-run 看計畫
python scripts/03_augment_with_teacher.py --provider deepseek --target 50 --dry-run

# 真的跑
python scripts/03_augment_with_teacher.py --provider deepseek --target 50

# 也可以指定其他 YuhuanStudio 上的 model
python scripts/03_augment_with_teacher.py --provider deepseek \
    --model DeepSeek-V4-Pro --target 50          # 更強更貴
python scripts/03_augment_with_teacher.py --provider deepseek \
    --model glm-5 --target 50                    # 智譜 GLM-5
```

#### B. Gemini 2.0 Flash（免費備案）

```bash
pip install google-generativeai
export GEMINI_API_KEY=AIza...

python scripts/03_augment_with_teacher.py --provider gemini --target 50
```

#### C. ModelScope（中國境內）

```bash
pip install openai
export MODELSCOPE_API_KEY=...

python scripts/03_augment_with_teacher.py --provider modelscope --target 50 \
    --model deepseek-ai/DeepSeek-V4-Flash
```

#### D. OpenAI / Anthropic

```bash
export OPENAI_API_KEY=sk-...
python scripts/03_augment_with_teacher.py --provider openai --target 50

export ANTHROPIC_API_KEY=sk-ant-...
python scripts/03_augment_with_teacher.py --provider anthropic --target 50
```

#### E. BYO-CLI：不把 API key 放進 script

如果你已經在本機 CLI（例如 commandcode / opencode / codex）設定好 DeepSeek，
可以讓 script 只負責「印 prompt」和「解析回應」，完全不碰 API key。

```bash
# 1. 產生每個 tool 一份 prompt 檔
python scripts/03_augment_with_teacher.py --target 50 \
    --print-prompts data/synth/prompts

# 2. 用自己的 CLI 跑 prompt
# 產生後會有 data/synth/prompts/_run_with_cli.sh，可先改裡面的 commandcode 指令
bash data/synth/prompts/_run_with_cli.sh

# 3. 解析 CLI 回應，輸出可 merge 的 jsonl
python scripts/03_augment_with_teacher.py \
    --parse-responses data/synth/responses
```

檔名規則：
- prompt 會是 `data/synth/prompts/<tool>.prompt.txt`
- response 請存成 `data/synth/responses/<tool>.txt`
- parser 會讀 response 內的 JSONL，過濾重複、錯 tool label、跨 tool keyword bleed
- 輸出會是 `data/synth/external_cli_augmented_<timestamp>.jsonl`

如果一個 tool 需要超過 `--max-batch`（預設 30）筆，prompt 只會先要求一批。
對同一個 prompt 多跑幾次，把回應合併到同一個 `<tool>.txt` 後再 parse。

#### 共通選項

只針對某些冷門 tool 補：
```bash
python scripts/03_augment_with_teacher.py --provider deepseek --target 50 \
    --only-tools record_strategy correlate_algorithms map_energy_semantics \
                 confirm_strategy_adoption compare_actual_predicted
```

跳過某些已經夠多的 tool：
```bash
python scripts/03_augment_with_teacher.py --provider deepseek --target 50 \
    --skip-tools query_energy_records __refusal__
```

輸出會寫到 `data/synth/<provider>_augmented_<timestamp>.jsonl`，**不會自動併入 train**。

### Step 3.5 — 抽查（10 分鐘）

打開 `data/synth/<provider>_augmented_*.jsonl`，隨機看 30–50 行：
- 問題真的對應到 expected_tool 嗎？
- 跟其他 tool 有沒有歧義？
- 有沒有不通順、有錯字、太短的？

把不要的行直接從檔案刪掉。剩下的就是要併入訓練的。

### Step 4 — 併入 train

```bash
python scripts/04_merge_and_rebuild_manifest.py \
    --source data/synth/gemini_augmented_<timestamp>.jsonl
```

它會：
1. 備份 `train.jsonl` + `manifest.json` 到 `data/_backup/`
2. 過濾掉跟 train/val/smoke 重複的 user 字串
3. 把通過的 row 補上完整 schema（system / user / assistant + sample_id 等）
4. append 到 `train.jsonl`
5. 重新生 `manifest.json`（version → 0.3）

跑完再 `python scripts/05_check_distribution.py` 確認分布。

### Step 5 — 重訓

回 Colab notebook：
1. 重啟 runtime（避免舊環境殘留）
2. Cell 1 → Cell 5 都跑
3. **跳過** Cell `e92027c7`（其實就是執行 `01_apply...`，已經跑過了不用再跑）
4. **跳過** distribution check cell（已在 step 0 / step 4 結尾跑過）
5. Cell 6 → Cell 7 → Cell 8 → Cell 9 → Cell 10 → Cell 11
6. 看 Cell 11 的 `Eval report:` accuracy

預期 val accuracy 從 23.7% 跳到 **60–80%**。

---

## 核心設計原則

1. **`00_config.py` 是唯一真理**：所有腳本都從這裡讀 tool 清單跟 system prompt。
   想加 tool？只改 `00_config.TOOLS`，然後重跑 `01_apply` + `04_merge`。
   不要在 notebook cell 裡手寫清單（之前就這樣搞到 13 vs 28 不一致）。

2. **絕對不動 val.jsonl 跟 smoke.jsonl 的內容**：
   `04_merge` 只會 append 進 `train.jsonl`。val/smoke 必須維持人工策劃，
   否則 eval 數字會自欺欺人。
   （`01_apply` 會改 val/smoke 的 system 訊息，但 user/assistant 內容不動。）

3. **都先備份再改**：`01_apply` / `02_dedupe` / `04_merge` 都會把原檔複製到
   `data/_backup/<name>_<timestamp>.jsonl`。

4. **跨 tool keyword bleed 過濾**：`03_augment` 會擋掉「明明叫 query_energy_records
   出題、卻冒出『比較』『趨勢』」這種會教歪模型的 query。

---

## 常見問題

### Q: 我已經有 augmented 檔了，可以直接 merge 嗎？

A: 可以。`04_merge` 接受任何 `{"user": ..., "expected_tool": ...}` 格式的 jsonl。

### Q: tool 清單之後改了會怎樣？

A: 改 `00_config.TOOLS` 後：
1. 重跑 `01_apply` 把新清單套到所有 jsonl 的 system
2. 既有 train 樣本如果 `expected_tool` 不在新清單，`05_check_distribution.py` 會警告
3. 重訓

### Q: augment 太慢 / API 太貴？

A: 把 `--target` 從 50 降到 30，或只挑 < 10 樣本的 tool 用 `--only-tools`。

### Q: 為什麼不直接生 1500 筆？

A: Teacher LLM 會重複，超過大概 50 筆會明顯退步（同義句反覆）。
50 筆通常已經夠單個 tool 達 80% accuracy 了。

---

## 排程預估（A100 40G）

| 步驟 | 預估時間 |
|---|---|
| 01 apply system | 5 秒 |
| 02 dedupe | 5 秒 |
| 03 augment（28 tool × ~30 筆 × Gemini Flash） | 10–20 分鐘 |
| 03.5 人工抽查 | 10 分鐘 |
| 04 merge + manifest | 5 秒 |
| 05 check | 1 秒 |
| 重訓（4 epoch、總共約 ~700 筆） | 5–8 分鐘 |
| Cell 10 + 11 評測 | 1–2 分鐘 |

整輪約 **30–40 分鐘**，預期出 60% 以上的 val accuracy。

---

## 驗收門檻

這輪不是只看 train loss。以 `outputs/gemma_router_strict_v02/eval/val_after_train_summary.json`
為準，至少要同時滿足：

| 指標 | 最低門檻 | 理想值 | 說明 |
|---|---:|---:|---|
| val tool accuracy | 60% | 80%+ | 先確認 router 真的脫離 21–24% 卡點 |
| malformed JSON rate | < 5% | 0% | 不能輸出 markdown、自然語言或壞 JSON |
| smoke accuracy | 80% | 95%+ | 基本行為不能壞掉 |
| `__refusal__` 誤判率 | < 20% | < 10% | 不要又變成 refusal-loving |
| 冷門 tool 命中 | 至少有改善 | 單 tool 50%+ | 尤其是原本 train < 10 的 tool |

如果 `val accuracy >= 60%` 但還沒到 80%，這輪仍算成功，因為已證明主因是資料分布與 system prompt。
下一輪再針對錯誤 tool 做 hard negative / contrastive 補強。

---

## 跑完後怎麼判斷

### Case A — 直接成功（80%+）

1. 保留目前 `data/`、`scripts/`、`outputs/gemma_router_strict_v02/adapter/`。
2. 如果要部署，再設 `EXPORT_GGUF=true` 重跑或只跑 export 段，把 GGUF 放到 `outputs/.../final_gguf/`。
3. 把 `val_after_train_summary.json` 和 `val_after_train.jsonl` 留作 v0.3 baseline。
4. 下一版才考慮補 arguments 品質；這版先不要同時擴大任務範圍。

### Case B — 有突破但未達標（60–79%）

先不要改訓練參數，先看錯誤分布：

```bash
python scripts/05_check_distribution.py
```

再打開：

```bash
outputs/gemma_router_strict_v02/eval/val_after_train.jsonl
```

把錯誤分成四類：

| 類型 | 現象 | 下一步 |
|---|---|---|
| A. tool 混淆 | predicted 是另一個相近 tool | 補 hard negative：同一句型改不同關鍵條件 |
| B. 冷門 tool 不會 | 該 tool train 仍 < 30 或錯很多 | 對該 tool 再 `--only-tools` 補到 50 |
| C. refusal 過多 | 能回答卻預測 `__refusal__` | 降低 refusal 比例；補「看似模糊但可路由」樣本 |
| D. JSON 壞掉 | `__parse_error__` 或多餘文字 | 降 epoch 或檢查 response-only mask |

優先處理 A / B，因為它們通常最影響 accuracy。

### Case C — 仍低於 60%

先不要繼續盲目 augment。照這個順序查：

1. 看訓練輸出的 `[mask-sanity]`，visible loss token ratio 必須大約落在 5–25%。
2. 看 `[infer-render]`，inference prompt 應該跟訓練 template 一致，且以 `<|turn>model\n` 收尾。
3. 確認 `01_apply_tool_list_system.py` 真的已套到 train/val/smoke 的 system 訊息。
4. 確認 `04_merge` 後 `train` 的冷門 tool 不再是 1–5 筆。
5. 若上述都正常，再把 `NUM_TRAIN_EPOCHS` 從 3 改 4；不要直接衝 8。

低於 60% 通常代表 template/masking 還有問題，不是資料量不夠。

---

## 第二輪補強 PLAN

只有在第一輪跑完、看過 `val_after_train.jsonl` 後才做。

### 1. 做錯誤矩陣

從 eval jsonl 抽出 `(expected_tool, predicted_tool)`，找 top confusion pair。
例如：

| expected | predicted | 可能原因 |
|---|---|---|
| compare_actual_predicted | compare_energy_usage | 「比較」太泛，沒強調 actual vs predicted |
| record_strategy | confirm_strategy_adoption | 「記錄」和「確認已採用」語意太近 |
| map_energy_semantics | search_docs | 「這棟是哪個 ID」被誤解成查文件 |

每個 confusion pair 補 10–20 筆 contrastive 樣本，不要平均亂補。

### 2. 補 hard negative，而不是只補同義句

好的補強樣本應該長這樣：

| 目的 | user 寫法 |
|---|---|
| 區分 actual/predicted vs 年度比較 | 「把 2024 實際用電和模型預測差多少列出來」 |
| 區分 record vs confirm | 「幫我把總圖二館空調調高 1 度這個策略記錄下來」 |
| 區分 confirm vs status | 「確認總圖二館空調調高 1 度是否已被採用」 |
| 區分 semantic map vs docs | 「使用者說的總圖二館要對應到哪個 building_id？」 |

不要只寫「請查詢 X」、「幫我比較 X」這種短句，模型會學不到邊界。

### 3. 控制 refusal 比例

`__refusal__` 不需要補到 50。它是 safety fallback，不是主業務 tool。

建議：
- train refusal 佔比 < 15%
- refusal query 要明確越界或真的缺資訊
- 不要把「缺年份但可追問」全部教成 refusal，否則 router 會過度保守

### 4. 每輪只改一件主因

不要同一輪同時改：
- system prompt
- chat template
- LoRA rank
- epoch
- 大量新資料

每輪最多改一個主因，否則 eval 變好也不知道原因。

---

## 最小可執行清單

如果只想照抄跑一次，用這組：

```bash
cd /content/drive/MyDrive/energy_lora_router_v02

python scripts/05_check_distribution.py
python scripts/01_apply_tool_list_system.py
python scripts/02_dedupe_train.py

pip install openai
export YUHUAN_API_KEY=...
python scripts/03_augment_with_teacher.py --provider deepseek --target 50

# 人工抽查 data/synth/deepseek_augmented_*.jsonl 後再 merge
python scripts/04_merge_and_rebuild_manifest.py \
    --source data/synth/deepseek_augmented_<timestamp>.jsonl

python scripts/05_check_distribution.py
python colab_train_router_strict_lora.py
```

Colab 重訓時建議先用預設：

```bash
NUM_TRAIN_EPOCHS=3
TRAIN_BATCH_SIZE=8
GRAD_ACCUM_STEPS=2
LEARNING_RATE=2e-4
EXPORT_GGUF=false
```

等 val 過關後再 export，不要第一次就花時間產 GGUF。

---

## 最終交付物

這個 PLAN 跑完後，應該留下：

| 檔案 / 目錄 | 用途 |
|---|---|
| `data/harness_v02_train.jsonl` | 已套 canonical system prompt、去重、併入 synth 的 train |
| `data/harness_v02_manifest.json` | v0.3 manifest 與資料分布紀錄 |
| `data/_backup/` | 每次修改前的備份 |
| `data/synth/*_augmented_*.jsonl` | teacher 產生、人工抽查後的原始補強資料 |
| `outputs/gemma_router_strict_v02/adapter/` | LoRA adapter |
| `outputs/gemma_router_strict_v02/eval/val_after_train_summary.json` | 主要驗收數字 |
| `outputs/gemma_router_strict_v02/eval/val_after_train.jsonl` | 錯誤分析用逐筆結果 |
| `outputs/gemma_router_strict_v02/final_gguf/` | 過關後才產的部署用 GGUF |

---

## v0.4 / v0.5 才做的事

這輪 v0.3 只解 router tool selection。以下先不要混進來：

1. arguments schema 精修與欄位抽取準確率。
2. 多工具 chaining。
3. RAG 文件回答品質。
4. 真實 tool execution result grounding。
5. 部署端 latency / quantization 比較。

等 v0.3 的 tool accuracy 穩定到 80%+，再開 v0.4；否則問題會混在一起，很難 debug。

---

## v0.4：主動異常告警 Agent

這類不是「使用者問問題 → tool」的互動式 router，而是事件驅動流程：

```text
IoT / RTEM / BMS 資料
→ 定時掃描器
→ 異常偵測 / 分類 / 診斷
→ 建立 alert event
→ 通知人類管理者
→ 人類追問「這是什麼錯？我該怎麼處理？」
→ agent 給診斷、證據、決策建議
→ 人類 acknowledge / close / false positive / 建工單
```

D 槽 demo 目前已新增最小工具層，位置：

```text
D:\idf優化\demo\src\proactive_alerts.py
D:\idf優化\demo\src\knowledge_mcp_server.py
D:\idf優化\demo\data\proactive_alert_training\
```

目前掛到 MCP 的工具：

| tool | 用途 |
|---|---|
| `scan_iot_snapshot_for_alerts` | 掃描 IoT/RTEM/BMS snapshot，偵測候選異常並可建立 alert |
| `create_energy_alert` | 建立持久化能源異常告警事件 |
| `list_active_energy_alerts` | 列出 open / acknowledged 告警供 dashboard 或人員決策 |
| `acknowledge_energy_alert` | 人員確認已收到告警 |
| `close_energy_alert` | 告警處理完成或標為誤報 |
| `notify_energy_manager` | 將通知寫入 outbox，後續可接 Email / LINE / Teams |
| `recommend_anomaly_decision` | 根據 alert 建議是否通知、開工單、監控或降級 |

目前實作是安全版：
- 不直接控制設備。
- 不直接發真正外部通知，只寫 `outputs/energy_manager/notification_outbox.jsonl`。
- alert 寫到 `outputs/energy_manager/alerts.jsonl`。
- 真正 Email / LINE / Teams sender 之後再接。

### v0.4 要補的訓練資料

v0.3 是：

```text
使用者問題 → tool-call JSON
```

v0.4 要多補：

```text
系統事件 / alert / scan result → tool-call JSON
人類追問異常 → 結構化決策回答
```

已放 seed：

```text
D:\idf優化\demo\data\proactive_alert_training\router_seed.jsonl
D:\idf優化\demo\data\proactive_alert_training\response_authoring_prompts.jsonl
```

給其他 agent 擴寫回答時，固定要求包含：

1. 異常類型
2. 嚴重度
3. 證據
4. 可能原因
5. 建議處置
6. 是否通知 / 是否開工單
7. 下一個建議工具

禁止：
- 不要捏造不存在的數值。
- 不要說已經派人或已經控制設備，除非 user 明確說已完成。
- 不要自動改控制參數；只能建議人工確認、通知或建工單。

---

## v0.5：ASHRAE / 台灣法規 / 演算法交叉決策

ASHRAE 值得補，但定位要正確：它多數是 engineering standard / guideline，不應直接標成 LAW。台灣法規與校方規範仍是合規優先。

決策建議應該走三層交叉：

```text
1. 現場資料：IoT / RTEM / BMS / meter trend
2. 演算法結果：anomaly、OpenBSE、PI-VD、portfolio ROI、M&V
3. 標準與法規：台灣綠建築 / 建築技術規則 / ASHRAE / IPMVP
```

優先補的 ASHRAE / guideline 類型：

| 類型 | 用途 |
|---|---|
| ASHRAE 90.1 | 建築能源效率與設備控制基準 |
| ASHRAE 55 | 熱舒適與室內溫度判斷 |
| ASHRAE 62.1 | 通風與 IAQ 判斷 |
| ASHRAE 135 | BACnet / BMS 通訊背景 |
| ASHRAE Guideline 14 | M&V、節能量驗證、baseline 判斷 |
| ASHRAE 202 | Commissioning / 調適與驗收流程 |
| ASHRAE TC 9.9 | Data center thermal guideline，接資料中心 demo 時使用 |

RAG metadata 建議：

```json
{
  "source": "ASHRAE Guideline 14",
  "type": "engineering_standard",
  "jurisdiction": "international",
  "mandatory": false,
  "topic": "measurement_and_verification",
  "applies_to": ["baseline", "energy_savings", "M&V"],
  "text": "..."
}
```

台灣法規 metadata：

```json
{
  "source": "建築技術規則綠建築專章",
  "type": "local_regulation",
  "jurisdiction": "TW",
  "mandatory": true,
  "topic": "green_building",
  "text": "..."
}
```

決策時的優先級：

1. 安全 / 法規強制要求
2. 台灣本地法規與校方規範
3. 合約 / 招標 / 驗收條件
4. ASHRAE / ISO / IPMVP 等工程標準
5. 演算法建議與最佳化結果

標準型回答範例：

```text
演算法判斷：疑似 step_change，高嚴重度，已持續 45 分鐘。
現場證據：空調主電表由 180kW 跳到 310kW 並維持。
工程判斷：可能是排程覆寫、AHU 同時啟動、冰水主機提前開機或控制設定變更。
標準參照：依 ASHRAE Guideline 14 的 M&V 精神，需先確認基準線與操作條件是否一致，不應直接宣稱節能或故障。
建議處置：先查 BMS 排程與 setpoint，再查 AHU / CH / Pump 同時段啟停紀錄；若下一輪掃描仍持續，通知維運並建立工單。
```

---

## v0.5：台灣節能改善案例 / 設備清冊 / ESG 報告模板

使用者手上若有 20 份以內的台灣中小企業節能改善 PDF、政府公文格式、設備 Excel，
這批資料很適合放進 RAG / report generation knowledge base，但**不要混進 v0.3 router 訓練**。

它的用途不是教模型選 tool，而是支援：

```text
節能診斷報告
改善建議書
ESCO / ROI 評估
政府補助型節能改善申請格式
ESG 報告書中的節能成果揭露
M&V 節能量驗證報告
設備汰換前後比較
```

### 建議資料位置

先放原始檔：

```text
D:\idf優化\demo\data\energy_report_knowledge\raw\
```

後續解析後可放：

```text
D:\idf優化\demo\data\energy_report_knowledge\processed\
D:\idf優化\demo\data\energy_report_knowledge\chunks\
D:\idf優化\demo\data\energy_report_knowledge\templates\
```

如果要整合現有知識庫，也可以 index 到：

```text
D:\idf優化\demo\data\knowledge_workbench\
```

### 文件分類

匯入時先標 `doc_type`：

| doc_type | 說明 |
|---|---|
| `case_study` | 中小企業節能改善案例 |
| `equipment_inventory` | 設備清冊 / 馬達 / 空調 / 照明 / 空壓機 |
| `government_form` | 政府公文、申請表、查核表 |
| `energy_audit_report` | 能源診斷報告 |
| `esg_template` | ESG / 永續報告揭露格式 |
| `mv_report` | M&V / 節能量驗證 |

### PDF chunk metadata

```json
{
  "source": "某中小企業節能改善報告.pdf",
  "doc_type": "energy_audit_report",
  "sector": "SME",
  "jurisdiction": "TW",
  "report_use": ["energy_saving_report", "esg_report", "subsidy_application"],
  "equipment_types": ["chiller", "air_compressor", "lighting"],
  "contains_numbers": true,
  "is_template": false
}
```

### Excel 設備清冊 metadata

```json
{
  "source": "設備清冊.xlsx",
  "doc_type": "equipment_inventory",
  "equipment_types": ["motor", "pump", "air_compressor", "lighting"],
  "fields": ["設備名稱", "數量", "額定功率", "運轉時數", "改善前耗能", "改善後耗能"],
  "jurisdiction": "TW"
}
```

### 未來可新增 tool

先不用急著加到 v0.3 router。v0.5 可新增：

| tool | 用途 |
|---|---|
| `search_energy_report_templates` | 查節能報告、公文、ESG 範本 |
| `extract_equipment_inventory` | 從 Excel 抽設備清冊與欄位 |
| `calculate_saving_from_equipment_table` | 用設備表估算節電量、費用、碳排 |
| `generate_energy_saving_report` | 產節能診斷 / 改善建議報告 |
| `generate_esg_energy_section` | 產 ESG 報告中的節能成果揭露段落 |
| `generate_subsidy_application_draft` | 產政府補助 / 改善申請草稿 |

如果暫時不加新 tool，可先用現有 `search_docs` 查範本；但長期建議把報告生成獨立成 tool，
因為「查文件」和「生成正式報告」是不同任務。

### 節能報告建議章節

```text
一、案場基本資料
二、能源使用現況
三、主要耗能設備
四、異常或高耗能問題
五、改善方案
六、節能量估算
七、投資成本與回收年限
八、碳排減量估算
九、M&V 驗證方式
十、ESG 揭露建議文字
十一、附件：設備清冊 / 計算表
```

### ESG 節能成果段落範例

```text
本年度本公司針對空調系統、照明設備及空壓系統進行節能改善，
包含高效率設備汰換、運轉排程優化及空壓洩漏改善。經改善前後用電資料比對，
年度節電量約 XX kWh，約當減碳 XX tCO2e。相關節能成效依據改善前基準線、
設備運轉時數與改善後實測資料估算，後續將持續透過能源管理系統追蹤成效。
```

### 重要原則

1. PDF / Excel 進 RAG，不進 router v0.3 訓練。
2. Excel 中的設備數值可用來計算，但報告必須標示「估算依據」。
3. ESG 段落不可捏造節電量、碳排係數或投資金額。
4. 政府公文格式只作格式參考；是否符合最新申請規定仍需人工確認。
5. ASHRAE / 台灣法規 / ESG 格式應在最終建議中分開標示，不要混成同一種「法律」。

### 未來整合流程

```text
PDF / Excel 原始檔
→ parse / OCR / Excel schema extraction
→ chunk + metadata
→ index 到 knowledge_workbench 或 energy_report_knowledge
→ search_energy_report_templates
→ generate_energy_saving_report / generate_esg_energy_section
→ 人工審核
→ save_curated_trace / 報告輸出
```
