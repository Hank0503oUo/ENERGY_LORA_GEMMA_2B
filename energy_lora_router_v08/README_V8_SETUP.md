# energy_lora_router_v08 — Boundary Repair Round

## Status: READY TO TRAIN

V7 把模型從「不會 JSON」拉到「會 JSON 但搞混邊界」。V8 修邊界。

## V7 診斷 → V8 修正

| V7 問題 | V8 修正 |
|---------|---------|
| `workflow_chain → clarify_needed`: 30 次 | 決策判準表：有建築名稱 → workflow，不是 clarify |
| `single_tool → clarify_needed`: 12 次 | 負面樣本：有建築+單一意圖 → single_tool |
| `no_evidence`: 0/21 全錯 | no_evidence vs clarify 邊界表 + 更多 no_evidence 樣本 |
| parse error 仍 20.4% | format smoke gate + JSON grammar |
| `document_search_dci` 被當成 clarify | 文件搜尋明確標為 workflow_chain |

## System prompt 關鍵改動

### 決策判準表（按順序判斷）

```
Step 1: 危險/超出範圍？ → refusal
Step 2: 完全模糊無法判斷？ → clarify_needed
Step 3: 單一工具可答？ → single_tool
Step 4: 多步驟/策略/文件搜尋？ → workflow_chain
Step 5: 建築+年份都有但年份不在範圍？ → no_evidence
```

### 關鍵規則：何時 NOT clarify_needed

- 有建築名稱 → single_tool 或 workflow_chain
- 問策略/改善/規劃 → workflow_chain
- 問文件/法規/定義 → workflow_chain(document_search_dci)
- 有年份但不在範圍 → no_evidence

## 資料

| Split | Count | 說明 |
|-------|-------|------|
| train | 1109 | 1031 reprompted + 78 boundary repair |
| val | 147 | reprompted |
| smoke | 16 | reprompted |
| format_smoke | 16 | schema-aware |

## V8 驗收門檻

| 指標 | V7 實際 | V8 目標 |
|------|---------|---------|
| Parse error rate | 20.4% | < 10% |
| overall accuracy | 24.5% | > 60% |
| clarify_needed precision | 過度觸發 | 不再搶 workflow |
| no_evidence accuracy | 0% | > 80% |
| workflow_chain accuracy | 12.5% | > 70% |
| single_tool accuracy | 29% | > 75% |
| answerability accuracy | 29.9% | > 70% |

## 參考

- V8 plan: `D:\idf優化\demo\docs\V7_TO_V8_IMPROVEMENT_PLAN.md`
- V7: `G:\我的雲端硬碟\energy_lora_router_v07`
