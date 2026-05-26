# energy_lora_router_v07 — Schema Clarity Round

## Status: READY TO TRAIN

V6 是成功的 recovery round（parse error 87%→32%），但不是 production round。
V7 目標：**把 schema 變得更好學**。

## V6 診斷 → V7 修正

| V6 問題 | V7 修正 |
|---------|---------|
| 模型 17 次把工具名當 dispatch_type | 三層架構表格 + 重複強調「dispatch_type 永遠不是工具名」 |
| 模型 22 次把工具名當 workflow_id | 同上 + 「workflow_id 永遠不是工具名」 |
| 120 筆 no_evidence 帶了 required_tools | 全部清為 `[]` |
| workflow_chain ↔ single_tool 邊界模糊 | 74 筆 contrast 樣本：同建築「只查」vs「查+改善」 |
| recommend_adaptive_strategies 被當成 dispatch_type | 對比樣本明確教「這是工具名」 |
| parse error 仍 32% | JSON grammar for constrained decoding |

## 資料

| Split | Count | 說明 |
|-------|-------|------|
| train | 1031 | 837 reprompted + 120 format curriculum + 74 contrast |
| val | 147 | reprompted + schema fixed |
| smoke | 16 | reprompted + schema fixed |
| format_smoke | 16 | schema-aware format test |

## 關鍵檔案

| 檔案 | 用途 |
|------|------|
| `scripts/00_config_v07.py` | 三層 schema + system prompt |
| `scripts/run_v07_pipeline.py` | 完整 pipeline |
| `scripts/train_lora.py` | 訓練 (v07) |
| `scripts/evaluate_router.py` | 評估 (v07 gates) |
| `v7_training_config.json` | 訓練參數 |
| `grammar/dispatch_schema.gbnf` | JSON grammar (llama.cpp) |
| `grammar/dispatch_json_schema.json` | JSON Schema |

## V7 驗收門檻

| 指標 | V6 實際 | V7 目標 |
|------|---------|---------|
| Parse error rate | 32% | < 15% |
| dispatch_type accuracy | ~50% | > 75% |
| single_tool accuracy | 0% | > 60% |
| workflow_chain accuracy | 7% | > 50% |
| Refusal accuracy | 80% | > 85% |
| Schema violation (tool as DT) | 17 | < 5 |

## 參考

- V7 建議來源: 使用者 feedback
- V6: `G:\我的雲端硬碟\energy_lora_router_v06`
- Recovery plan: `G:\我的雲端硬碟\energy_lora_router_v06\V6_DISPATCH_RECOVERY_PLAN.md`
