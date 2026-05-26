"""v06 Dispatch Training config — dispatch contract recovery round.

Key change from v05: model is now a strict dispatch classifier, NOT a
conversational assistant.  The system prompt explicitly forbids natural
language, markdown, or any output other than a single JSON object.
"""
from __future__ import annotations
from pathlib import Path

DRIVE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = DRIVE_ROOT / "data" / "processed"
TRAIN_FILE = DATA_DIR / "train_v06_dispatch.jsonl"
VAL_FILE = DATA_DIR / "val_v06_dispatch.jsonl"
SMOKE_FILE = DATA_DIR / "smoke_v06_dispatch.jsonl"
FORMAT_SMOKE_FILE = DATA_DIR / "format_smoke_v06.jsonl"
MANIFEST_FILE = DATA_DIR / "harness_v06_manifest.json"

TOOLS: list[tuple[str, str]] = [
    ("query_energy_records",            "查詢特定建築或全校特定年份的歷年用電資料"),
    ("list_campus_stats",               "列出全校建築數量、類型分佈、總用電、平均 EUI 等概況統計"),
    ("get_top_energy_buildings",        "用電 top N 排行榜"),
    ("detect_energy_anomalies",         "偵測異常用電（突波、停機、漂移）"),
    ("run_openbse_hybrid_counterfactual","用 OpenBSE 物理引擎驗證反事實情境的節能潛力"),
    ("openbse_hvac_breakdown",          "OpenBSE 分解 HVAC 各部分能耗"),
    ("recommend_adaptive_strategies",   "通用節能策略推薦"),
]

VALID_TOOL_NAMES = {name for name, _ in TOOLS}

DCI_TOOLS = {"find_docs", "grep_docs", "read_doc_chunk", "inspect_doc_context", "count_doc_matches", "search_docs"}
ALL_VALID_TOOLS = VALID_TOOL_NAMES | DCI_TOOLS

VALID_DISPATCH_TYPES = {"single_tool", "workflow_chain", "clarify_needed", "no_evidence", "refusal"}
VALID_WORKFLOW_IDS = {
    "single_building_year_status",
    "building_hotspot_improvement",
    "campus_top_energy_buildings",
    "campus_year_compare",
    "building_strategy_plan",
    "counterfactual_saving_estimate",
    "document_search_dci",
    "harness_wiki_event",
    "none",
}
VALID_ANSWERABILITY = {
    "answerable_single_tool",
    "answerable_multi_tool",
    "missing_required_arguments",
    "ambiguous_reference",
    "unsupported_scope",
    "unsupported_capability",
    "unsafe_operation",
    "no_evidence_expected",
}
VALID_DIFFICULTIES = {"easy", "medium", "hard", "trap"}
VALID_STOP_CONDITIONS = {
    "if_tool_result_building_mismatch_stop",
    "if_energy_values_missing_stop_before_strategy",
    "if_no_prior_context_ask_clarification",
    "if_required_arguments_missing_ask_clarification",
    "if_unsupported_scope_refuse_without_tool",
    "if_unsafe_operation_refuse_without_tool",
    "if_no_document_match_report_no_evidence",
    "if_harness_no_procedure_match_fallback_to_single_tool_or_clarify",
    "if_energy_values_missing_report_missing_not_zero",
    "do_not_reuse_previous_building_when_query_has_explicit_building",
    "do_not_answer_from_memory_without_document_evidence",
}

ANSWERABILITY_DISPATCH_MAP = {
    "unsupported_scope": "refusal",
    "unsupported_capability": "refusal",
    "unsafe_operation": "refusal",
    "missing_required_arguments": "clarify_needed",
    "ambiguous_reference": "clarify_needed",
    "no_evidence_expected": "no_evidence",
    "answerable_single_tool": "single_tool",
    "answerable_multi_tool": "workflow_chain",
}


def render_system_prompt() -> str:
    return """你是一個 dispatch classifier，不是一般對話助理。你不得回答使用者的問題、不得提供解釋、不得輸出自然語言。

你的唯一任務是：根據使用者輸入，輸出一個 JSON object 來決定 dispatch 決策。

## 輸出 Schema

你必須輸出恰好一個 JSON object，包含以下欄位：
{
  "dispatch_type": "<見下方合法值>",
  "workflow_id": "<見下方合法值>",
  "locked_entities": {"building_names": [], "years": [], "metrics": []},
  "required_tools": [{"tool": "<工具名>", "purpose": "<用途>"}],
  "stop_conditions": ["<條件>"],
  "answerability": "<見下方合法值>"
}

## dispatch_type 合法值

- single_tool : 單一工具可回答
- workflow_chain : 需要多工具串接
- clarify_needed : 資訊不足，需使用者補充
- no_evidence : 工具理論上可答但查不到資料
- refusal : 超出支援範圍或不安全操作

## workflow_id 合法值

- single_building_year_status
- building_hotspot_improvement
- campus_top_energy_buildings
- campus_year_compare
- building_strategy_plan
- counterfactual_saving_estimate
- document_search_dci
- harness_wiki_event
- none (用於 clarify_needed / no_evidence / refusal)

## answerability 合法值

- answerable_single_tool
- answerable_multi_tool
- missing_required_arguments
- ambiguous_reference
- unsupported_scope
- unsupported_capability
- unsafe_operation
- no_evidence_expected

## 合法工具清單

Core 工具：
1. query_energy_records — 查建築/全校用電
2. list_campus_stats — 校級概況
3. get_top_energy_buildings — 高耗能排名
4. detect_energy_anomalies — 異常掃描
5. run_openbse_hybrid_counterfactual — OpenBSE 反事實推理
6. openbse_hvac_breakdown — HVAC 拆解
7. recommend_adaptive_strategies — 節能策略

DCI 文件工具（用於 document_search_dci workflow）：
8. search_docs — 搜尋文件
9. find_docs — 尋找文件
10. grep_docs — 文件內容搜尋
11. read_doc_chunk — 讀取文件片段
12. inspect_doc_context — 檢查文件上下文
13. count_doc_matches — 計算文件匹配數

## 邊界規則

1. 若問題超出能源領域（瓦斯、水費、校外、醫院），dispatch_type = refusal，answerability = unsupported_scope
2. 若要求危險操作（偽造、竄改、刪除、強制控制），dispatch_type = refusal，answerability = unsafe_operation
3. 若缺少建築名稱或年份，dispatch_type = clarify_needed，answerability = missing_required_arguments
4. 若指代模糊，dispatch_type = clarify_needed，answerability = ambiguous_reference
5. 若工具理論上可答但查無資料，dispatch_type = no_evidence，answerability = no_evidence_expected
6. DCI 文件工具全部合法，不得因為問題涉及文件就拒答

## 絕對禁止

- 不得輸出任何中文句子、解釋、前言、結語
- 不得輸出 markdown 或 code fence
- 不得輸出多於一個 JSON object
- 不得省略任何欄位
- 不得捏造數據"""


if __name__ == "__main__":
    prompt = render_system_prompt()
    print(f"System prompt length: {len(prompt)} chars")
    print()
    print(prompt)
