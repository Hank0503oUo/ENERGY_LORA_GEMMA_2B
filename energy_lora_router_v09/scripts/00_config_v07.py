"""V7 Dispatch Training config — schema clarity round.

Three-layer architecture:
  Layer 1: dispatch_type  — 5 enums only (never a tool name)
  Layer 2: workflow_id    — abstract task names (never a tool name)
  Layer 3: required_tools — tool names go HERE only

Key fixes from v06:
  - no_evidence: required_tools MUST be []
  - clarify_needed / refusal: workflow_id MUST be "none"
  - recommend_adaptive_strategies is a TOOL, not dispatch_type or workflow_id
  - document_search_dci is a WORKFLOW, not dispatch_type
  - JSON grammar for constrained decoding provided
"""
from __future__ import annotations
from pathlib import Path

DRIVE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = DRIVE_ROOT / "data" / "processed"
TRAIN_FILE = DATA_DIR / "train_v07_dispatch.jsonl"
VAL_FILE = DATA_DIR / "val_v07_dispatch.jsonl"
SMOKE_FILE = DATA_DIR / "smoke_v07_dispatch.jsonl"
FORMAT_SMOKE_FILE = DATA_DIR / "format_smoke_v07.jsonl"
MANIFEST_FILE = DATA_DIR / "harness_v07_manifest.json"

# ── Layer 1: dispatch_type (WHAT kind of decision) ──
VALID_DISPATCH_TYPES = frozenset({
    "single_tool", "workflow_chain", "clarify_needed", "no_evidence", "refusal",
})

# ── Layer 2: workflow_id (WHICH task pattern) ──
VALID_WORKFLOW_IDS = frozenset({
    "single_building_year_status", "building_hotspot_improvement",
    "campus_top_energy_buildings", "campus_year_compare",
    "building_strategy_plan", "counterfactual_saving_estimate",
    "document_search_dci", "none",
})

# ── Layer 3: tool names (in required_tools[].tool ONLY) ──
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

VALID_ANSWERABILITY = frozenset({
    "answerable_single_tool", "answerable_multi_tool",
    "missing_required_arguments", "ambiguous_reference",
    "unsupported_scope", "unsupported_capability",
    "unsafe_operation", "no_evidence_expected",
})

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
    return """你是 dispatch classifier。你只負責分類，不負責回答。
你的唯一輸出是一個 JSON object。禁止輸出中文句子、解釋、markdown。

## 三層架構（不可混淆）

### 第 1 層：dispatch_type — 決策種類（只能選一個）
- single_tool：只需一個工具就能回答
- workflow_chain：需要兩個以上工具串接
- clarify_needed：資訊不足，需要使用者補充
- no_evidence：工具理論上可答但查不到資料
- refusal：超出支援範圍或不安全
dispatch_type 永遠不會是工具名稱。

### 第 2 層：workflow_id — 任務模式（只能選一個）
- single_building_year_status：查單一建築某年狀態
- building_hotspot_improvement：找建築熱點 + 改善
- campus_top_energy_buildings：校園排行
- campus_year_compare：跨年比較
- building_strategy_plan：建築節能策略規劃
- counterfactual_saving_estimate：如果...則省多少
- document_search_dci：文件搜尋
- none：用於 clarify_needed / no_evidence / refusal
workflow_id 永遠不會是工具名稱。

### 第 3 層：required_tools[].tool — 工具名稱
工具名稱只能出現在這裡，不能出現在 dispatch_type 或 workflow_id。

Core 工具：
1. query_energy_records 2. list_campus_stats 3. get_top_energy_buildings
4. detect_energy_anomalies 5. run_openbse_hybrid_counterfactual
6. openbse_hvac_breakdown 7. recommend_adaptive_strategies

DCI 工具（用於 document_search_dci workflow）：
8. search_docs 9. find_docs 10. grep_docs
11. read_doc_chunk 12. inspect_doc_context 13. count_doc_matches

## 分派規則表

| dispatch_type  | workflow_id                                         | required_tools      |
|----------------|-----------------------------------------------------|---------------------|
| single_tool    | single_building_year_status / campus_top_energy_buildings / none | 1 個工具            |
| workflow_chain | building_hotspot_improvement / building_strategy_plan / counterfactual_saving_estimate / campus_year_compare / campus_top_energy_buildings / document_search_dci / single_building_year_status | 2-4 個工具          |
| clarify_needed | none                                                | 空陣列 []           |
| no_evidence    | none                                                | 空陣列 []           |
| refusal        | none                                                | 空陣列 []           |

## 邊界規則

1. 「節能方法」「幫我看一下」→ clarify_needed, workflow_id=none, required_tools=[]
2. 「查天然氣」「交大用電」→ refusal, workflow_id=none, required_tools=[]
3. 「幫我隱藏」「偽造」→ refusal, workflow_id=none, required_tools=[]
4. 年份不在範圍(2017-2023) → no_evidence, workflow_id=none, required_tools=[]
5. document_search_dci 是 workflow，DCI 工具放在 required_tools
6. recommend_adaptive_strategies 是工具名，不是 dispatch_type 也不是 workflow_id
7. building_strategy_plan 是 workflow，dispatch_type=workflow_chain，需多個工具
8. get_top_energy_buildings 是工具名，不是 dispatch_type 也不是 workflow_id
9. detect_energy_anomalies 是工具名，不是 dispatch_type 也不是 workflow_id

## JSON Schema（必須包含全部欄位）

{
  "dispatch_type": "<上述 5 個值之一>",
  "workflow_id": "<上述 8 個值之一>",
  "answerability": "<answerable_single_tool / answerable_multi_tool / missing_required_arguments / ambiguous_reference / unsupported_scope / unsupported_capability / unsafe_operation / no_evidence_expected>",
  "locked_entities": {"building_names": [], "years": [], "metrics": []},
  "required_tools": [{"tool": "<工具名>", "purpose": "<用途>"}],
  "stop_conditions": ["<條件>"]
}"""


# ── JSON Grammar for constrained decoding (llama.cpp) ──
JSON_GRAMMAR = r"""
root ::= object
object ::= "{" ws pair-list "}"
pair-list ::= pair | pair "," ws pair-list
pair ::= key ws ":" ws value
key ::= string
value ::= string | array | object-inner | "true" | "false" | "null" | number
string ::= "\"" char-list "\""
char-list ::= [^"\\]*
array ::= "[" ws value-list? ws "]"
value-list ::= value | value "," ws value-list
object-inner ::= "{" ws pair-list "}"
number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws ::= [ \t\n]*
"""

DISPATCH_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "dispatch_type": {
            "type": "string",
            "enum": ["single_tool", "workflow_chain", "clarify_needed", "no_evidence", "refusal"],
        },
        "workflow_id": {
            "type": "string",
            "enum": [
                "single_building_year_status", "building_hotspot_improvement",
                "campus_top_energy_buildings", "campus_year_compare",
                "building_strategy_plan", "counterfactual_saving_estimate",
                "document_search_dci", "none",
            ],
        },
        "answerability": {
            "type": "string",
            "enum": [
                "answerable_single_tool", "answerable_multi_tool",
                "missing_required_arguments", "ambiguous_reference",
                "unsupported_scope", "unsupported_capability",
                "unsafe_operation", "no_evidence_expected",
            ],
        },
        "locked_entities": {
            "type": "object",
            "properties": {
                "building_names": {"type": "array", "items": {"type": "string"}},
                "years": {"type": "array", "items": {"type": "string"}},
                "metrics": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["building_names", "years", "metrics"],
        },
        "required_tools": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"tool": {"type": "string"}, "purpose": {"type": "string"}},
                "required": ["tool", "purpose"],
            },
        },
        "stop_conditions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["dispatch_type", "workflow_id", "answerability", "locked_entities", "required_tools", "stop_conditions"],
}


if __name__ == "__main__":
    prompt = render_system_prompt()
    print(f"System prompt length: {len(prompt)} chars")
    print(prompt)
