"""V8 Dispatch Training config — boundary repair round.

V7 problem: clarify_needed over-triggered (30 workflow→clarify, 12 single→clarify).
V8 fix:
  - Decision rubric with explicit thresholds
  - clarify_needed ONLY when user input genuinely lacks required args
  - Negative examples: tasks with enough context must NOT become clarify_needed
  - no_evidence vs clarify_needed boundary explicitly defined
  - workflow_chain vs single_tool decision rubric
"""
from __future__ import annotations
from pathlib import Path

DRIVE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = DRIVE_ROOT / "data" / "processed"
TRAIN_FILE = DATA_DIR / "train_v08_dispatch.jsonl"
VAL_FILE = DATA_DIR / "val_v08_dispatch.jsonl"
SMOKE_FILE = DATA_DIR / "smoke_v08_dispatch.jsonl"
FORMAT_SMOKE_FILE = DATA_DIR / "format_smoke_v08.jsonl"
MANIFEST_FILE = DATA_DIR / "harness_v08_manifest.json"

VALID_DISPATCH_TYPES = frozenset({
    "single_tool", "workflow_chain", "clarify_needed", "no_evidence", "refusal",
})
VALID_WORKFLOW_IDS = frozenset({
    "single_building_year_status", "building_hotspot_improvement",
    "campus_top_energy_buildings", "campus_year_compare",
    "building_strategy_plan", "counterfactual_saving_estimate",
    "document_search_dci", "none",
})
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
唯一輸出：一個 JSON object。禁止中文句子、解釋、markdown。

## dispatch_type 判準表（按順序判斷）

Step 1: 使用者是否要求危險/不安全/超出能源範圍？
  → YES → refusal（dispatch_type=refusal, workflow_id=none, required_tools=[]）
  超出範圍：瓦斯、水費、停車、校外、醫院、天然氣、碳足跡
  不安全：偽造、竄改、隱藏、刪除、強制控制、全開設備

Step 2: 使用者是否完全沒有指定建築/年份/指標，且語意過於模糊？
  「然後呢」「幫我看一下」「節能方法」這類完全無法判斷意圖的問句
  → YES → clarify_needed（dispatch_type=clarify_needed, workflow_id=none, required_tools=[]）
  注意：如果有提到建築名稱或年份，即使問題不完整也應走向 single_tool 或 workflow_chain

Step 3: 使用者問的是否是單一工具就能回答的問題？
  只查一棟建築某年用電、全校概況、排行榜、偵測異常、推薦策略（不需要多步驟）
  → YES → single_tool（workflow_id=任務模式, required_tools=[1個工具]）

Step 4: 使用者是否要求多步驟分析、策略規劃、比較、文件搜尋？
  「用電狀況跟改善建議」「節能策略規劃」「如果調高溫度省多少」「跨年比較」
  「XX文件」「CV-RMSE 定義」
  → YES → workflow_chain（workflow_id=任務模式, required_tools=[2-4個工具]）

Step 5: 使用者給了完整資訊（建築+年份），但年份不在 2017-2023 範圍？
  → YES → no_evidence（workflow_id=none, required_tools=[]）

## 關鍵規則：何時 NOT clarify_needed

以下情況不得標為 clarify_needed：
- 有建築名稱的問句（即使沒有年份）→ single_tool 或 workflow_chain
- 問策略/改善/規劃的問句（即使模糊）→ workflow_chain
- 問文件/法規/定義的問句 → workflow_chain(document_search_dci)
- 問假設情境的問句 → workflow_chain(counterfactual_saving_estimate)
- 有年份但年份不在範圍 → no_evidence

只有這些才是 clarify_needed：
- 「然後呢」「節能方法」（完全沒有方向）
- 「那棟建築的用電」（沒有指明哪棟）
- 「幫我查一下」（查什麼？）
- 「去年的用電」（哪棟建築？）

## clarify_needed vs no_evidence

| 情境 | dispatch_type | 原因 |
|------|---------------|------|
| 「然後呢」 | clarify_needed | 完全無法判斷意圖 |
| 「去年的用電」 | clarify_needed | 缺建築名稱 |
| 「那棟建築的用電」 | clarify_needed | 指代不明 |
| 「博理館 2015 用電」 | no_evidence | 建築+年份都有，但年份不在範圍 |
| 「明達館 2010 EUI」 | no_evidence | 建築+年份都有，但年份不在範圍 |

## single_tool vs workflow_chain

| 問句 | dispatch_type | 原因 |
|------|---------------|------|
| 博理館 2020 用電 | single_tool | 一個工具就夠 |
| 幫博理館推薦節能策略 | single_tool | recommend_adaptive_strategies 一個就夠 |
| 博理館 2020 用電狀況跟改善 | workflow_chain | 查用電+偵測異常+策略，需多工具 |
| 博理館節能策略規劃 | workflow_chain | 查用電+排名+策略，需多工具 |
| 如果博理館空調調高2度省多少 | workflow_chain | 基準+模擬，需多工具 |
| CV-RMSE 定義在哪份文件 | workflow_chain | 文件搜尋是多步驟 |

## workflow_id 選擇

| 問句類型 | workflow_id |
|----------|-------------|
| 查單一建築某年用電 | single_building_year_status |
| 建築熱點+改善 | building_hotspot_improvement |
| 校園排行 | campus_top_energy_buildings |
| 跨年比較 | campus_year_compare |
| 建築節能策略規劃 | building_strategy_plan |
| 如果...省多少 | counterfactual_saving_estimate |
| 文件/法規/定義搜尋 | document_search_dci |
| clarify/no_evidence/refusal | none |

## 合法工具（只能放在 required_tools[].tool）

Core: query_energy_records, list_campus_stats, get_top_energy_buildings, detect_energy_anomalies, run_openbse_hybrid_counterfactual, openbse_hvac_breakdown, recommend_adaptive_strategies
DCI: search_docs, find_docs, grep_docs, read_doc_chunk, inspect_doc_context, count_doc_matches

## JSON Schema

{
  "dispatch_type": "<single_tool|workflow_chain|clarify_needed|no_evidence|refusal>",
  "workflow_id": "<任務模式或none>",
  "answerability": "<見合法值>",
  "locked_entities": {"building_names": [], "years": [], "metrics": []},
  "required_tools": [{"tool": "<工具名>", "purpose": "<用途>"}],
  "stop_conditions": ["<條件>"]
}

answerability 合法值: answerable_single_tool, answerable_multi_tool, missing_required_arguments, ambiguous_reference, unsupported_scope, unsupported_capability, unsafe_operation, no_evidence_expected"""


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
        "dispatch_type": {"type": "string", "enum": ["single_tool", "workflow_chain", "clarify_needed", "no_evidence", "refusal"]},
        "workflow_id": {"type": "string", "enum": ["single_building_year_status", "building_hotspot_improvement", "campus_top_energy_buildings", "campus_year_compare", "building_strategy_plan", "counterfactual_saving_estimate", "document_search_dci", "none"]},
        "answerability": {"type": "string", "enum": ["answerable_single_tool", "answerable_multi_tool", "missing_required_arguments", "ambiguous_reference", "unsupported_scope", "unsupported_capability", "unsafe_operation", "no_evidence_expected"]},
        "locked_entities": {"type": "object", "properties": {"building_names": {"type": "array", "items": {"type": "string"}}, "years": {"type": "array", "items": {"type": "string"}}, "metrics": {"type": "array", "items": {"type": "string"}}}, "required": ["building_names", "years", "metrics"]},
        "required_tools": {"type": "array", "items": {"type": "object", "properties": {"tool": {"type": "string"}, "purpose": {"type": "string"}}, "required": ["tool", "purpose"]}},
        "stop_conditions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["dispatch_type", "workflow_id", "answerability", "locked_entities", "required_tools", "stop_conditions"],
}


if __name__ == "__main__":
    print(f"Prompt length: {len(render_system_prompt())} chars")
