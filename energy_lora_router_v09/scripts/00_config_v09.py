"""V9 dispatch training config - format and no_evidence repair round."""
from __future__ import annotations

from pathlib import Path

DRIVE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = DRIVE_ROOT / "data" / "processed"
TRAIN_FILE = DATA_DIR / "train_v09_dispatch.jsonl"
VAL_FILE = DATA_DIR / "val_v09_dispatch.jsonl"
SMOKE_FILE = DATA_DIR / "smoke_v09_dispatch.jsonl"
FORMAT_SMOKE_FILE = DATA_DIR / "format_smoke_v09.jsonl"
MANIFEST_FILE = DATA_DIR / "harness_v09_manifest.json"

DATA_YEAR_MIN = 2017
DATA_YEAR_MAX = 2023

VALID_DISPATCH_TYPES = frozenset({
    "single_tool",
    "workflow_chain",
    "clarify_needed",
    "no_evidence",
    "refusal",
})

VALID_WORKFLOW_IDS = frozenset({
    "single_building_year_status",
    "building_hotspot_improvement",
    "campus_top_energy_buildings",
    "campus_year_compare",
    "building_strategy_plan",
    "counterfactual_saving_estimate",
    "document_search_dci",
    "none",
})

TOOLS: list[tuple[str, str]] = [
    ("query_energy_records", "查詢建築或校園能源資料"),
    ("list_campus_stats", "列出校園能源統計概況"),
    ("get_top_energy_buildings", "取得高耗能或節能潛力建築排名"),
    ("detect_energy_anomalies", "偵測建築能源異常或熱點"),
    ("run_openbse_hybrid_counterfactual", "執行 OpenBSE 反事實節能估算"),
    ("openbse_hvac_breakdown", "分析 OpenBSE HVAC 用能拆解"),
    ("recommend_adaptive_strategies", "推薦節能改善策略"),
]

VALID_TOOL_NAMES = {name for name, _ in TOOLS}
DCI_TOOLS = {
    "search_docs",
    "find_docs",
    "grep_docs",
    "read_doc_chunk",
    "inspect_doc_context",
    "count_doc_matches",
}
ALL_VALID_TOOLS = VALID_TOOL_NAMES | DCI_TOOLS

VALID_ANSWERABILITY = frozenset({
    "answerable_single_tool",
    "answerable_multi_tool",
    "missing_required_arguments",
    "ambiguous_reference",
    "unsupported_scope",
    "unsupported_capability",
    "unsafe_operation",
    "no_evidence_expected",
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
    return f"""你是 dispatch classifier。你只負責分類，不負責回答使用者問題。

唯一輸出：一個 JSON object。
禁止輸出解釋、推理過程、markdown、條列說明或 JSON 以外文字。
JSON 必須完整閉合，不得省略欄位。

有效 dispatch_type：
single_tool, workflow_chain, clarify_needed, no_evidence, refusal

有效 workflow_id：
single_building_year_status, building_hotspot_improvement,
campus_top_energy_buildings, campus_year_compare,
building_strategy_plan, counterfactual_saving_estimate,
document_search_dci, none

有效 answerability：
answerable_single_tool, answerable_multi_tool,
missing_required_arguments, ambiguous_reference,
unsupported_scope, unsupported_capability, unsafe_operation,
no_evidence_expected

## 判斷順序

Step 1. 安全與範圍
如果使用者要求偽造、竄改、隱藏、刪除資料，或強制控制設備、全開設備，輸出 refusal。
如果問題超出本系統範圍，例如瓦斯、水費、停車、校外學校、醫院、天然氣、碳排或個人隱私用電，輸出 refusal。
refusal 必須使用 workflow_id=none, required_tools=[]。

Step 2. 已知無證據
如果使用者提供建築名稱與年份，且年份不在 {DATA_YEAR_MIN}-{DATA_YEAR_MAX}，輸出 no_evidence。
no_evidence 必須使用 workflow_id=none, answerability=no_evidence_expected, required_tools=[]。
不要把超出年份的建築用電問題送去 query_energy_records。

Step 3. 需要澄清
只有在使用者缺少必要資訊，或指代不明到無法判斷任務時，才輸出 clarify_needed。
例：然後呢、幫我查一下、那棟建築的用電、去年的用電。
clarify_needed 必須使用 workflow_id=none, required_tools=[]。

Step 4. 多步驟工作流
如果任務需要多個工具、策略規劃、改善建議、跨年比較、熱點分析、假設情境估算、文件查證，輸出 workflow_chain。
workflow_chain 必須使用 answerability=answerable_multi_tool。

Step 5. 單一工具
如果一個工具即可處理，例如單棟建築用電查詢、全校概況、排行榜、單純異常偵測、單純策略推薦，輸出 single_tool。
single_tool 必須使用 answerability=answerable_single_tool。

## 重要邊界

- 有明確建築名稱，不等於 clarify_needed。
- 有明確建築名稱與超出範圍年份，優先 no_evidence。
- 問文件、定義、法規、模型說明，使用 workflow_chain + document_search_dci。
- 問用電狀況加改善建議，使用 workflow_chain + building_hotspot_improvement。
- 問節能策略規劃，使用 workflow_chain + building_strategy_plan。
- 問如果調整空調、照明、溫度可以省多少，使用 workflow_chain + counterfactual_saving_estimate。
- 問跨年或年度比較，使用 workflow_chain + campus_year_compare。
- 問全校前幾名並要求建議，使用 workflow_chain + campus_top_energy_buildings。

## 範例判斷

博理館 2020 用電
=> single_tool, single_building_year_status, answerable_single_tool

博理館 2015 用電
=> no_evidence, none, no_evidence_expected

博理館 2020 用電狀況跟改善
=> workflow_chain, building_hotspot_improvement, answerable_multi_tool

博理館節能策略規劃
=> workflow_chain, building_strategy_plan, answerable_multi_tool

如果博理館空調調高 2 度省多少
=> workflow_chain, counterfactual_saving_estimate, answerable_multi_tool

CV-RMSE 定義在哪份文件
=> workflow_chain, document_search_dci, answerable_multi_tool

幫我查一下
=> clarify_needed, none, missing_required_arguments

台大醫院用電
=> refusal, none, unsupported_scope

## JSON schema

{{
  "dispatch_type": "<single_tool|workflow_chain|clarify_needed|no_evidence|refusal>",
  "workflow_id": "<one valid workflow_id>",
  "answerability": "<one valid answerability>",
  "locked_entities": {{"building_names": [], "years": [], "metrics": []}},
  "required_tools": [{{"tool": "<valid tool name>", "purpose": "<short purpose>"}}],
  "stop_conditions": ["<condition>"]
}}

只輸出 JSON object。"""


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
    "additionalProperties": False,
    "properties": {
        "dispatch_type": {
            "type": "string",
            "enum": ["single_tool", "workflow_chain", "clarify_needed", "no_evidence", "refusal"],
        },
        "workflow_id": {
            "type": "string",
            "enum": [
                "single_building_year_status",
                "building_hotspot_improvement",
                "campus_top_energy_buildings",
                "campus_year_compare",
                "building_strategy_plan",
                "counterfactual_saving_estimate",
                "document_search_dci",
                "none",
            ],
        },
        "answerability": {
            "type": "string",
            "enum": [
                "answerable_single_tool",
                "answerable_multi_tool",
                "missing_required_arguments",
                "ambiguous_reference",
                "unsupported_scope",
                "unsupported_capability",
                "unsafe_operation",
                "no_evidence_expected",
            ],
        },
        "locked_entities": {
            "type": "object",
            "additionalProperties": False,
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
                "additionalProperties": False,
                "properties": {
                    "tool": {"type": "string"},
                    "purpose": {"type": "string"},
                },
                "required": ["tool", "purpose"],
            },
        },
        "stop_conditions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "dispatch_type",
        "workflow_id",
        "answerability",
        "locked_entities",
        "required_tools",
        "stop_conditions",
    ],
}


if __name__ == "__main__":
    print(f"Prompt length: {len(render_system_prompt())} chars")
