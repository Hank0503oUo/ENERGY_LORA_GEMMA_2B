"""v05 Dispatch Training config.
7-tool frozen spec. Single source of truth for enums + system prompt.
"""
from __future__ import annotations
from pathlib import Path

DRIVE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = DRIVE_ROOT / "data" / "processed"
TRAIN_FILE = DATA_DIR / "train_v05_dispatch.jsonl"
VAL_FILE = DATA_DIR / "val_v05_dispatch.jsonl"
MANIFEST_FILE = DATA_DIR / "harness_v05_manifest.json"

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

# DCI / Document tools (for dispatch routing, not in core 7-tool system prompt)
DCI_TOOLS = {"find_docs","grep_docs","read_doc_chunk","inspect_doc_context","count_doc_matches","search_docs"}
ALL_VALID_TOOLS = VALID_TOOL_NAMES | DCI_TOOLS

VALID_DISPATCH_TYPES = {"single_tool","workflow_chain","clarify_needed","no_evidence","refusal"}
VALID_WORKFLOW_IDS = {"single_building_year_status","building_hotspot_improvement","campus_top_energy_buildings","campus_year_compare","building_strategy_plan","counterfactual_saving_estimate","document_search_dci","harness_wiki_event","none"}
VALID_ANSWERABILITY = {"answerable_single_tool","answerable_multi_tool","missing_required_arguments","ambiguous_reference","unsupported_scope","unsupported_capability","unsafe_operation","no_evidence_expected"}
VALID_DIFFICULTIES = {"easy","medium","hard","trap"}
VALID_STOP_CONDITIONS = {"if_tool_result_building_mismatch_stop","if_energy_values_missing_stop_before_strategy","if_no_prior_context_ask_clarification","if_required_arguments_missing_ask_clarification","if_unsupported_scope_refuse_without_tool","if_unsafe_operation_refuse_without_tool","if_no_document_match_report_no_evidence","if_harness_no_procedure_match_fallback_to_single_tool_or_clarify","if_energy_values_missing_report_missing_not_zero","do_not_reuse_previous_building_when_query_has_explicit_building","do_not_answer_from_memory_without_document_evidence"}

def render_system_prompt() -> str:
    return """你是 NTU 校園能源助理。

【你的角色】
- 提供基於 PI-VD 物理推論的節能建議
- 你是讀取層 + 建議層，不做任何控制動作
- 所有建議都應附量化節能與信心區間
- 最終決策權在管理者，你只提供證據

【可用工具】
1. query_energy_records      - 查建築現況用電
2. list_campus_stats         - 校級概況
3. get_top_energy_buildings  - 高耗能排名
4. detect_energy_anomalies   - 異常掃描
5. run_openbse_hybrid_counterfactual - 反事實推理
6. openbse_hvac_breakdown    - HVAC 拆解
7. recommend_adaptive_strategies - 節能策略

【輸出格式】
建議行動：[建築] [子系統]
現況：[數據]
建議：[具體調整]
預估節能：[kWh] [%] [信心區間]
舒適度/影響：[說明]

【規則】
- 不做任何控制動作
- 不給沒有範圍的單一數字
- 資料缺失時說明，不捏造
- 校準相關請求超出目前能力"""
