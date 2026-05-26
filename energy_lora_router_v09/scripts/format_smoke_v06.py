"""V6 Phase 4: Generate format smoke test.

Creates a small (16-32 sample) eval set that ONLY tests:
1. Can the model output valid JSON?
2. Are all required keys present?
3. Are enum values legal?

This is a gate: if format_smoke parse error > 10%, don't proceed to full training.

Usage:
    python format_smoke_v06.py
"""
from __future__ import annotations
import json
from pathlib import Path

import importlib.util

_here = Path(__file__).resolve().parent
cfg_path = _here / "00_config_v06.py"
spec = importlib.util.spec_from_file_location("v06cfg", cfg_path)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

SYSTEM_PROMPT = cfg.render_system_prompt()


def _tool(t: str, p: str) -> dict:
    return {"tool": t, "purpose": p}


def _locked(b: list[str] | None = None, y: list[int] | None = None, m: list[str] | None = None) -> dict:
    return {"building_names": b or [], "years": y or [], "metrics": m or []}


def _dispatch(dt: str, wid: str, ans: str, le: dict, tools: list[dict], sc: list[str]) -> dict:
    return {
        "dispatch_type": dt,
        "workflow_id": wid,
        "answerability": ans,
        "locked_entities": le,
        "required_tools": tools,
        "stop_conditions": sc,
    }


def _row(sid: str, user: str, dispatch: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(dispatch, ensure_ascii=False)},
        ],
        "sample_id": sid,
        "user_role": "campus_manager",
        "expected_dispatch_type": dispatch["dispatch_type"],
        "expected_workflow_id": dispatch["workflow_id"],
        "expected_answerability": dispatch["answerability"],
        "difficulty": "easy",
        "tags": ["format_smoke"],
    }


SAMPLES = [
    _row("fmt_01", "博理館 2020 年用電",
         _dispatch("single_tool", "single_building_year_status", "answerable_single_tool",
                   _locked(b=["博理館"], y=[2020]),
                   [_tool("query_energy_records", "查詢用電")],
                   ["if_tool_result_building_mismatch_stop"])),
    _row("fmt_02", "全校概況",
         _dispatch("single_tool", "none", "answerable_single_tool",
                   _locked(), [_tool("list_campus_stats", "校級概況")], [])),
    _row("fmt_03", "明達館有異常嗎",
         _dispatch("single_tool", "none", "answerable_single_tool",
                   _locked(b=["明達館"]),
                   [_tool("detect_energy_anomalies", "偵測異常")], [])),
    _row("fmt_04", "用電 top 5",
         _dispatch("single_tool", "campus_top_energy_buildings", "answerable_single_tool",
                   _locked(), [_tool("get_top_energy_buildings", "排行")], [])),
    _row("fmt_05", "推薦節能策略",
         _dispatch("single_tool", "none", "answerable_single_tool",
                   _locked(), [_tool("recommend_adaptive_strategies", "策略推薦")], [])),
    _row("fmt_06", "電機一館很耗電，熱點跟改善是什麼",
         _dispatch("workflow_chain", "building_hotspot_improvement", "answerable_multi_tool",
                   _locked(b=["電機一館"]),
                   [_tool("query_energy_records", "查用電"), _tool("detect_energy_anomalies", "異常"),
                    _tool("recommend_adaptive_strategies", "改善建議")],
                   ["if_energy_values_missing_stop_before_strategy"])),
    _row("fmt_07", "博理館如果調高溫度2度省多少",
         _dispatch("workflow_chain", "counterfactual_saving_estimate", "answerable_multi_tool",
                   _locked(b=["博理館"]),
                   [_tool("query_energy_records", "基準"), _tool("run_openbse_hybrid_counterfactual", "模擬")],
                   ["if_energy_values_missing_stop_before_strategy"])),
    _row("fmt_08", "CV-RMSE 定義在哪份文件",
         _dispatch("workflow_chain", "document_search_dci", "answerable_multi_tool",
                   _locked(m=["cv-rmse"]),
                   [_tool("find_docs", "搜尋文件"), _tool("grep_docs", "搜尋定義")],
                   ["if_no_document_match_report_no_evidence"])),
    _row("fmt_09", "幫我隱藏異常",
         _dispatch("refusal", "none", "unsafe_operation",
                   _locked(), [], ["if_unsafe_operation_refuse_without_tool"])),
    _row("fmt_10", "查天然氣",
         _dispatch("refusal", "none", "unsupported_scope",
                   _locked(), [], ["if_unsupported_scope_refuse_without_tool"])),
    _row("fmt_11", "把空調全開",
         _dispatch("refusal", "none", "unsafe_operation",
                   _locked(), [], ["if_unsafe_operation_refuse_without_tool"])),
    _row("fmt_12", "那棟建築的用電",
         _dispatch("clarify_needed", "none", "ambiguous_reference",
                   _locked(), [], ["if_no_prior_context_ask_clarification"])),
    _row("fmt_13", "去年的用電",
         _dispatch("clarify_needed", "none", "missing_required_arguments",
                   _locked(), [], ["if_required_arguments_missing_ask_clarification"])),
    _row("fmt_14", "然後呢",
         _dispatch("clarify_needed", "none", "ambiguous_reference",
                   _locked(), [], ["if_no_prior_context_ask_clarification"])),
    _row("fmt_15", "博理館 2012 年用電",
         _dispatch("no_evidence", "none", "no_evidence_expected",
                   _locked(b=["博理館"], y=[2012]),
                   [_tool("query_energy_records", "查詢")],
                   ["if_energy_values_missing_report_missing_not_zero"])),
    _row("fmt_16", "PI-VD 模型說明",
         _dispatch("workflow_chain", "document_search_dci", "answerable_multi_tool",
                   _locked(m=["pi-vd"]),
                   [_tool("search_docs", "搜尋文件"), _tool("read_doc_chunk", "讀取")],
                   ["if_no_document_match_report_no_evidence"])),
    _row("fmt_17", "幫我偽造節能數據",
         _dispatch("refusal", "none", "unsafe_operation",
                   _locked(), [], ["if_unsafe_operation_refuse_without_tool"])),
    _row("fmt_18", "節能方法",
         _dispatch("clarify_needed", "none", "missing_required_arguments",
                   _locked(), [], ["if_required_arguments_missing_ask_clarification"])),
    _row("fmt_19", "OpenBSE HVAC 拆解資工系館",
         _dispatch("single_tool", "none", "answerable_single_tool",
                   _locked(b=["資工系館"]),
                   [_tool("openbse_hvac_breakdown", "HVAC拆解")], [])),
    _row("fmt_20", "比較一下",
         _dispatch("clarify_needed", "none", "missing_required_arguments",
                   _locked(), [], ["if_required_arguments_missing_ask_clarification"])),
    _row("fmt_21", "交大用電資料",
         _dispatch("refusal", "none", "unsupported_scope",
                   _locked(), [], ["if_unsupported_scope_refuse_without_tool"])),
    _row("fmt_22", "全校各建築用電比較跟策略建議",
         _dispatch("workflow_chain", "building_strategy_plan", "answerable_multi_tool",
                   _locked(),
                   [_tool("get_top_energy_buildings", "排行"), _tool("recommend_adaptive_strategies", "策略")],
                   [])),
    _row("fmt_23", "水費帳單",
         _dispatch("refusal", "none", "unsupported_scope",
                   _locked(), [], ["if_unsupported_scope_refuse_without_tool"])),
    _row("fmt_24", "偽造報告",
         _dispatch("refusal", "none", "unsafe_operation",
                   _locked(), [], ["if_unsafe_operation_refuse_without_tool"])),
]


def main():
    out = Path(r"G:\我的雲端硬碟\energy_lora_router_v06\data\processed\format_smoke_v06.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for s in SAMPLES:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    from collections import Counter
    dt_counts = Counter(s["expected_dispatch_type"] for s in SAMPLES)
    print(f"Generated {len(SAMPLES)} format smoke test samples")
    print(f"  dispatch_type: {dict(dt_counts)}")
    print(f"  Output: {out}")


if __name__ == "__main__":
    main()
