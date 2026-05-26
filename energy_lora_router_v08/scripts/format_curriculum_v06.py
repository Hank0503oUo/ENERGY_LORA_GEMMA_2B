"""V6 Phase 3: Generate format curriculum samples.

Creates 80-150 samples specifically designed to teach the model to:
1. Always output valid JSON
2. Always include all fields
3. Never output natural language

These are mixed into the training split.

Usage:
    python format_curriculum_v06.py
    python format_curriculum_v06.py --count 120
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import importlib.util

_here = Path(__file__).resolve().parent
cfg_path = _here / "00_config_v06.py"
spec = importlib.util.spec_from_file_location("v06cfg", cfg_path)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

SYSTEM_PROMPT = cfg.render_system_prompt()

BUILDINGS = [
    "博理館", "明達館", "新博士生宿舍", "電機一館", "電機二館",
    "資工系館", "工學院綜合大樓", "共同教室館", "新生大樓", "應力館",
    "凝態科學研究中心", "物理學館", "化學工程學系館", "材料科學與工程學系館",
]

YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023]


def _tool_entry(tool: str, purpose: str) -> dict:
    return {"tool": tool, "purpose": purpose}


def _locked(buildings: list[str] | None = None, years: list[int] | None = None,
            metrics: list[str] | None = None) -> dict:
    return {
        "building_names": buildings or [],
        "years": years or [],
        "metrics": metrics or [],
    }


def _dispatch(dispatch_type: str, workflow_id: str, answerability: str,
              locked: dict, tools: list[dict], stop_conditions: list[str]) -> dict:
    return {
        "dispatch_type": dispatch_type,
        "workflow_id": workflow_id,
        "answerability": answerability,
        "locked_entities": locked,
        "required_tools": tools,
        "stop_conditions": stop_conditions,
    }


def generate_samples(count: int = 120) -> list[dict]:
    rng = random.Random(42)
    samples: list[dict] = []
    idx = 0

    category_templates = [
        _gen_short_queries,
        _gen_document_queries,
        _gen_single_tool_queries,
        _gen_workflow_queries,
        _gen_refusal_queries,
        _gen_clarify_queries,
        _gen_no_evidence_queries,
    ]

    per_cat = count // len(category_templates)
    remainder = count - per_cat * len(category_templates)

    for gen_fn in category_templates:
        n = per_cat + (1 if remainder > 0 else 0)
        remainder = max(0, remainder - 1)
        for row in gen_fn(rng, n):
            idx += 1
            row["sample_id"] = f"v06_format_{idx:04d}"
            row["difficulty"] = "easy"
            samples.append(row)

    rng.shuffle(samples)
    return samples


def _make_row(rng: random.Random, user_query: str, dispatch: dict, tags: list[str]) -> dict:
    assistant_content = json.dumps(dispatch, ensure_ascii=False)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": assistant_content},
    ]
    return {
        "messages": messages,
        "user_role": "campus_manager",
        "expected_dispatch_type": dispatch["dispatch_type"],
        "expected_workflow_id": dispatch["workflow_id"],
        "expected_answerability": dispatch["answerability"],
        "tags": tags,
    }


def _gen_short_queries(rng: random.Random, n: int) -> list[dict]:
    queries = [
        ("然後呢", "clarify_needed", "none", "ambiguous_reference",
         _locked(), [], ["if_no_prior_context_ask_clarification"]),
        ("幫我看一下", "clarify_needed", "none", "missing_required_arguments",
         _locked(), [], ["if_required_arguments_missing_ask_clarification"]),
        ("節能方法", "clarify_needed", "none", "missing_required_arguments",
         _locked(), [], ["if_required_arguments_missing_ask_clarification"]),
        ("用電", "clarify_needed", "none", "missing_required_arguments",
         _locked(), [], ["if_required_arguments_missing_ask_clarification"]),
        ("異常", "clarify_needed", "none", "ambiguous_reference",
         _locked(), [], ["if_no_prior_context_ask_clarification"]),
        ("那棟建築呢", "clarify_needed", "none", "ambiguous_reference",
         _locked(), [], ["if_no_prior_context_ask_clarification"]),
        ("比較一下", "clarify_needed", "none", "missing_required_arguments",
         _locked(), [], ["if_required_arguments_missing_ask_clarification"]),
        ("今天", "clarify_needed", "none", "ambiguous_reference",
         _locked(), [], ["if_no_prior_context_ask_clarification"]),
    ]
    rows = []
    for i in range(n):
        q = queries[i % len(queries)]
        rows.append(_make_row(rng, q[0],
            _dispatch(q[1], q[2], q[3], q[4], q[5], q[6]),
            ["format_curriculum", "short_query"]))
    return rows


def _gen_document_queries(rng: random.Random, n: int) -> list[dict]:
    queries = [
        ("CV-RMSE 的定義在哪份文件",
         "workflow_chain", "document_search_dci", "answerable_multi_tool",
         _locked(metrics=["cv-rmse"]),
         [_tool_entry("find_docs", "搜尋 CV-RMSE 相關文件"), _tool_entry("grep_docs", "搜尋定義")],
         ["if_no_document_match_report_no_evidence"]),
        ("PI-VD 模型說明文件",
         "workflow_chain", "document_search_dci", "answerable_multi_tool",
         _locked(metrics=["pi-vd"]),
         [_tool_entry("find_docs", "搜尋 PI-VD 文件"), _tool_entry("read_doc_chunk", "讀取相關內容")],
         ["if_no_document_match_report_no_evidence"]),
        ("節能法規條文在哪裡",
         "workflow_chain", "document_search_dci", "answerable_multi_tool",
         _locked(),
         [_tool_entry("search_docs", "搜尋節能法規"), _tool_entry("inspect_doc_context", "檢查上下文")],
         ["if_no_document_match_report_no_evidence"]),
        ("OpenBSE 的技術手冊",
         "workflow_chain", "document_search_dci", "answerable_multi_tool",
         _locked(),
         [_tool_entry("find_docs", "搜尋 OpenBSE 手冊"), _tool_entry("count_doc_matches", "確認有匹配")],
         ["if_no_document_match_report_no_evidence"]),
        ("ASHP 性能係數說明",
         "workflow_chain", "document_search_dci", "answerable_multi_tool",
         _locked(metrics=["ashp"]),
         [_tool_entry("search_docs", "搜尋 ASHP 文件"), _tool_entry("grep_docs", "搜尋性能係數")],
         ["if_no_document_match_report_no_evidence"]),
    ]
    rows = []
    for i in range(n):
        q = queries[i % len(queries)]
        rows.append(_make_row(rng, q[0],
            _dispatch(q[1], q[2], q[3], q[4], q[5], q[6]),
            ["format_curriculum", "document_query"]))
    return rows


def _gen_single_tool_queries(rng: random.Random, n: int) -> list[dict]:
    bldg = rng.choice(BUILDINGS)
    year = rng.choice(YEARS)
    queries = [
        (f"{bldg} {year} 用電", "single_tool", "single_building_year_status", "answerable_single_tool",
         _locked(buildings=[bldg], years=[year]),
         [_tool_entry("query_energy_records", f"查詢{bldg} {year}年用電")],
         ["if_tool_result_building_mismatch_stop"]),
        (f"{bldg} 有異常嗎", "single_tool", "none", "answerable_single_tool",
         _locked(buildings=[bldg]),
         [_tool_entry("detect_energy_anomalies", f"偵測{bldg}異常")],
         []),
        ("全校概況統計", "single_tool", "none", "answerable_single_tool",
         _locked(),
         [_tool_entry("list_campus_stats", "校級概況統計")],
         []),
        ("用電 top 10", "single_tool", "campus_top_energy_buildings", "answerable_single_tool",
         _locked(),
         [_tool_entry("get_top_energy_buildings", "用電排行")],
         []),
        (f"推薦{bldg}的節能策略", "single_tool", "none", "answerable_single_tool",
         _locked(buildings=[bldg]),
         [_tool_entry("recommend_adaptive_strategies", f"{bldg}節能策略")],
         []),
    ]
    rows = []
    for i in range(n):
        q = queries[i % len(queries)]
        rows.append(_make_row(rng, q[0],
            _dispatch(q[1], q[2], q[3], q[4], q[5], q[6]),
            ["format_curriculum", "single_tool"]))
    return rows


def _gen_workflow_queries(rng: random.Random, n: int) -> list[dict]:
    bldg = rng.choice(BUILDINGS)
    year = rng.choice(YEARS)
    queries = [
        (f"{bldg} 很耗電，熱點跟改善方案是什麼",
         "workflow_chain", "building_hotspot_improvement", "answerable_multi_tool",
         _locked(buildings=[bldg]),
         [_tool_entry("query_energy_records", f"查{bldg}用電"), _tool_entry("detect_energy_anomalies", "偵測異常"),
          _tool_entry("recommend_adaptive_strategies", "改善建議")],
         ["if_energy_values_missing_stop_before_strategy"]),
        (f"{bldg} 如果調高空調溫度2度可以省多少",
         "workflow_chain", "counterfactual_saving_estimate", "answerable_multi_tool",
         _locked(buildings=[bldg]),
         [_tool_entry("query_energy_records", f"查{bldg}用電基準"), _tool_entry("run_openbse_hybrid_counterfactual", "反事實模擬")],
         ["if_energy_values_missing_stop_before_strategy"]),
        (f"全校各建築用電比較跟{year}年度策略建議",
         "workflow_chain", "building_strategy_plan", "answerable_multi_tool",
         _locked(years=[year]),
         [_tool_entry("get_top_energy_buildings", "排行"), _tool_entry("recommend_adaptive_strategies", "策略建議")],
         []),
    ]
    rows = []
    for i in range(n):
        q = queries[i % len(queries)]
        rows.append(_make_row(rng, q[0],
            _dispatch(q[1], q[2], q[3], q[4], q[5], q[6]),
            ["format_curriculum", "workflow"]))
    return rows


def _gen_refusal_queries(rng: random.Random, n: int) -> list[dict]:
    queries = [
        ("幫我隱藏異常事件", "refusal", "none", "unsafe_operation",
         _locked(), [], ["if_unsafe_operation_refuse_without_tool"]),
        ("查天然氣用量", "refusal", "none", "unsupported_scope",
         _locked(), [], ["if_unsupported_scope_refuse_without_tool"]),
        ("幫我修改電表數據", "refusal", "none", "unsafe_operation",
         _locked(), [], ["if_unsafe_operation_refuse_without_tool"]),
        ("台大醫院用電多少", "refusal", "none", "unsupported_scope",
         _locked(), [], ["if_unsupported_scope_refuse_without_tool"]),
        ("把全校空調開到最大", "refusal", "none", "unsafe_operation",
         _locked(), [], ["if_unsafe_operation_refuse_without_tool"]),
        ("交大的用電資料", "refusal", "none", "unsupported_scope",
         _locked(), [], ["if_unsupported_scope_refuse_without_tool"]),
        ("水費帳單查詢", "refusal", "none", "unsupported_scope",
         _locked(), [], ["if_unsupported_scope_refuse_without_tool"]),
        ("偽造節能報告", "refusal", "none", "unsafe_operation",
         _locked(), [], ["if_unsafe_operation_refuse_without_tool"]),
    ]
    rows = []
    for i in range(n):
        q = queries[i % len(queries)]
        rows.append(_make_row(rng, q[0],
            _dispatch(q[1], q[2], q[3], q[4], q[5], q[6]),
            ["format_curriculum", "refusal"]))
    return rows


def _gen_clarify_queries(rng: random.Random, n: int) -> list[dict]:
    queries = [
        ("那棟建築的用電", "clarify_needed", "none", "ambiguous_reference",
         _locked(), [], ["if_no_prior_context_ask_clarification"]),
        ("去年的用電", "clarify_needed", "none", "missing_required_arguments",
         _locked(), [], ["if_required_arguments_missing_ask_clarification"]),
        ("我要查資料", "clarify_needed", "none", "missing_required_arguments",
         _locked(), [], ["if_required_arguments_missing_ask_clarification"]),
        ("跟隔壁比", "clarify_needed", "none", "ambiguous_reference",
         _locked(), [], ["if_no_prior_context_ask_clarification"]),
        ("比較一下這幾年", "clarify_needed", "none", "missing_required_arguments",
         _locked(), [], ["if_required_arguments_missing_ask_clarification"]),
    ]
    rows = []
    for i in range(n):
        q = queries[i % len(queries)]
        rows.append(_make_row(rng, q[0],
            _dispatch(q[1], q[2], q[3], q[4], q[5], q[6]),
            ["format_curriculum", "clarify"]))
    return rows


def _gen_no_evidence_queries(rng: random.Random, n: int) -> list[dict]:
    queries = [
        ("博理館 2012 年的用電", "no_evidence", "none", "no_evidence_expected",
         _locked(buildings=["博理館"], years=[2012]),
         [_tool_entry("query_energy_records", "查詢2012年資料")],
         ["if_energy_values_missing_report_missing_not_zero"]),
        ("凝態中心 2010 年 EUI", "no_evidence", "none", "no_evidence_expected",
         _locked(buildings=["凝態科學研究中心"], years=[2010]),
         [_tool_entry("query_energy_records", "查詢2010年資料")],
         ["if_energy_values_missing_report_missing_not_zero"]),
        ("新生大樓 2008 年能耗", "no_evidence", "none", "no_evidence_expected",
         _locked(buildings=["新生大樓"], years=[2008]),
         [_tool_entry("query_energy_records", "查詢2008年資料")],
         ["if_energy_values_missing_report_missing_not_zero"]),
    ]
    rows = []
    for i in range(n):
        q = queries[i % len(queries)]
        rows.append(_make_row(rng, q[0],
            _dispatch(q[1], q[2], q[3], q[4], q[5], q[6]),
            ["format_curriculum", "no_evidence"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate v06 format curriculum samples")
    parser.add_argument("--count", type=int, default=120, help="Number of samples to generate")
    parser.add_argument("--output", type=Path,
                        default=Path(r"G:\我的雲端硬碟\energy_lora_router_v06\data\synth\v06_format_curriculum.jsonl"),
                        help="Output file path")
    args = parser.parse_args()

    samples = generate_samples(args.count)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    from collections import Counter
    dt_counts = Counter(s["expected_dispatch_type"] for s in samples)
    ans_counts = Counter(s["expected_answerability"] for s in samples)
    print(f"Generated {len(samples)} format curriculum samples")
    print(f"  dispatch_type: {dict(dt_counts)}")
    print(f"  answerability: {dict(ans_counts)}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
