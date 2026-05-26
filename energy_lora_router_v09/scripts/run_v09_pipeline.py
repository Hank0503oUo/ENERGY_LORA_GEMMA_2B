"""V9 data preparation pipeline.

V9 fixes the issues seen in v08 selection:
- move no_evidence before actionable routing
- remove workflow_id placeholder text from the prompt
- add focused no_evidence / clarify / workflow_chain contrast samples
- regenerate format smoke with clean labels
"""
from __future__ import annotations

import importlib.util
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "processed"
SYNTH = ROOT / "data" / "synth"
GRAMMAR = ROOT / "grammar"

spec = importlib.util.spec_from_file_location("v09cfg", HERE / "00_config_v09.py")
cfg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cfg)

NEW_PROMPT = cfg.render_system_prompt()

BUILDINGS = [
    "博理館",
    "明達館",
    "普通教學館",
    "總圖書館",
    "管理學院一館",
    "電機二館",
]


def locked(buildings=None, years=None, metrics=None) -> dict[str, list[str]]:
    return {
        "building_names": [str(x) for x in (buildings or [])],
        "years": [str(x) for x in (years or [])],
        "metrics": [str(x) for x in (metrics or [])],
    }


def tool(name: str, purpose: str) -> dict[str, str]:
    return {"tool": name, "purpose": purpose}


def dispatch(
    dispatch_type: str,
    workflow_id: str,
    answerability: str,
    locked_entities: dict[str, list[str]],
    required_tools: list[dict[str, str]],
    stop_conditions: list[str],
) -> dict[str, Any]:
    return {
        "dispatch_type": dispatch_type,
        "workflow_id": workflow_id,
        "answerability": answerability,
        "locked_entities": locked_entities,
        "required_tools": required_tools,
        "stop_conditions": stop_conditions,
    }


def row(sample_id: str, query: str, expected: dict[str, Any], tags: list[str], difficulty: str = "easy") -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": NEW_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": json.dumps(expected, ensure_ascii=False)},
        ],
        "sample_id": sample_id,
        "user_role": "campus_manager",
        "expected_dispatch_type": expected["dispatch_type"],
        "expected_workflow_id": expected["workflow_id"],
        "expected_answerability": expected["answerability"],
        "difficulty": difficulty,
        "tags": tags,
    }


def normalize_expected(expected: dict[str, Any]) -> dict[str, Any]:
    expected = dict(expected)
    expected.setdefault("locked_entities", locked())
    expected.setdefault("required_tools", [])
    expected.setdefault("stop_conditions", [])

    if expected.get("dispatch_type") == "no_evidence":
        expected["workflow_id"] = "none"
        expected["answerability"] = "no_evidence_expected"
        expected["required_tools"] = []
    elif expected.get("dispatch_type") in {"clarify_needed", "refusal"}:
        expected["workflow_id"] = "none"
        expected["required_tools"] = []

    if expected.get("workflow_id") == "任務模式":
        expected["workflow_id"] = "none"
    return expected


def reprompt_and_fix(src: Path, dst: Path) -> tuple[int, int]:
    total = fixed = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            item = json.loads(line)
            messages = item.get("messages", [])
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = NEW_PROMPT
                fixed += 1
            if messages and messages[-1].get("role") == "assistant":
                try:
                    expected = normalize_expected(json.loads(messages[-1]["content"]))
                    messages[-1]["content"] = json.dumps(expected, ensure_ascii=False)
                    item["expected_dispatch_type"] = expected["dispatch_type"]
                    item["expected_workflow_id"] = expected["workflow_id"]
                    item["expected_answerability"] = expected["answerability"]
                except json.JSONDecodeError:
                    pass
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    return total, fixed


def generate_boundary_samples() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    idx = 0

    for b in BUILDINGS:
        for year in (2012, 2015, 2016, 2024):
            expected = dispatch(
                "no_evidence",
                "none",
                "no_evidence_expected",
                locked([b], [year]),
                [],
                [],  # no_evidence does not invoke tools
            )
            samples.append(row(f"v09b_{idx:04d}", f"{b} {year} 用電", expected, ["v09", "no_evidence_year"], "easy"))
            idx += 1

        expected = dispatch(
            "single_tool",
            "single_building_year_status",
            "answerable_single_tool",
            locked([b], [2020]),
            [tool("query_energy_records", "查詢指定建築指定年份用電")],
            ["if_tool_result_building_mismatch_stop"],
        )
        samples.append(row(f"v09b_{idx:04d}", f"{b} 2020 用電", expected, ["v09", "single_tool_contrast"], "easy"))
        idx += 1

        expected = dispatch(
            "workflow_chain",
            "building_hotspot_improvement",
            "answerable_multi_tool",
            locked([b], [2020]),
            [
                tool("query_energy_records", "查詢指定建築用電資料"),
                tool("detect_energy_anomalies", "找出能源異常或熱點"),
                tool("recommend_adaptive_strategies", "提出改善策略"),
            ],
            ["if_energy_values_missing_stop_before_strategy"],
        )
        samples.append(row(f"v09b_{idx:04d}", f"{b} 2020 用電狀況跟改善", expected, ["v09", "workflow_contrast"], "medium"))
        idx += 1

        expected = dispatch(
            "workflow_chain",
            "building_strategy_plan",
            "answerable_multi_tool",
            locked([b]),
            [
                tool("query_energy_records", "查詢建築能源基準"),
                tool("get_top_energy_buildings", "比對校園節能優先序"),
                tool("recommend_adaptive_strategies", "規劃節能策略"),
            ],
            ["if_energy_values_missing_stop_before_strategy"],
        )
        samples.append(row(f"v09b_{idx:04d}", f"{b} 節能策略規劃", expected, ["v09", "workflow_strategy"], "medium"))
        idx += 1

        expected = dispatch(
            "workflow_chain",
            "counterfactual_saving_estimate",
            "answerable_multi_tool",
            locked([b], metrics=["hvac_setpoint"]),
            [
                tool("query_energy_records", "取得反事實估算基準"),
                tool("run_openbse_hybrid_counterfactual", "估算調整後節能量"),
            ],
            ["if_energy_values_missing_stop_before_strategy"],
        )
        samples.append(row(f"v09b_{idx:04d}", f"如果 {b} 空調調高 2 度可以省多少", expected, ["v09", "workflow_counterfactual"], "medium"))
        idx += 1

    for query, metric in [
        ("CV-RMSE 定義在哪份文件", "cv-rmse"),
        ("PI-VD 模型說明文件在哪裡", "pi-vd"),
        ("OpenBSE 校準流程文件", "openbse"),
        ("ISO 50001 相關規範在哪份文件", "iso-50001"),
        ("HVAC 控制策略文件", "hvac"),
        ("EUI 指標定義", "eui"),
    ]:
        expected = dispatch(
            "workflow_chain",
            "document_search_dci",
            "answerable_multi_tool",
            locked(metrics=[metric]),
            [
                tool("find_docs", "定位相關文件"),
                tool("grep_docs", "搜尋文件內關鍵詞"),
            ],
            ["if_no_document_match_report_no_evidence"],
        )
        samples.append(row(f"v09b_{idx:04d}", query, expected, ["v09", "document_search"], "medium"))
        idx += 1

    clarify_cases = [
        ("然後呢", "ambiguous_reference"),
        ("那棟建築的用電", "ambiguous_reference"),
        ("幫我查一下", "missing_required_arguments"),
        ("去年的用電", "missing_required_arguments"),
        ("節能方法", "missing_required_arguments"),
        ("看一下異常", "missing_required_arguments"),
    ]
    for query, answerability in clarify_cases:
        stop = "if_no_prior_context_ask_clarification" if answerability == "ambiguous_reference" else "if_required_arguments_missing_ask_clarification"
        expected = dispatch("clarify_needed", "none", answerability, locked(), [], [stop])
        samples.append(row(f"v09b_{idx:04d}", query, expected, ["v09", "clarify_boundary"], "easy"))
        idx += 1

    refusal_cases = [
        ("台大醫院用電", "unsupported_scope"),
        ("查天然氣用量", "unsupported_scope"),
        ("把全校冷氣全部開到最強", "unsafe_operation"),
        ("幫我把用電數字改低一點", "unsafe_operation"),
    ]
    for query, answerability in refusal_cases:
        stop = "if_unsafe_operation_refuse_without_tool" if answerability == "unsafe_operation" else "if_unsupported_scope_refuse_without_tool"
        expected = dispatch("refusal", "none", answerability, locked(), [], [stop])
        samples.append(row(f"v09b_{idx:04d}", query, expected, ["v09", "refusal_boundary"], "trap"))
        idx += 1

    random.Random(42).shuffle(samples)
    return samples


def generate_format_smoke() -> list[dict[str, Any]]:
    tests = [
        ("博理館 2020 用電", "single_tool", "single_building_year_status", "answerable_single_tool", locked(["博理館"], [2020]), [tool("query_energy_records", "查詢用電")], ["if_tool_result_building_mismatch_stop"]),
        ("博理館 2015 用電", "no_evidence", "none", "no_evidence_expected", locked(["博理館"], [2015]), [], []),
        ("明達館 2024 EUI", "no_evidence", "none", "no_evidence_expected", locked(["明達館"], [2024], ["EUI"]), [], []),
        ("博理館 2020 用電狀況跟改善", "workflow_chain", "building_hotspot_improvement", "answerable_multi_tool", locked(["博理館"], [2020]), [tool("query_energy_records", "查詢用電"), tool("detect_energy_anomalies", "找熱點"), tool("recommend_adaptive_strategies", "提出改善")], ["if_energy_values_missing_stop_before_strategy"]),
        ("博理館節能策略規劃", "workflow_chain", "building_strategy_plan", "answerable_multi_tool", locked(["博理館"]), [tool("query_energy_records", "查詢基準"), tool("get_top_energy_buildings", "比對排名"), tool("recommend_adaptive_strategies", "規劃策略")], ["if_energy_values_missing_stop_before_strategy"]),
        ("如果博理館空調調高 2 度省多少", "workflow_chain", "counterfactual_saving_estimate", "answerable_multi_tool", locked(["博理館"], metrics=["hvac_setpoint"]), [tool("query_energy_records", "查詢基準"), tool("run_openbse_hybrid_counterfactual", "估算節能")], ["if_energy_values_missing_stop_before_strategy"]),
        ("CV-RMSE 定義在哪份文件", "workflow_chain", "document_search_dci", "answerable_multi_tool", locked(metrics=["cv-rmse"]), [tool("find_docs", "定位文件"), tool("grep_docs", "搜尋定義")], ["if_no_document_match_report_no_evidence"]),
        ("全校用電前 5 名各自的節能建議", "workflow_chain", "campus_top_energy_buildings", "answerable_multi_tool", locked(), [tool("get_top_energy_buildings", "找前 5 名"), tool("recommend_adaptive_strategies", "逐棟建議")], []),
        ("2019 跟 2020 全校用電比較", "workflow_chain", "campus_year_compare", "answerable_multi_tool", locked(years=[2019, 2020]), [tool("query_energy_records", "查詢兩年度資料"), tool("list_campus_stats", "整理年度統計")], []),
        ("全校用電概況", "single_tool", "none", "answerable_single_tool", locked(), [tool("list_campus_stats", "列出概況")], []),
        ("幫博理館推薦節能策略", "single_tool", "none", "answerable_single_tool", locked(["博理館"]), [tool("recommend_adaptive_strategies", "推薦策略")], []),
        ("然後呢", "clarify_needed", "none", "ambiguous_reference", locked(), [], ["if_no_prior_context_ask_clarification"]),
        ("幫我查一下", "clarify_needed", "none", "missing_required_arguments", locked(), [], ["if_required_arguments_missing_ask_clarification"]),
        ("台大醫院用電", "refusal", "none", "unsupported_scope", locked(), [], ["if_unsupported_scope_refuse_without_tool"]),
        ("查天然氣用量", "refusal", "none", "unsupported_scope", locked(), [], ["if_unsupported_scope_refuse_without_tool"]),
        ("把全校冷氣全部開到最強", "refusal", "none", "unsafe_operation", locked(), [], ["if_unsafe_operation_refuse_without_tool"]),
    ]
    return [
        row(f"fmt_v09_{i:03d}", query, dispatch(dt, wid, ans, le, tools, stops), ["format_smoke_v09"])
        for i, (query, dt, wid, ans, le, tools, stops) in enumerate(tests, 1)
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def count_jsonl(path: Path) -> int:
    return sum(1 for line in path.open("r", encoding="utf-8") if line.strip()) if path.exists() else 0


def main() -> None:
    print("=" * 60)
    print("V9 FORMAT + NO_EVIDENCE REPAIR PIPELINE")
    print("=" * 60)

    for src_name, dst_name in [
        ("train_v08_dispatch.jsonl", "train_v09_dispatch.jsonl"),
        ("val_v08_dispatch.jsonl", "val_v09_dispatch.jsonl"),
        ("smoke_v08_dispatch.jsonl", "smoke_v09_dispatch.jsonl"),
    ]:
        total, fixed = reprompt_and_fix(DATA / src_name, DATA / dst_name)
        print(f"[reprompt] {dst_name}: {total} rows, {fixed} prompts replaced")

    boundary_rows = generate_boundary_samples()
    write_jsonl(SYNTH / "v09_boundary_curriculum.jsonl", boundary_rows)
    print(f"[boundary] {len(boundary_rows)} samples: {dict(Counter(r['expected_dispatch_type'] for r in boundary_rows))}")

    train_path = DATA / "train_v09_dispatch.jsonl"
    existing = [json.loads(line) for line in train_path.open("r", encoding="utf-8") if line.strip()]
    write_jsonl(train_path, boundary_rows + existing)
    print(f"[merge] train_v09_dispatch.jsonl: {len(existing)} + {len(boundary_rows)} = {len(existing) + len(boundary_rows)}")

    format_smoke = generate_format_smoke()
    write_jsonl(DATA / "format_smoke_v09.jsonl", format_smoke)
    print(f"[format-smoke] {len(format_smoke)} samples")

    GRAMMAR.mkdir(parents=True, exist_ok=True)
    (GRAMMAR / "dispatch_schema.gbnf").write_text(cfg.JSON_GRAMMAR, encoding="utf-8")
    (GRAMMAR / "dispatch_json_schema.json").write_text(json.dumps(cfg.DISPATCH_JSON_SCHEMA, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[grammar] updated")

    manifest = {
        "version": "0.9",
        "profile": "agent_dispatch_training",
        "schema": "dispatch_chat_messages",
        "total": count_jsonl(DATA / "train_v09_dispatch.jsonl") + count_jsonl(DATA / "val_v09_dispatch.jsonl") + count_jsonl(DATA / "smoke_v09_dispatch.jsonl"),
        "train": count_jsonl(DATA / "train_v09_dispatch.jsonl"),
        "val": count_jsonl(DATA / "val_v09_dispatch.jsonl"),
        "smoke": count_jsonl(DATA / "smoke_v09_dispatch.jsonl"),
        "format_smoke": count_jsonl(DATA / "format_smoke_v09.jsonl"),
        "v09_changes": [
            "no_evidence is evaluated before actionable tool routing",
            "removed workflow_id placeholder wording from prompt",
            "added no_evidence year-range pre-gate examples",
            "added clean workflow_chain vs single_tool contrast examples",
            "increased evaluation max_new_tokens default to 512",
        ],
    }
    (DATA / "harness_v09_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[manifest] train={manifest['train']} val={manifest['val']} smoke={manifest['smoke']} format_smoke={manifest['format_smoke']}")

    print("=" * 60)
    print("V9 PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
