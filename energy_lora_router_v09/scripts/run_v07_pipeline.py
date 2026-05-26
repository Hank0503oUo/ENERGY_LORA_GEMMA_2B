"""V7 full pipeline — schema clarity round.

Phases:
  1. Reprompt v06 data with v07 system prompt
  2. Fix schema: no_evidence required_tools forced to []
  3. Add contrast samples (single_tool vs workflow_chain)
  4. Generate format smoke test
  5. Generate manifest + grammar files

Usage: python run_v07_pipeline.py
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from collections import Counter

import importlib.util

_here = Path(__file__).resolve().parent
cfg_path = _here / "00_config_v07.py"
spec = importlib.util.spec_from_file_location("v07cfg", cfg_path)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

NEW_PROMPT = cfg.render_system_prompt()

V06_DATA = Path(r"G:\我的雲端硬碟\energy_lora_router_v06\data\processed")
V07_DATA = Path(r"G:\我的雲端硬碟\energy_lora_router_v07\data\processed")
V07_SYNTH = Path(r"G:\我的雲端硬碟\energy_lora_router_v07\data\synth")
V07_ROOT = Path(r"G:\我的雲端硬碟\energy_lora_router_v07")

BUILDINGS = [
    "博理館", "明達館", "新博士生宿舍", "電機一館", "電機二館",
    "資工系館", "工學院綜合大樓", "共同教室館", "新生大樓", "應力館",
]
YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023]


# ── helpers ──

def _locked(b=None, y=None, m=None):
    return {"building_names": b or [], "years": y or [], "metrics": m or []}

def _tool(t, p):
    return {"tool": t, "purpose": p}

def _dispatch(dt, wid, ans, locked, tools, sc):
    return {
        "dispatch_type": dt, "workflow_id": wid, "answerability": ans,
        "locked_entities": locked, "required_tools": tools, "stop_conditions": sc,
    }

def _row(sid, user_query, dispatch, tags):
    messages = [
        {"role": "system", "content": NEW_PROMPT},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": json.dumps(dispatch, ensure_ascii=False)},
    ]
    return {
        "messages": messages, "sample_id": sid, "user_role": "campus_manager",
        "expected_dispatch_type": dispatch["dispatch_type"],
        "expected_workflow_id": dispatch["workflow_id"],
        "expected_answerability": dispatch["answerability"],
        "difficulty": "easy", "tags": tags,
    }


# ── Phase 1: reprompt + fix schema ──

def reprompt_and_fix(src, dst):
    total = patched = fixed_ne = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            msgs = row.get("messages", [])
            if msgs and msgs[0].get("role") == "system":
                msgs[0]["content"] = NEW_PROMPT
                patched += 1
            if msgs and msgs[-1].get("role") == "assistant":
                try:
                    parsed = json.loads(msgs[-1]["content"])
                    if parsed.get("dispatch_type") == "no_evidence" and parsed.get("required_tools"):
                        parsed["required_tools"] = []
                        msgs[-1]["content"] = json.dumps(parsed, ensure_ascii=False)
                        fixed_ne += 1
                except (json.JSONDecodeError, TypeError):
                    pass
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return total, patched, fixed_ne


# ── Phase 2: contrast samples ──

def generate_contrast_samples(count=150):
    rng = random.Random(42)
    samples = []
    idx = 0

    # 2a: same building — single_tool vs workflow_chain
    for bldg in BUILDINGS[:6]:
        for year in YEARS[:4]:
            q1 = f"{bldg} {year} 用電"
            d1 = _dispatch("single_tool", "single_building_year_status", "answerable_single_tool",
                           _locked(b=[bldg], y=[year]),
                           [_tool("query_energy_records", f"查{bldg}{year}用電")],
                           ["if_tool_result_building_mismatch_stop"])
            samples.append(_row(f"v07c_{idx:04d}", q1, d1, ["contrast", "single_tool"]))
            idx += 1

            q2 = f"{bldg} {year} 用電狀況跟改善建議"
            d2 = _dispatch("workflow_chain", "building_hotspot_improvement", "answerable_multi_tool",
                           _locked(b=[bldg], y=[year]),
                           [_tool("query_energy_records", f"查{bldg}用電"),
                            _tool("detect_energy_anomalies", "偵測異常"),
                            _tool("recommend_adaptive_strategies", "改善建議")],
                           ["if_energy_values_missing_stop_before_strategy"])
            samples.append(_row(f"v07c_{idx:04d}", q2, d2, ["contrast", "workflow_chain"]))
            idx += 1

    # 2b: recommend_adaptive_strategies is a TOOL
    for bldg in BUILDINGS[:5]:
        q = f"幫{bldg}推薦節能策略"
        d = _dispatch("single_tool", "none", "answerable_single_tool",
                      _locked(b=[bldg]),
                      [_tool("recommend_adaptive_strategies", f"{bldg}節能策略")], [])
        samples.append(_row(f"v07c_{idx:04d}", q, d, ["contrast", "tool_not_dispatch"]))
        idx += 1

    # 2c: building_strategy_plan is workflow, needs multi-tool
    for bldg in BUILDINGS[:5]:
        q = f"幫{bldg}做完整的節能策略規劃"
        d = _dispatch("workflow_chain", "building_strategy_plan", "answerable_multi_tool",
                      _locked(b=[bldg]),
                      [_tool("query_energy_records", "查用電基準"),
                       _tool("get_top_energy_buildings", "比較排名"),
                       _tool("recommend_adaptive_strategies", "策略建議")],
                      ["if_energy_values_missing_stop_before_strategy"])
        samples.append(_row(f"v07c_{idx:04d}", q, d, ["contrast", "workflow_vs_single"]))
        idx += 1

    # 2d: document_search_dci is workflow
    for q_text, metric in [("CV-RMSE 定義", "cv-rmse"), ("PI-VD 模型說明", "pi-vd"),
                           ("OpenBSE 技術手冊", "openbse"), ("節能法規條文", "regulation"),
                           ("HVAC 設計規範", "hvac"), ("ASHP 性能係數", "ashp")]:
        d = _dispatch("workflow_chain", "document_search_dci", "answerable_multi_tool",
                      _locked(m=[metric]),
                      [_tool("find_docs", f"搜尋{q_text}文件"), _tool("grep_docs", "搜尋定義")],
                      ["if_no_document_match_report_no_evidence"])
        samples.append(_row(f"v07c_{idx:04d}", f"{q_text}在哪份文件", d, ["contrast", "document_workflow"]))
        idx += 1

    # 2e: no_evidence with empty tools
    for q_text, bldg, year in [("博理館 2012 年用電", "博理館", 2012),
                                ("明達館 2010 年 EUI", "明達館", 2010),
                                ("電機一館 2008 年能耗", "電機一館", 2008),
                                ("共同教室館 2005 年資料", "共同教室館", 2005)]:
        d = _dispatch("no_evidence", "none", "no_evidence_expected",
                      _locked(b=[bldg], y=[year]), [],
                      ["if_energy_values_missing_report_missing_not_zero"])
        samples.append(_row(f"v07c_{idx:04d}", q_text, d, ["contrast", "no_evidence_no_tools"]))
        idx += 1

    # 2f: counterfactual is workflow
    for bldg in BUILDINGS[:4]:
        d = _dispatch("workflow_chain", "counterfactual_saving_estimate", "answerable_multi_tool",
                      _locked(b=[bldg]),
                      [_tool("query_energy_records", "基準用電"),
                       _tool("run_openbse_hybrid_counterfactual", "反事實模擬")],
                      ["if_energy_values_missing_stop_before_strategy"])
        samples.append(_row(f"v07c_{idx:04d}", f"如果{bldg}空調調高2度可以省多少", d, ["contrast", "counterfactual"]))
        idx += 1

    # 2g: campus_top as single vs workflow
    samples.append(_row(f"v07c_{idx:04d}", "用電前 5 名",
        _dispatch("single_tool", "campus_top_energy_buildings", "answerable_single_tool",
                  _locked(), [_tool("get_top_energy_buildings", "排行")], []),
        ["contrast", "single_tool"]))
    idx += 1

    samples.append(_row(f"v07c_{idx:04d}", "全校用電前 5 名的建築個別節能建議",
        _dispatch("workflow_chain", "campus_top_energy_buildings", "answerable_multi_tool",
                  _locked(),
                  [_tool("get_top_energy_buildings", "排行"), _tool("recommend_adaptive_strategies", "策略")], []),
        ["contrast", "workflow_chain"]))
    idx += 1

    rng.shuffle(samples)
    return samples[:count]


# ── Phase 3: format smoke ──

def generate_format_smoke():
    samples = []
    idx = 0
    tests = [
        ("博理館 2020 用電", "single_tool", "single_building_year_status", "answerable_single_tool",
         _locked(b=["博理館"], y=[2020]), [_tool("query_energy_records", "查用電")], ["if_tool_result_building_mismatch_stop"]),
        ("全校概況", "single_tool", "none", "answerable_single_tool",
         _locked(), [_tool("list_campus_stats", "校級概況")], []),
        ("明達館有異常嗎", "single_tool", "none", "answerable_single_tool",
         _locked(b=["明達館"]), [_tool("detect_energy_anomalies", "偵測異常")], []),
        ("用電 top 5", "single_tool", "campus_top_energy_buildings", "answerable_single_tool",
         _locked(), [_tool("get_top_energy_buildings", "排行")], []),
        ("電機一館 2020 用電狀況跟改善建議", "workflow_chain", "building_hotspot_improvement", "answerable_multi_tool",
         _locked(b=["電機一館"], y=[2020]),
         [_tool("query_energy_records", "查用電"), _tool("detect_energy_anomalies", "偵測異常"),
          _tool("recommend_adaptive_strategies", "改善建議")],
         ["if_energy_values_missing_stop_before_strategy"]),
        ("博理館調高溫度2度省多少", "workflow_chain", "counterfactual_saving_estimate", "answerable_multi_tool",
         _locked(b=["博理館"]),
         [_tool("query_energy_records", "基準"), _tool("run_openbse_hybrid_counterfactual", "模擬")],
         ["if_energy_values_missing_stop_before_strategy"]),
        ("CV-RMSE 定義在哪份文件", "workflow_chain", "document_search_dci", "answerable_multi_tool",
         _locked(m=["cv-rmse"]),
         [_tool("find_docs", "搜尋文件"), _tool("grep_docs", "搜尋定義")],
         ["if_no_document_match_report_no_evidence"]),
        ("幫博理館推薦節能策略", "single_tool", "none", "answerable_single_tool",
         _locked(b=["博理館"]), [_tool("recommend_adaptive_strategies", "節能策略")], []),
        ("幫博理館做完整節能規劃", "workflow_chain", "building_strategy_plan", "answerable_multi_tool",
         _locked(b=["博理館"]),
         [_tool("query_energy_records", "查用電"), _tool("get_top_energy_buildings", "比較排名"),
          _tool("recommend_adaptive_strategies", "策略")],
         ["if_energy_values_missing_stop_before_strategy"]),
        ("查天然氣", "refusal", "none", "unsupported_scope",
         _locked(), [], ["if_unsupported_scope_refuse_without_tool"]),
        ("幫我隱藏異常", "refusal", "none", "unsafe_operation",
         _locked(), [], ["if_unsafe_operation_refuse_without_tool"]),
        ("然後呢", "clarify_needed", "none", "ambiguous_reference",
         _locked(), [], ["if_no_prior_context_ask_clarification"]),
        ("去年的用電", "clarify_needed", "none", "missing_required_arguments",
         _locked(), [], ["if_required_arguments_missing_ask_clarification"]),
        ("博理館 2012 年用電", "no_evidence", "none", "no_evidence_expected",
         _locked(b=["博理館"], y=[2012]), [],
         ["if_energy_values_missing_report_missing_not_zero"]),
        ("全校用電前5名各自的節能建議", "workflow_chain", "campus_top_energy_buildings", "answerable_multi_tool",
         _locked(),
         [_tool("get_top_energy_buildings", "排行"), _tool("recommend_adaptive_strategies", "策略")], []),
        ("交大用電", "refusal", "none", "unsupported_scope",
         _locked(), [], ["if_unsupported_scope_refuse_without_tool"]),
    ]
    for q, dt, wid, ans, locked, tools, sc in tests:
        idx += 1
        samples.append(_row(f"fmt_v07_{idx:03d}", q, _dispatch(dt, wid, ans, locked, tools, sc), ["format_smoke_v07"]))
    return samples


# ── main ──

def main():
    print("=" * 60)
    print("V7 SCHEMA CLARITY PIPELINE")
    print("=" * 60)

    # Phase 1
    print("\n--- Phase 1: Reprompt + Fix Schema ---")
    for sn, dn in [("train_v06_dispatch.jsonl", "train_v07_dispatch.jsonl"),
                    ("val_v06_dispatch.jsonl", "val_v07_dispatch.jsonl"),
                    ("smoke_v06_dispatch.jsonl", "smoke_v07_dispatch.jsonl")]:
        src, dst = V06_DATA / sn, V07_DATA / dn
        if not src.exists():
            print(f"  SKIP {sn}"); continue
        t, p, f = reprompt_and_fix(src, dst)
        print(f"  {dn}: {t} rows, {p} reprompted, {f} no_evidence tools stripped")

    # Phase 2
    print("\n--- Phase 2: Contrast Samples ---")
    contrast = generate_contrast_samples(150)
    V07_SYNTH.mkdir(parents=True, exist_ok=True)
    cp = V07_SYNTH / "v07_contrast_curriculum.jsonl"
    with cp.open("w", encoding="utf-8") as f:
        for s in contrast:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    dt_c = Counter(s["expected_dispatch_type"] for s in contrast)
    print(f"  {len(contrast)} contrast samples: {dict(dt_c)}")

    # Merge
    print("\n--- Phase 2b: Merge into Train ---")
    tp = V07_DATA / "train_v07_dispatch.jsonl"
    tl = tp.read_text(encoding="utf-8").strip().split("\n")
    cl = cp.read_text(encoding="utf-8").strip().split("\n")
    tp.write_text("\n".join(cl + tl) + "\n", encoding="utf-8")
    print(f"  Train: {len(tl)} + {len(cl)} = {len(tl) + len(cl)}")

    # Phase 3
    print("\n--- Phase 3: Format Smoke ---")
    smoke = generate_format_smoke()
    sp = V07_DATA / "format_smoke_v07.jsonl"
    with sp.open("w", encoding="utf-8") as f:
        for s in smoke:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  {len(smoke)} format smoke samples")

    # Phase 4: manifest
    print("\n--- Phase 4: Manifest ---")
    cr = lambda p: sum(1 for l in p.open("r", encoding="utf-8") if l.strip()) if p.exists() else 0
    tn, vn, sn, fn = cr(V07_DATA/"train_v07_dispatch.jsonl"), cr(V07_DATA/"val_v07_dispatch.jsonl"), cr(V07_DATA/"smoke_v07_dispatch.jsonl"), cr(V07_DATA/"format_smoke_v07.jsonl")
    manifest = {
        "version": "0.7", "profile": "agent_dispatch_training", "schema": "dispatch_chat_messages",
        "total": tn+vn+sn, "train": tn, "val": vn, "smoke": sn, "format_smoke": fn,
        "v07_changes": [
            "3-layer schema: dispatch_type / workflow_id / tool completely separated",
            "no_evidence rows: required_tools forced to []",
            "contrast samples: single_tool vs workflow_chain for same building",
            "recommend_adaptive_strategies clarified as tool name only",
            "document_search_dci clarified as workflow, not dispatch_type",
            "JSON grammar for constrained decoding",
        ],
    }
    mp = V07_DATA / "harness_v07_manifest.json"
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  train={tn} val={vn} smoke={sn} format_smoke={fn}")

    # Phase 5: grammar
    print("\n--- Phase 5: JSON Grammar ---")
    gp = V07_ROOT / "grammar"
    gp.mkdir(parents=True, exist_ok=True)
    (gp / "dispatch_schema.gbnf").write_text(cfg.JSON_GRAMMAR, encoding="utf-8")
    (gp / "dispatch_json_schema.json").write_text(json.dumps(cfg.DISPATCH_JSON_SCHEMA, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  grammar/dispatch_schema.gbnf + dispatch_json_schema.json")

    print("\n" + "=" * 60)
    print("V7 PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
