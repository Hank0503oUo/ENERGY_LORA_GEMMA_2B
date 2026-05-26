"""V8 full pipeline — boundary repair round.

Key fixes from v7 eval:
  - clarify_needed over-triggered (30 workflow→clarify, 12 single→clarify)
  - no_evidence still 0/21
  - workflow_chain vs single_tool unstable

Pipeline:
  1. Reprompt v07 data with v08 decision-rubric prompt
  2. Fix schema: no_evidence required_tools=[], fix clarify over-trigger in data
  3. Add boundary-repair contrast samples
  4. Format smoke test
  5. Manifest + grammar
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from collections import Counter

import importlib.util

_here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v08cfg", _here / "00_config_v08.py")
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)
NEW_PROMPT = cfg.render_system_prompt()

V07_DATA = Path(r"G:\我的雲端硬碟\energy_lora_router_v07\data\processed")
V08_DATA = Path(r"G:\我的雲端硬碟\energy_lora_router_v08\data\processed")
V08_SYNTH = Path(r"G:\我的雲端硬碟\energy_lora_router_v08\data\synth")
V08_ROOT = Path(r"G:\我的雲端硬碟\energy_lora_router_v08")

B = ["博理館","明達館","新博士生宿舍","電機一館","電機二館","資工系館","工學院綜合大樓","共同教室館","新生大樓","應力館"]
Y = [2017,2018,2019,2020,2021,2022,2023]

def _l(b=None,y=None,m=None): return {"building_names":b or [],"years":y or [],"metrics":m or []}
def _t(t,p): return {"tool":t,"purpose":p}
def _d(dt,wid,ans,le,tools,sc): return {"dispatch_type":dt,"workflow_id":wid,"answerability":ans,"locked_entities":le,"required_tools":tools,"stop_conditions":sc}
def _r(sid,q,d,tags):
    return {"messages":[{"role":"system","content":NEW_PROMPT},{"role":"user","content":q},{"role":"assistant","content":json.dumps(d,ensure_ascii=False)}],
            "sample_id":sid,"user_role":"campus_manager","expected_dispatch_type":d["dispatch_type"],
            "expected_workflow_id":d["workflow_id"],"expected_answerability":d["answerability"],"difficulty":"easy","tags":tags}

def reprompt_and_fix(src, dst):
    total=patched=fixed_ne=fixed_clarify=0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r",encoding="utf-8") as fin, dst.open("w",encoding="utf-8") as fout:
        for line in fin:
            line=line.strip()
            if not line: continue
            total+=1
            row=json.loads(line)
            msgs=row.get("messages",[])
            if msgs and msgs[0].get("role")=="system":
                msgs[0]["content"]=NEW_PROMPT; patched+=1
            if msgs and msgs[-1].get("role")=="assistant":
                try:
                    p=json.loads(msgs[-1]["content"])
                    dt=p.get("dispatch_type","")
                    if dt=="no_evidence" and p.get("required_tools"):
                        p["required_tools"]=[]; msgs[-1]["content"]=json.dumps(p,ensure_ascii=False); fixed_ne+=1
                except: pass
            fout.write(json.dumps(row,ensure_ascii=False)+"\n")
    return total, patched, fixed_ne


def generate_boundary_samples(count=200):
    rng=random.Random(42); samples=[]; idx=0

    # 1. NOT clarify_needed: tasks WITH building name → workflow_chain
    for bldg in B[:6]:
        for q, wid, tools in [
            (f"{bldg} 的節能改造優先順序", "building_strategy_plan",
             [_t("query_energy_records","查用電"),_t("get_top_energy_buildings","排名"),_t("recommend_adaptive_strategies","策略")]),
            (f"{bldg} 有哪些可行的節能方案", "building_hotspot_improvement",
             [_t("query_energy_records","查用電"),_t("detect_energy_anomalies","偵測異常"),_t("recommend_adaptive_strategies","改善建議")]),
            (f"夏季 {bldg} 的節能策略", "building_strategy_plan",
             [_t("query_energy_records","查用電基準"),_t("recommend_adaptive_strategies","策略")]),
            (f"{bldg} 照明降 30% 一年省多少", "counterfactual_saving_estimate",
             [_t("query_energy_records","基準用電"),_t("run_openbse_hybrid_counterfactual","模擬")]),
        ]:
            d=_d("workflow_chain",wid,"answerable_multi_tool",_l(b=[bldg]),tools,
                 ["if_energy_values_missing_stop_before_strategy"])
            samples.append(_r(f"v08b_{idx:04d}",q,d,["boundary","not_clarify"]))
            idx+=1

    # 2. NOT clarify_needed: document search → workflow_chain
    for q_text,m in [("CV-RMSE 的定義","cv-rmse"),("PI-VD 模型說明","pi-vd"),("OpenBSE 技術手冊","openbse"),
                      ("節能法規條文","regulation"),("HVAC 設計規範","hvac"),("ASHP 性能係數","ashp"),
                      ("EUI 計算方式","eui"),("冰水主機選型","chiller")]:
        d=_d("workflow_chain","document_search_dci","answerable_multi_tool",_l(m=[m]),
             [_t("find_docs",f"搜尋{q_text}"),_t("grep_docs","搜尋定義")],
             ["if_no_document_match_report_no_evidence"])
        samples.append(_r(f"v08b_{idx:04d}",f"{q_text}在哪份文件",d,["boundary","not_clarify_doc"]))
        idx+=1

    # 3. NOT clarify_needed: single_tool with building
    for bldg in B[:5]:
        for q, wid, tool in [
            (f"{bldg} 用電","single_building_year_status",_t("query_energy_records","查用電")),
            (f"{bldg} 有異常嗎","none",_t("detect_energy_anomalies","偵測異常")),
            (f"幫{bldg}推薦節能策略","none",_t("recommend_adaptive_strategies","策略")),
            (f"{bldg} HVAC 分解","none",_t("openbse_hvac_breakdown","HVAC拆解")),
        ]:
            d=_d("single_tool",wid,"answerable_single_tool",_l(b=[bldg]),[tool],[])
            samples.append(_r(f"v08b_{idx:04d}",q,d,["boundary","not_clarify_single"]))
            idx+=1

    # 4. YES clarify_needed: truly ambiguous / missing info
    clarify_qs = [
        ("然後呢","ambiguous_reference"),("幫我看一下","missing_required_arguments"),
        ("節能方法","missing_required_arguments"),("用電","missing_required_arguments"),
        ("那棟建築的用電","ambiguous_reference"),("去年的用電","missing_required_arguments"),
        ("比較一下","missing_required_arguments"),("幫我查一下","missing_required_arguments"),
        ("那個建築呢","ambiguous_reference"),("我要查資料","missing_required_arguments"),
        ("異常","ambiguous_reference"),("今天","ambiguous_reference"),
    ]
    for q, ans in clarify_qs:
        d=_d("clarify_needed","none",ans,_l(),[],
             ["if_no_prior_context_ask_clarification"] if "ambiguous" in ans else ["if_required_arguments_missing_ask_clarification"])
        samples.append(_r(f"v08b_{idx:04d}",q,d,["boundary","yes_clarify"]))
        idx+=1

    # 5. no_evidence: building+year but year out of range
    for bldg,yr in [("博理館",2012),("明達館",2010),("電機一館",2008),("共同教室館",2005),
                     ("資工系館",2015),("新生大樨",2009),("博理館",2013),("明達館",2011)]:
        d=_d("no_evidence","none","no_evidence_expected",_l(b=[bldg],y=[yr]),[],
             ["if_energy_values_missing_report_missing_not_zero"])
        samples.append(_r(f"v08b_{idx:04d}",f"{bldg} {yr} 年用電",d,["boundary","no_evidence"]))
        idx+=1

    # 6. no_evidence vs refusal contrast
    for q, exp in [("查瓦斯","refusal"),("查天然氣","refusal"),("博理館 2015 用電","no_evidence"),
                    ("台大醫院用電","refusal"),("交大用電","refusal"),("明達館 2007 EUI","no_evidence")]:
        if exp=="refusal":
            d=_d("refusal","none","unsupported_scope",_l(),[],["if_unsupported_scope_refuse_without_tool"])
        else:
            d=_d("no_evidence","none","no_evidence_expected",_l(b=["明達館"],y=[2007]),[],
                 ["if_energy_values_missing_report_missing_not_zero"])
        samples.append(_r(f"v08b_{idx:04d}",q,d,["boundary","ne_vs_refusal"]))
        idx+=1

    rng.shuffle(samples)
    return samples[:count]


def generate_format_smoke():
    samples=[]; idx=0
    tests=[
        ("博理館 2020 用電","single_tool","single_building_year_status","answerable_single_tool",
         _l(b=["博理館"],y=[2020]),[_t("query_energy_records","查用電")],["if_tool_result_building_mismatch_stop"]),
        ("博理館 2020 用電狀況跟改善","workflow_chain","building_hotspot_improvement","answerable_multi_tool",
         _l(b=["博理館"],y=[2020]),
         [_t("query_energy_records","查用電"),_t("detect_energy_anomalies","偵測"),_t("recommend_adaptive_strategies","改善")],
         ["if_energy_values_missing_stop_before_strategy"]),
        ("博理館節能策略規劃","workflow_chain","building_strategy_plan","answerable_multi_tool",
         _l(b=["博理館"]),
         [_t("query_energy_records","查用電"),_t("get_top_energy_buildings","排名"),_t("recommend_adaptive_strategies","策略")],
         ["if_energy_values_missing_stop_before_strategy"]),
        ("如果博理館空調調高2度省多少","workflow_chain","counterfactual_saving_estimate","answerable_multi_tool",
         _l(b=["博理館"]),[_t("query_energy_records","基準"),_t("run_openbse_hybrid_counterfactual","模擬")],
         ["if_energy_values_missing_stop_before_strategy"]),
        ("CV-RMSE 定義在哪份文件","workflow_chain","document_search_dci","answerable_multi_tool",
         _l(m=["cv-rmse"]),[_t("find_docs","搜尋"),_t("grep_docs","搜尋")],["if_no_document_match_report_no_evidence"]),
        ("PI-VD 模型說明文件","workflow_chain","document_search_dci","answerable_multi_tool",
         _l(m=["pi-vd"]),[_t("search_docs","搜尋文件")],["if_no_document_match_report_no_evidence"]),
        ("幫博理館推薦節能策略","single_tool","none","answerable_single_tool",
         _l(b=["博理館"]),[_t("recommend_adaptive_strategies","策略")],[]),
        ("博理館 2012 年用電","no_evidence","none","no_evidence_expected",
         _l(b=["博理館"],y=[2012]),[],["if_energy_values_missing_report_missing_not_zero"]),
        ("明達館 2008 EUI","no_evidence","none","no_evidence_expected",
         _l(b=["明達館"],y=[2008]),[],["if_energy_values_missing_report_missing_not_zero"]),
        ("查天然氣","refusal","none","unsupported_scope",_l(),[],["if_unsupported_scope_refuse_without_tool"]),
        ("幫我隱藏異常","refusal","none","unsafe_operation",_l(),[],["if_unsafe_operation_refuse_without_tool"]),
        ("然後呢","clarify_needed","none","ambiguous_reference",_l(),[],["if_no_prior_context_ask_clarification"]),
        ("去年的用電","clarify_needed","none","missing_required_arguments",_l(),[],["if_required_arguments_missing_ask_clarification"]),
        ("全校用電前5名各自的節能建議","workflow_chain","campus_top_energy_buildings","answerable_multi_tool",
         _l(),[_t("get_top_energy_buildings","排行"),_t("recommend_adaptive_strategies","策略")],[]),
        ("全校概況","single_tool","none","answerable_single_tool",_l(),[_t("list_campus_stats","概況")],[]),
        ("交大用電","refusal","none","unsupported_scope",_l(),[],["if_unsupported_scope_refuse_without_tool"]),
    ]
    for q,dt,wid,ans,le,tools,sc in tests:
        idx+=1
        samples.append(_r(f"fmt_v08_{idx:03d}",q,_d(dt,wid,ans,le,tools,sc),["format_smoke_v08"]))
    return samples


def main():
    print("="*60+"\nV8 BOUNDARY REPAIR PIPELINE\n"+"="*60)

    print("\n--- Phase 1: Reprompt + Fix ---")
    for sn,dn in [("train_v07_dispatch.jsonl","train_v08_dispatch.jsonl"),
                  ("val_v07_dispatch.jsonl","val_v08_dispatch.jsonl"),
                  ("smoke_v07_dispatch.jsonl","smoke_v08_dispatch.jsonl")]:
        src,dst = V07_DATA/sn, V08_DATA/dn
        if not src.exists(): print(f"  SKIP {sn}"); continue
        t,p,f = reprompt_and_fix(src,dst)
        print(f"  {dn}: {t} rows, {f} no_evidence fixed")

    print("\n--- Phase 2: Boundary Samples ---")
    bs = generate_boundary_samples(200)
    V08_SYNTH.mkdir(parents=True, exist_ok=True)
    bp = V08_SYNTH / "v08_boundary_curriculum.jsonl"
    with bp.open("w",encoding="utf-8") as f:
        for s in bs: f.write(json.dumps(s,ensure_ascii=False)+"\n")
    dt_c = Counter(s["expected_dispatch_type"] for s in bs)
    print(f"  {len(bs)} boundary samples: {dict(dt_c)}")

    print("\n--- Phase 2b: Merge ---")
    tp = V08_DATA / "train_v08_dispatch.jsonl"
    tl = tp.read_text(encoding="utf-8").strip().split("\n")
    bl = bp.read_text(encoding="utf-8").strip().split("\n")
    tp.write_text("\n".join(bl+tl)+"\n", encoding="utf-8")
    print(f"  Train: {len(tl)} + {len(bl)} = {len(tl)+len(bl)}")

    print("\n--- Phase 3: Format Smoke ---")
    smoke = generate_format_smoke()
    sp = V08_DATA / "format_smoke_v08.jsonl"
    with sp.open("w",encoding="utf-8") as f:
        for s in smoke: f.write(json.dumps(s,ensure_ascii=False)+"\n")
    print(f"  {len(smoke)} format smoke samples")

    print("\n--- Phase 4: Manifest ---")
    cr = lambda p: sum(1 for l in p.open("r",encoding="utf-8") if l.strip()) if p.exists() else 0
    tn,vn,sn2,fn = cr(V08_DATA/"train_v08_dispatch.jsonl"),cr(V08_DATA/"val_v08_dispatch.jsonl"),cr(V08_DATA/"smoke_v08_dispatch.jsonl"),cr(V08_DATA/"format_smoke_v08.jsonl")
    manifest = {"version":"0.8","profile":"agent_dispatch_training","schema":"dispatch_chat_messages",
                "total":tn+vn+sn2,"train":tn,"val":vn,"smoke":sn2,"format_smoke":fn,
                "v08_changes":["decision rubric in prompt","clarify_needed only for truly ambiguous input",
                                "negative examples: tasks with building→workflow not clarify",
                                "no_evidence vs clarify_needed boundary table",
                                "workflow_chain vs single_tool decision table"]}
    mp = V08_DATA / "harness_v08_manifest.json"
    mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"  train={tn} val={vn} smoke={sn2} format_smoke={fn}")

    print("\n--- Phase 5: Grammar ---")
    gp = V08_ROOT / "grammar"; gp.mkdir(parents=True,exist_ok=True)
    (gp/"dispatch_schema.gbnf").write_text(cfg.JSON_GRAMMAR,encoding="utf-8")
    (gp/"dispatch_json_schema.json").write_text(json.dumps(cfg.DISPATCH_JSON_SCHEMA,ensure_ascii=False,indent=2),encoding="utf-8")
    print("  grammar/ updated")

    print("\n"+"="*60+"\nV8 PIPELINE COMPLETE\n"+"="*60)


if __name__=="__main__":
    main()
