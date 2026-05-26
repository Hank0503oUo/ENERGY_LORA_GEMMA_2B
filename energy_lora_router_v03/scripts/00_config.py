"""Shared config for harness router-strict LoRA data pipeline (v03 edition).

Single source of truth for:
- valid tool list + Chinese descriptions (used by augmenter teacher prompt)
- canonical system prompt with the choose-one-from-list constraint
- file paths

This is a copy of v02/scripts/00_config.py with paths re-pointed at v03.
Tool catalogue + BOUNDARY_RULES are kept identical to v02 so transferring
v02 training data into v03 doesn't shift the system prompt.

Other scripts import from here so the system prompt can never go out of sync
between train / val / smoke / augmenter prompts.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DRIVE_ROOT = Path("/content/drive/MyDrive/energy_lora_router_v03")
# Local-Windows fallback so these scripts can be tested without Colab too.
if not DRIVE_ROOT.exists():
    DRIVE_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = DRIVE_ROOT / "data" / "processed"
TRAIN_FILE = DATA_DIR / "train_v03.jsonl"
VAL_FILE = DATA_DIR / "val_v03.jsonl"
SMOKE_FILE = DATA_DIR / "smoke_v03.jsonl"
MANIFEST_FILE = DATA_DIR / "harness_v03_manifest.json"

BACKUP_DIR = DRIVE_ROOT / "data" / "backup"

# ---------------------------------------------------------------------------
# Valid tool catalogue
#
# Order matters: we render the system prompt with this order so the model
# sees a stable list across all training samples. Adding a tool ⇒ append to
# the end (don't reshuffle, that breaks token-level memorisation).
# ---------------------------------------------------------------------------

TOOLS: list[tuple[str, str]] = [
    ("query_energy_records",            "查詢特定建築或全校特定年份的歷年用電資料；含建築名稱、年份、EUI、R²、面積、平均功率等資料查詢時優先選此"),
    ("compare_energy_usage",            "全校或建築的跨年/跨月用電差異比較；問題含 vs、差多少、月度比較、年增率、用電占比時選此"),
    ("compare_building_trends",         "單一或多棟建築的多年趨勢比較；問題含趨勢、歷年變化、peak_kw/mean_kw/EUI 變化且主體是建築時選此"),
    ("compare_actual_predicted",        "比較實際用電 vs 預測值，看誤差"),
    ("generate_meter_chart",            "從電表 CSV 產生折線/長條/比較圖"),
    ("search_docs",                     "查詢 HJPLUS 法規、建築/能源相關文件；問題含法規、條款、消防、排煙、避難、綠建築、建築執照時選此"),
    ("run_counterfactual_for_building", "對單棟建築跑「如果改變 X 會省多少/增加多少」的快速反事實模擬；關鍵詞包含如果、假設、調高溫度、照明比例、人員/設備比例"),
    ("run_openbse_hybrid_counterfactual","用 OpenBSE/物理引擎驗證單棟建築已明確指定的物理情境；問題明講 OpenBSE、物理模型、物理引擎、HVAC 模擬時選此，不用於 ROI/全校投資排序"),
    ("openbse_hvac_breakdown",          "OpenBSE 分解 HVAC 各部分（冰水主機、冷卻水塔、AHU、泵）能耗；需使用者明講 HVAC 分解、空調分解時選此"),
    ("seasonal_strategies",             "僅針對季節性節能策略；問題明確包含夏季、冬季、過渡季、四季、不同季節、夏天空調、冬天照明時選此"),
    ("recommend_adaptive_strategies",   "通用節能策略推薦，適用全年、不分季節；問題含節能建議、改善方案、策略推薦但未指定夏季/冬季/過渡季時選此"),
    ("optimize_energy_portfolio",       "多建築節能投資組合最佳化與 ROI/預算排序；問題含全校、預算、哪幾棟、投資、ROI、優先順序、最佳組合時選此"),
    ("get_top_energy_buildings",        "用電 top N 排行榜（單一年度）"),
    ("rank_energy_buildings_across_years","跨年度跨建築排名"),
    ("detect_energy_anomalies",         "偵測異常用電（突波、停機、漂移）"),
    ("classify_anomaly",                "把異常波形分類成 zero_flatline / spike / oscillation / step_change"),
    ("diagnose_energy_anomaly",         "深入診斷單一異常事件的可能原因"),
    ("validate_strategy_openbse",       "用 OpenBSE 物理模擬驗證已採用策略的實際節能"),
    ("run_pvid",                        "跑 PI-VD 短期負載預測"),
    ("calibrate_sensitivity",           "依實測回灌校準預測模型靈敏度"),
    ("get_sensitivity_status",          "查詢目前靈敏度校準狀態"),
    ("record_strategy",                 "記錄已採用的節能策略以便後續追蹤"),
    ("confirm_strategy_adoption",       "確認某策略是否已被採用"),
    ("check_strategy_status",           "查詢策略目前狀態（已採用 / 進行中 / 取消）"),
    ("correlate_algorithms",            "比較不同預測演算法的關聯性"),
    ("list_campus_stats",               "列出全校建築數量、類型分佈、總用電、平均 EUI 等概況統計；不含特定建築、特定年份、趨勢或比較查詢"),
    ("list_rtem_sources",               "列出 RTEM/BMS/電表資料來源清單；只在使用者問有哪些資料源、感測器來源、BMS 清單時使用"),
    ("map_energy_semantics",            "將既有建築/電表/感測器欄位對應到 Haystack 或 Brick-lite 語意標籤；不是一般用電查詢、不是資料來源清單"),
]

REFUSAL = "__refusal__"

VALID_TOOL_NAMES: set[str] = {name for name, _ in TOOLS} | {REFUSAL}


BOUNDARY_RULES: list[str] = [
    "list_campus_stats 只回答全校概況統計；若有特定建築、年份、EUI、用電量或趨勢，改選 query_energy_records/compare_*。",
    "query_energy_records 是資料查詢工具；含建築名稱、年度、EUI、R²、面積、平均功率、全校特定年度用電時優先選此。",
    "compare_energy_usage 用於跨年、跨月、占比、差多少、vs、年增率；compare_building_trends 用於建築多年趨勢。",
    "seasonal_strategies 僅限夏季/冬季/過渡季/四季；一般節能建議選 recommend_adaptive_strategies。",
    "run_counterfactual_for_building 是單棟建築快速 what-if；run_openbse_hybrid_counterfactual 是明確要求 OpenBSE/物理模型驗證。",
    "optimize_energy_portfolio 是多棟/全校預算、ROI、投資排序；不要用它回答單棟假設情境。",
    "map_energy_semantics 是語意標籤對應；list_rtem_sources 是列資料來源；一般用電查詢不要選這兩個。",
    "search_docs 處理法規/文件查詢；不要因為涉及法規就選 __refusal__。",
    "系統沒有電費、水費、瓦斯、停車、醫院或校外資料時，回覆 __refusal__。",
]


def render_system_prompt() -> str:
    """Canonical system message used by every training sample (Step 1 fix).

    The model sees the full list of legal tools, so when it hallucinates
    `energy_data_tool` we have a strong signal that says 'this is not in
    the list'. Combined with enough samples per tool, this should bring
    val accuracy out of the 21% pit.
    """
    lines = [
        "你是 NTU 校園能源助理。你的唯一任務是判斷使用者問題應該呼叫哪個工具。",
        "",
        "# 工具邊界規則",
    ]
    for i, rule in enumerate(BOUNDARY_RULES, 1):
        lines.append(f"{i}. {rule}")
    lines.extend([
        "",
        "# 合法工具（從以下其中之一選一個，不可自創）",
    ])
    for name, desc in TOOLS:
        lines.append(f"- `{name}`: {desc}")
    lines.append(f"- `{REFUSAL}`: 問題模糊、超出能源領域、或需使用者進一步說明時使用")
    lines.extend([
        "",
        "# 回覆規則",
        '1. 必須輸出單一 JSON 物件：`{"tool": "<上述清單中某個名稱>", "arguments": {...}}`',
        "2. tool 名稱必須與清單**完全一致**（包含底線、大小寫），不可意譯、翻譯、縮寫",
        "3. 不可在 JSON 外加任何文字、不可加 markdown code fence",
        "4. arguments 用工具實際需要的欄位（如 buildings、years、metric、chart_type）",
        "5. 模糊或越界一律 `{\"tool\": \"__refusal__\", \"reason\": \"<簡短原因>\"}`",
        "6. 所有數字必須來自工具回傳值，不可憑空捏造",
        "7. 使用繁體中文（除了 tool 名稱與英數欄位）",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Eval gates (mirrors manifest.eval_gates so we can validate locally)
# ---------------------------------------------------------------------------

EVAL_GATES = {
    "tool_accuracy_min": 0.80,
    "malformed_json_max": 0.05,
    "hard_trap_accuracy_min": 0.60,
}


if __name__ == "__main__":
    # Quick self-check
    print(f"DRIVE_ROOT = {DRIVE_ROOT}")
    print(f"DATA_DIR exists: {DATA_DIR.exists()}")
    print(f"# tools = {len(TOOLS)} (+ refusal = {len(VALID_TOOL_NAMES)})")
    print(f"system prompt length = {len(render_system_prompt())} chars")
    print()
    print(render_system_prompt())
