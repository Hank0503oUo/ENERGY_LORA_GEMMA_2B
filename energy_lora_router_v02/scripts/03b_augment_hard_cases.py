"""Generate hard cases (trap, malformed, hard) to rebalance difficulty distribution.

This script generates:
1. TRAP: Semantic traps that look like tool X but should use tool Y
2. MALFORMED: Poorly formatted, ambiguous, or typo-laden queries
3. HARD: Complex, multi-step, boundary-adjacent queries

Usage:
    export GEMINI_API_KEY=...
    python scripts/03b_augment_hard_cases.py --difficulty trap --count 100
    python scripts/03b_augment_hard_cases.py --difficulty malformed --count 80
    python scripts/03b_augment_hard_cases.py --difficulty hard --count 60
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util as _ilu
_cfg_spec = _ilu.spec_from_file_location("cfg00", Path(__file__).resolve().parent / "00_config.py")
_cfg = _ilu.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg)

TRAIN_FILE = _cfg.TRAIN_FILE
DATA_DIR = _cfg.DATA_DIR
SYNTH_DIR = DATA_DIR / "synth"
TOOLS = _cfg.TOOLS
SYSTEM_PROMPT = _cfg.render_system_prompt()
VALID_TOOL_NAMES = _cfg.VALID_TOOL_NAMES


# ---------------------------------------------------------------------------
# Teacher prompts for hard cases
# ---------------------------------------------------------------------------

TRAP_TEMPLATE = """你是 NTU 校園能源助理測試題出題者。

# 任務：生成語義陷阱題目

你的目標是生成會「看起來像某個工具，但實際上應該用另一個工具」的台灣繁體中文 user query。
這些題目用來測試模型能否正確區分難以區分的工具邊界。

常見的陷阱對：
- seasonal_strategies vs recommend_adaptive_strategies：季節 vs 通用建議
- run_counterfactual_for_building vs run_openbse_hybrid_counterfactual：快速what-if vs 物理模型
- compare_energy_usage vs compare_building_trends：跨年/月差異 vs 趨勢
- query_energy_records vs list_campus_stats：特定建築 vs 全校概況
- search_docs vs 其他工具：法規/文件 vs 數據分析

# 要求
1. 每個 query 都應該包含「可能誤導」的關鍵詞，但實際上應該用不同的工具
2. query 應該自然、像真實使用者會打的句子
3. 涵蓋不同 NTU 建築與場景
4. 共生成 {count} 個陷阱題目

# 輸出格式
只輸出 JSONL（每行一個 JSON object），格式：
{{"user": "<misleading query>", "expected_tool": "<correct_tool>"}}

不要有編號、不要 markdown code fence、不要任何解釋文字。直接從第一行開始輸出 {count} 筆 JSONL。"""


MALFORMED_TEMPLATE = """你是 NTU 校園能源助理測試題出題者。

# 任務：生成格式不規範的題目

生成語法錯誤、語意模糊、簡寫不規範、含多個相互矛盾要求的台灣繁體中文 query。
這些題目用來測試模型面對真實使用者輸入的魯棒性。

例子：
- 打字錯誤：「查一下保健中心去年用電情況喔」(中文打字常見錯誤)
- 語意模糊：「幫我看一下過去」(不清楚是哪一年/建築)
- 混合要求：「我要看全校用電跟單棟趨勢並比較」
- 簡寫/口語：「AT1001 怎樣？」「啊到底咧」
- 非常簡短：「EUI?」「去年?」

# 要求
1. 保持真實性——這些是實際使用者可能打出來的
2. 涵蓋各種「不規範」類型
3. 仍然應該能對應到某個有效工具（不要完全無法理解）
4. 共生成 {count} 個題目

# 輸出格式
只輸出 JSONL（每行一個 JSON object），格式：
{{"user": "<malformed_query>", "expected_tool": "<tool>"}}

不要有編號、不要 markdown code fence、不要任何解釋文字。直接從第一行開始輸出 {count} 筆 JSONL。"""


HARD_TEMPLATE = """你是 NTU 校園能源助理測試題出題者。

# 任務：生成複雜邊界題目

生成需要對工具邊界有深入理解、涉及多步驟或邊界情況的 query。
這些題目測試模型是否真正理解工具的適用範圍，而不只是簡單的關鍵詞匹配。

例子：
- 邊界情況：「我想看 2015-2019 的平均用電，然後跟這個季節的平均比較」(時間跨度 + 季節策略)
- 隱含的全校推理：「哪棟建築去年夏天表現最好，但今年夏天下降最多」(排名 + 趨勢 + 季節)
- 複雜的反事實：「如果調整通風在保持舒適度的前提下，會不會比改照明效益還好」(多變數what-if)
- 邊界的否定：「這不是季節性，那怎麼辦」(__refusal__ vs 其他工具的判斷)

# 要求
1. 每個 query 都應該需要「思考」工具選擇，而不是直接的關鍵詞匹配
2. 可以涉及隱含的多步驟邏輯
3. 保持語言自然、真實
4. 共生成 {count} 個題目

# 輸出格式
只輸出 JSONL（每行一個 JSON object），格式：
{{"user": "<complex_query>", "expected_tool": "<tool>"}}

不要有編號、不要 markdown code fence、不要任何解釋文字。直接從第一行開始輸出 {count} 筆 JSONL。"""


def call_gemini(prompt: str, model_name: str = "gemini-2.0-flash-exp") -> str:
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt, generation_config={"temperature": 0.9, "max_output_tokens": 4096})
    return resp.text or ""


JSON_LINE_RE = re.compile(r'^\s*\{.*"user".*"expected_tool".*\}\s*$')


def parse_teacher_output(text: str, difficulty: str, existing_users: set[str]) -> tuple[list[dict], dict]:
    accepted: list[dict] = []
    stats = Counter()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("```"):
            stats["fence_skipped"] += 1
            continue
        if not JSON_LINE_RE.match(line):
            stats["non_json_line"] += 1
            continue
        try:
            obj = json.loads(line)
        except Exception:
            stats["json_decode_error"] += 1
            continue
        user = (obj.get("user") or "").strip()
        tool = obj.get("expected_tool")
        if not user or len(user) < 4:
            stats["too_short"] += 1
            continue
        if tool not in VALID_TOOL_NAMES:
            stats["unknown_tool"] += 1
            continue
        if user in existing_users:
            stats["duplicate_user"] += 1
            continue
        existing_users.add(user)
        # Add difficulty field
        obj["difficulty"] = difficulty
        accepted.append(obj)
        stats["accepted"] += 1
    return accepted, dict(stats)


def load_existing_train_distribution() -> set[str]:
    users: set[str] = set()
    with TRAIN_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            for m in obj.get("messages", []):
                if m.get("role") == "user":
                    users.add(m["content"].strip())
    return users


def cmd_print_prompt(args) -> None:
    """Write prompt to file so you can run it with any CLI."""
    templates = {
        "trap": TRAP_TEMPLATE,
        "malformed": MALFORMED_TEMPLATE,
        "hard": HARD_TEMPLATE,
    }
    template = templates[args.difficulty]
    prompt = template.format(count=args.count)
    
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    prompt_path = SYNTH_DIR / f"hard_cases_{args.difficulty}_{args.count}.prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    print(f"# wrote prompt -> {prompt_path}")
    print(f"# Run with any CLI (e.g., opencode, DeepSeek, Claude):")
    print(f"#   opencode run --model gpt-4o < {prompt_path} > {prompt_path.with_suffix('.txt')}")
    print(f"# Then parse responses:")
    print(f"#   python scripts/03b_augment_hard_cases.py --parse-response {prompt_path.with_suffix('.txt')} --difficulty {args.difficulty}")


def cmd_parse_response(args) -> None:
    """Parse CLI response and write JSONL."""
    response_file = Path(args.parse_response)
    if not response_file.is_absolute():
        response_file = (_cfg.DRIVE_ROOT / response_file).resolve()
    if not response_file.exists():
        raise FileNotFoundError(f"Response file not found: {response_file}")
    
    text = response_file.read_text(encoding="utf-8")
    existing_users = load_existing_train_distribution()
    
    accepted, stats = parse_teacher_output(text, args.difficulty, existing_users)
    print(f"# stats: {dict(stats)}")
    print(f"# accepted {len(accepted)} samples")
    print()
    
    if not accepted:
        print("# no valid samples generated")
        return
    
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SYNTH_DIR / f"hard_cases_{args.difficulty}_{stamp}.jsonl"
    
    with out_path.open("w", encoding="utf-8") as f:
        for r in accepted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"# wrote {len(accepted)} samples -> {out_path}")
    print()
    print("# next:")
    print(f"#   python scripts/04_merge_and_rebuild_manifest.py --source {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--difficulty", choices=["trap", "malformed", "hard"], required=True)
    p.add_argument("--count", type=int, default=50)
    p.add_argument("--print-prompt", action="store_true", help="write prompt to file")
    p.add_argument("--parse-response", metavar="FILE", default=None, help="parse CLI response from FILE")
    p.add_argument("--dry-run", action="store_true", help="print prompt, don't call API")
    args = p.parse_args()

    if args.print_prompt:
        cmd_print_prompt(args)
        return
    
    if args.parse_response:
        cmd_parse_response(args)
        return

    templates = {
        "trap": TRAP_TEMPLATE,
        "malformed": MALFORMED_TEMPLATE,
        "hard": HARD_TEMPLATE,
    }
    template = templates[args.difficulty]
    prompt = template.format(count=args.count)

    if args.dry_run:
        print(f"# dry-run: {args.difficulty} x {args.count}")
        print(prompt)
        return

    print(f"# augment_hard_cases (difficulty={args.difficulty}, count={args.count})")
    print()

    existing_users = load_existing_train_distribution()
    print(f"# existing train users: {len(existing_users)}")
    print()

    try:
        text = call_gemini(prompt)
    except Exception as exc:
        print(f"! API error: {exc}")
        return

    accepted, stats = parse_teacher_output(text, args.difficulty, existing_users)
    print(f"# stats: {dict(stats)}")
    print(f"# accepted {len(accepted)} samples")
    print()

    if not accepted:
        print("# no valid samples generated")
        return

    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SYNTH_DIR / f"hard_cases_{args.difficulty}_{stamp}.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for r in accepted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"# wrote {len(accepted)} samples -> {out_path}")
    print()
    print("# next:")
    print(f"#   python scripts/04_merge_and_rebuild_manifest.py --source {out_path}")


if __name__ == "__main__":
    main()
