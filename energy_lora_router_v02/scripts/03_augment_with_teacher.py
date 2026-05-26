"""Step 2: ask a teacher LLM to generate more training queries per tool.

Default backend = Google Gemini 2.0 Flash (free tier 1500 RPM, good Chinese).
Set GEMINI_API_KEY in the environment, or pass --provider openai / --provider anthropic
and set the matching OPENAI_API_KEY / ANTHROPIC_API_KEY.

Usage:
    pip install google-generativeai
    export GEMINI_API_KEY=...
    python scripts/03_augment_with_teacher.py --target 50

No-key BYO-CLI workflow:
    python scripts/03_augment_with_teacher.py --target 50 --print-prompts data/synth/prompts
    # Run data/synth/prompts/_run_with_cli.sh or feed prompts manually.
    python scripts/03_augment_with_teacher.py --parse-responses data/synth/responses

What it does:
    1. Reads current train.jsonl
    2. Counts existing samples per tool
    3. For every tool below TARGET, asks teacher LLM for `target - current` new queries
    4. Filters out malformed JSON / empty / duplicate / cross-tool keyword bleed
    5. Writes data/synth/gemini_augmented_<timestamp>.jsonl  (does NOT touch train.jsonl)
    6. Prints what to run next: scripts/04_merge_and_rebuild_manifest.py

We never auto-merge: a human should eyeball ~10% before adding to training.
"""
from __future__ import annotations

import argparse
import json
import os
import random
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
# Teacher prompt
# ---------------------------------------------------------------------------

TEACHER_TEMPLATE = """你是 NTU 校園能源助理測試題出題者。

# 目標 tool
名稱：`{tool_name}`
用途：{description}

# 任務
請生成 {n} 個會觸發這個 tool 的台灣繁體中文 user query。

# 要求
1. 每個 query 都應該獨立、自然、像真實使用者會打的句子（口語、半正式都行）
2. 涵蓋不同 phrasing：很短的（5-10 字）、中等長度、囉嗦的、有錯字的
3. 涵蓋不同 NTU 建築：保健中心(AT2045)、總圖書館(AT1001)、化學工程館(AT2007)、共同教學館(AT3010)、土木研究大樓(AT5043)、男一舍(AT8001)、體育館、活動中心、計算機中心、博理館、明達館、生科館、物理館等
4. 涵蓋不同年份範圍（2014-2020）— 不一定每筆都要寫年份
5. 各筆之間不要太相似（同義詞、語序、詳略要有差異）
6. **絕對不可** 包含其他 tool 的明顯關鍵詞（避免歧義） — 例如生 query_energy_records 時不可有「比較」「趨勢」「排名」「圖表」「異常」「節能」「策略」這些字
7. 不要產生會被歸類為 __refusal__ 的問題（即不要超出能源領域、不要太模糊）

# 輸出格式
只輸出 JSONL（每行一個 JSON object），格式：
{{"user": "<user query>", "expected_tool": "{tool_name}"}}

不要有編號、不要 markdown code fence、不要任何解釋文字。直接從第一行開始輸出 {n} 筆 JSONL。"""


# ---------------------------------------------------------------------------
# Cross-tool keyword bleed filter
# ---------------------------------------------------------------------------

# If a query for tool X contains keywords strongly associated with tool Y,
# we drop it to avoid teaching the model contradictory routing.
TOOL_KEYWORDS = {
    "compare_energy_usage":            ["比較", "對比", "vs", "差多少", "增減", "yoy", "年增"],
    "compare_building_trends":         ["趨勢", "變化", "走勢"],
    "rank_energy_buildings_across_years": ["跨年排名"],
    "get_top_energy_buildings":        ["排名", "排行", "top", "最高", "最低", "最耗電", "最省電"],
    "generate_meter_chart":            ["圖", "畫", "繪", "視覺化", "chart", "plot"],
    "search_docs":                     ["法規", "法律", "文件", "hjplus", "建築執照"],
    "run_counterfactual_for_building": ["反事實", "counterfactual", "假設", "如果", "省多少", "what-if"],
    "run_openbse_hybrid_counterfactual":["openbse", "物理模擬"],
    "openbse_hvac_breakdown":          ["hvac 分解", "空調分解"],
    "seasonal_strategies":             ["季節", "夏季", "冬季", "過渡季"],
    "recommend_adaptive_strategies":   ["節能策略", "改善建議", "節能建議"],
    "optimize_energy_portfolio":       ["全校預算", "投資組合", "roi", "資源配置", "哪幾棟"],
    "detect_energy_anomalies":         ["異常", "突波", "outlier", "abnormal"],
    "classify_anomaly":                ["分類", "歸類"],
    "diagnose_energy_anomaly":         ["診斷", "原因"],
}


def has_other_tool_keywords(query: str, target_tool: str) -> str | None:
    """Return name of conflicting tool if query has its keywords; else None."""
    q = query.lower()
    target_kws = set(TOOL_KEYWORDS.get(target_tool, []))
    for other_tool, kws in TOOL_KEYWORDS.items():
        if other_tool == target_tool:
            continue
        for kw in kws:
            if kw in target_kws:
                continue  # shared term, ignore
            if kw.lower() in q:
                return other_tool
    return None


# ---------------------------------------------------------------------------
# Teacher backends
# ---------------------------------------------------------------------------

def call_gemini(prompt: str, model_name: str = "gemini-2.0-flash-exp") -> str:
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt, generation_config={"temperature": 0.9, "max_output_tokens": 4096})
    return resp.text or ""


def call_openai(prompt: str, model_name: str = "gpt-4o-mini") -> str:
    from openai import OpenAI
    client = OpenAI()  # uses OPENAI_API_KEY env var
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=4096,
    )
    return resp.choices[0].message.content or ""


def call_anthropic(prompt: str, model_name: str = "claude-haiku-4-5-20251001") -> str:
    import anthropic
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    msg = client.messages.create(
        model=model_name,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if hasattr(b, "text"))


def call_yuhuanstudio(prompt: str, model_name: str = "DeepSeek-V4-Flash") -> str:
    """YuhuanStudio aggregator — OpenAI-compatible.

    Default model = DeepSeek-V4-Flash (cheap, very strong Chinese).
    Other usable models the aggregator exposes:
        DeepSeek-V4-Flash         (default — cheapest, fastest)
        DeepSeek-V4-Pro           (better, slower)
        glm-5                     (also strong Chinese)
        gemini-3-flash-preview    (Google's flash via aggregator)

    Set YUHUAN_API_KEY (or YUHUANSTUDIO_API_KEY) in env.
    """
    from openai import OpenAI
    api_key = os.environ.get("YUHUAN_API_KEY") or os.environ.get("YUHUANSTUDIO_API_KEY")
    if not api_key:
        raise RuntimeError("Set YUHUAN_API_KEY or YUHUANSTUDIO_API_KEY")
    base_url = os.environ.get("YUHUAN_BASE_URL", "https://api.yuhuanstudio.com/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=4096,
    )
    return resp.choices[0].message.content or ""


def call_modelscope(prompt: str, model_name: str = "deepseek-ai/DeepSeek-V4-Flash") -> str:
    """ModelScope — OpenAI-compatible, free tier available.

    Set MODELSCOPE_API_KEY (or MODELSCOPE_TOKEN) in env.
    """
    from openai import OpenAI
    api_key = os.environ.get("MODELSCOPE_API_KEY") or os.environ.get("MODELSCOPE_TOKEN")
    if not api_key:
        raise RuntimeError("Set MODELSCOPE_API_KEY or MODELSCOPE_TOKEN")
    base_url = os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=4096,
    )
    return resp.choices[0].message.content or ""


PROVIDERS = {
    "gemini":         call_gemini,
    "openai":         call_openai,
    "anthropic":      call_anthropic,
    "yuhuanstudio":   call_yuhuanstudio,
    "deepseek":       call_yuhuanstudio,    # alias — DeepSeek via YuhuanStudio is the recommended path
    "modelscope":     call_modelscope,
}


# ---------------------------------------------------------------------------
# Parsing & filtering
# ---------------------------------------------------------------------------

JSON_LINE_RE = re.compile(r'^\s*\{.*"user".*"expected_tool".*\}\s*$')


def parse_teacher_output(text: str, target_tool: str, existing_users: set[str]) -> tuple[list[dict], dict]:
    accepted: list[dict] = []
    stats = Counter()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # tolerate code fences
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
        if tool != target_tool:
            stats["wrong_tool_label"] += 1
            continue
        if user in existing_users:
            stats["duplicate_user"] += 1
            continue
        bleed = has_other_tool_keywords(user, target_tool)
        if bleed:
            stats[f"bleed_{bleed}"] += 1
            continue
        existing_users.add(user)
        accepted.append({"user": user, "expected_tool": target_tool})
        stats["accepted"] += 1
    return accepted, dict(stats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# BYO-CLI mode: print prompts / parse responses
# ---------------------------------------------------------------------------

def _safe_filename(tool: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", tool)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(_cfg.DRIVE_ROOT))
    except ValueError:
        return str(path)


def cmd_print_prompts(args) -> None:
    """Write one prompt-per-tool file you can feed to any external CLI."""
    out_dir = Path(args.print_prompts)
    if not out_dir.is_absolute():
        out_dir = (_cfg.DRIVE_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    resp_dir = out_dir.parent / "responses"

    dist, _ = load_existing_train_distribution()

    plan = []
    for tool, _desc in TOOLS:
        if args.only_tools and tool not in args.only_tools:
            continue
        if tool in args.skip_tools:
            continue
        current = dist.get(tool, 0)
        needed = max(0, args.target - current)
        if needed > 0:
            plan.append((tool, current, needed))

    print(f"# print-prompts mode -> {_display_path(out_dir)}")
    print(f"# {len(plan)} tool(s) need augmentation, target = {args.target} each\n")

    for tool, current, needed in plan:
        desc = next(d for n, d in TOOLS if n == tool)
        prompt = TEACHER_TEMPLATE.format(
            tool_name=tool, description=desc, n=min(needed, args.max_batch)
        )
        # We emit the chunked size (max-batch) — if you need more than that for
        # one tool, run the prompt multiple times and concatenate the outputs.
        path = out_dir / f"{_safe_filename(tool)}.prompt.txt"
        path.write_text(prompt, encoding="utf-8")
        print(f"  wrote {_display_path(path)}  "
              f"(current={current}, +{needed})")

    # Helper batch script
    runner = out_dir / "_run_with_cli.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "# Replace the CLI command below with your actual one. Examples:\n"
        "#   commandcode --model DeepSeek-V4-Flash < \"$f\" > \"$out\"\n"
        "#   opencode run --model DeepSeek-V4-Flash --prompt-file \"$f\" > \"$out\"\n"
        "#   codex --model DeepSeek-V4-Flash exec < \"$f\" > \"$out\"\n"
        "set -e\n"
        f'PROMPT_DIR="{out_dir.as_posix()}"\n'
        f'RESP_DIR="{resp_dir.as_posix()}"\n'
        'mkdir -p "$RESP_DIR"\n'
        'for f in "$PROMPT_DIR"/*.prompt.txt; do\n'
        '    name="$(basename "$f" .prompt.txt).txt"\n'
        '    out="$RESP_DIR/$name"\n'
        '    if [ -s "$out" ]; then\n'
        '        echo "skip (exists): $name"\n'
        '        continue\n'
        '    fi\n'
        '    echo "-> $name"\n'
        '    # EDIT THIS LINE: replace with your CLI command.\n'
        '    commandcode --model DeepSeek-V4-Flash < "$f" > "$out"\n'
        'done\n'
        'echo "done. Now: python scripts/03_augment_with_teacher.py "\\\n'
        f'     "--parse-responses {_display_path(resp_dir)}"\n',
        encoding="utf-8",
    )

    print()
    print(f"# next steps")
    print(f"# 1. Edit {_display_path(runner)} to use your actual CLI command")
    print(f"# 2. Run it (or feed each *.prompt.txt manually to your CLI)")
    print(f"# 3. Save responses as data/synth/responses/<tool>.txt")
    print(f"# 4. python scripts/03_augment_with_teacher.py --parse-responses {_display_path(resp_dir)}")


def cmd_parse_responses(args) -> None:
    """Walk DIR for CLI response files, extract JSONL rows, write augmented file."""
    in_dir = Path(args.parse_responses)
    if not in_dir.is_absolute():
        in_dir = (_cfg.DRIVE_ROOT / in_dir).resolve()
    if not in_dir.is_dir():
        raise FileNotFoundError(f"response dir not found: {in_dir}")

    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.provider if args.provider != "gemini" else "external_cli"  # gemini is the default; if the user didn't set --provider, label as cli
    out_path = SYNTH_DIR / f"{label}_augmented_{stamp}.jsonl"
    log_path = SYNTH_DIR / f"{label}_augmented_{stamp}.log.json"

    _, existing_users = load_existing_train_distribution()

    valid_tool_names = {name for name, _ in TOOLS} | {"__refusal__"}
    grand_accepted: list[dict] = []
    all_logs: list[dict] = []

    for fpath in sorted(in_dir.glob("*.txt")):
        # Recover tool name from filename (handle the safe_filename mangling)
        stem = fpath.stem.removesuffix(".prompt")
        # Try exact match first, fall back to fuzzy
        tool = None
        for t in valid_tool_names:
            if _safe_filename(t) == stem or t == stem:
                tool = t
                break
        if tool is None:
            print(f"  skip (unknown tool): {fpath.name}")
            continue

        text = fpath.read_text(encoding="utf-8", errors="replace")
        new_rows, stats = parse_teacher_output(text, tool, existing_users)
        grand_accepted.extend(new_rows)
        all_logs.append({"tool": tool, "file": fpath.name, "stats": stats})
        print(f"  {tool:42s}  accepted {len(new_rows):>3d}  stats={dict(stats)}")

    with out_path.open("w", encoding="utf-8") as f:
        for r in grand_accepted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log_path.write_text(json.dumps(all_logs, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"# wrote {len(grand_accepted)} new samples -> {out_path}")
    print(f"# log -> {log_path}")
    print()
    print("# next:")
    print(f"#   python scripts/04_merge_and_rebuild_manifest.py --source {_display_path(out_path)}")


def load_existing_train_distribution() -> tuple[Counter, set[str]]:
    dist = Counter()
    users: set[str] = set()
    with TRAIN_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            tool = obj.get("expected_tool")
            if tool:
                dist[tool] += 1
            for m in obj.get("messages", []):
                if m.get("role") == "user":
                    users.add(m["content"].strip())
    return dist, users


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=50, help="target sample count per tool")
    p.add_argument("--provider", choices=list(PROVIDERS), default="gemini")
    p.add_argument("--model", default=None, help="override model name")
    p.add_argument("--max-batch", type=int, default=30,
                   help="max queries to ask teacher in one call (split if target > this)")
    p.add_argument("--skip-tools", nargs="*", default=[],
                   help="tools to skip (already enough or out of scope)")
    p.add_argument("--only-tools", nargs="*", default=None,
                   help="if set, only augment these tools")
    p.add_argument("--dry-run", action="store_true", help="print plan, don't call API")
    # ---- BYO-CLI workflow: no API key in this script ----
    p.add_argument("--print-prompts", metavar="DIR", default=None,
                   help="don't call any API; just write one prompt file per tool to DIR. "
                        "Then feed each file through your own CLI (commandcode / opencode / "
                        "codex etc.) and save the response, then run --parse-responses.")
    p.add_argument("--parse-responses", metavar="DIR", default=None,
                   help="read CLI responses from DIR (filenames must match the prompt files), "
                        "extract JSONL rows, write data/synth/<provider>_augmented_<ts>.jsonl. "
                        "No API needed.")
    args = p.parse_args()

    # ---- Branch into print / parse modes if requested ----
    if args.print_prompts:
        cmd_print_prompts(args)
        return
    if args.parse_responses:
        cmd_parse_responses(args)
        return

    backend = PROVIDERS[args.provider]
    model_kw = {} if args.model is None else {"model_name": args.model}

    dist, existing_users = load_existing_train_distribution()
    print(f"# augment_with_teacher (provider={args.provider})")
    print(f"# current train: {sum(dist.values())} rows, {len(dist)} unique tools")
    print()

    # Plan
    plan = []
    for tool, _desc in TOOLS:
        if args.only_tools and tool not in args.only_tools:
            continue
        if tool in args.skip_tools:
            continue
        current = dist.get(tool, 0)
        needed = max(0, args.target - current)
        if needed > 0:
            plan.append((tool, current, needed))

    print(f"# plan: {len(plan)} tools to augment, target = {args.target} each")
    for tool, cur, need in plan:
        print(f"    {tool:42s}  current={cur:>3d}  +{need}")
    print()

    if args.dry_run:
        print("(dry run - no API calls)")
        return

    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SYNTH_DIR / f"{args.provider}_augmented_{stamp}.jsonl"
    log_path = SYNTH_DIR / f"{args.provider}_augmented_{stamp}.log.json"
    all_logs: list[dict] = []
    grand_accepted: list[dict] = []

    for tool, cur, need in plan:
        desc = next(d for n, d in TOOLS if n == tool)
        remaining = need
        accepted_for_tool: list[dict] = []
        attempt = 0
        while remaining > 0 and attempt < 3:
            attempt += 1
            n = min(remaining, args.max_batch)
            prompt = TEACHER_TEMPLATE.format(tool_name=tool, description=desc, n=n)
            try:
                text = backend(prompt, **model_kw) if model_kw else backend(prompt)
            except Exception as exc:
                print(f"  ! {tool} attempt {attempt} api error: {exc}")
                time.sleep(2)
                continue
            new_rows, stats = parse_teacher_output(text, tool, existing_users)
            accepted_for_tool.extend(new_rows)
            remaining = need - len(accepted_for_tool)
            print(f"  {tool:42s}  attempt {attempt}: requested {n}, accepted {len(new_rows)}, "
                  f"running total {len(accepted_for_tool)}/{need}, stats={dict(stats)}")
            all_logs.append({"tool": tool, "attempt": attempt, "stats": stats})
            time.sleep(0.5)  # be polite to the API
        grand_accepted.extend(accepted_for_tool)

    with out_path.open("w", encoding="utf-8") as f:
        for r in grand_accepted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log_path.write_text(json.dumps(all_logs, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"# wrote {len(grand_accepted)} new samples -> {out_path}")
    print(f"# log -> {log_path}")
    print()
    print("# next steps")
    print("# 1. Eyeball ~10% of the file. Manually delete bad rows.")
    print("# 2. python scripts/04_merge_and_rebuild_manifest.py "
          f"--source {out_path}")


if __name__ == "__main__":
    main()
