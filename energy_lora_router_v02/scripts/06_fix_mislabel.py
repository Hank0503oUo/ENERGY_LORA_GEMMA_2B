"""桶 1: Relabel mislabeled training samples.

Scans train JSONL for known label-noise patterns and produces a relabeled copy.

Fix patterns:
1a. calibrate_sensitivity vs get_sensitivity_status:
    - 「分析/報告/紀錄/查詢/數據/解析」→ get_sensitivity_status (not calibrate)
    - 「回灌/校準/調整」→ calibrate_sensitivity

1b. confirm_strategy_adoption vs check_strategy_status:
    - 「目前」「在不在」「是否在實施」→ check_status
    - 「確認」「是否已採用」「用了沒」(past tense yes/no) → confirm_adoption

1c. query_energy_records edge cases:
    - 「即時/realtime」→ __refusal__ (system doesn't support)
    - 「電費/度數」→ __refusal__ (per rule #8)

Usage:
    python 06_fix_mislabel.py [--dry-run] [--input INPUT] [--output OUTPUT]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def get_user_input(row: dict) -> str:
    for m in row.get("messages", []):
        if m.get("role") == "user":
            return m.get("content", "").strip()
    return ""


def get_assistant_content(row: dict) -> str:
    for m in row.get("messages", []):
        if m.get("role") == "assistant":
            return m.get("content", "").strip()
    return ""


def fix_calibrate_sensitivity(query: str, current_tool: str) -> tuple[str, str, str]:
    """Fix calibrate_sensitivity vs get_sensitivity_status.

    Returns (new_tool, reason, action).
    """
    if current_tool != "calibrate_sensitivity":
        return current_tool, "", "keep"

    passive_keywords = ["分析", "報告", "紀錄", "查詢", "查尋", "數據", "解析", "狀態", "目前"]
    active_keywords = ["回灌", "校準", "調整", "更新", "修正", "重新"]

    has_passive = any(kw in query for kw in passive_keywords)
    has_active = any(kw in query for kw in active_keywords)

    if has_passive and not has_active:
        return "get_sensitivity_status", "passive_kw_in_query", "relabel"
    elif has_active and not has_passive:
        return "calibrate_sensitivity", "active_kw_correct", "keep"

    return current_tool, "", "keep"


def fix_confirm_vs_check(query: str, current_tool: str) -> tuple[str, str, str]:
    """Fix confirm_strategy_adoption vs check_strategy_status."""
    if current_tool not in ("confirm_strategy_adoption", "check_strategy_status"):
        return current_tool, "", "keep"

    check_keywords = ["目前", "在不在", "是否在實施", "現在", "目前狀態", "狀態查詢"]
    confirm_keywords = ["確認", "是否已採用", "用了沒", "已經採用", "已經用", "正式上線"]

    has_check = any(kw in query for kw in check_keywords)
    has_confirm = any(kw in query for kw in confirm_keywords)

    if current_tool == "confirm_strategy_adoption" and has_check and not has_confirm:
        return "check_strategy_status", "check_kw_in_query", "relabel"

    if current_tool == "check_strategy_status" and has_confirm and not has_check:
        return "confirm_strategy_adoption", "confirm_kw_in_query", "relabel"

    return current_tool, "", "keep"


def fix_query_edge(query: str, current_tool: str) -> tuple[str, str, str]:
    """Fix query_energy_records edge cases → __refusal__."""
    if current_tool != "query_energy_records":
        return current_tool, "", "keep"

    realtime_kw = ["即時", "real-time", "realtime", "即時電力"]
    bill_kw = ["電費", "度數", "帳單"]

    if any(kw in query for kw in realtime_kw):
        return "__refusal__", "realtime_not_supported", "relabel"
    if any(kw in query for kw in bill_kw):
        return "__refusal__", "bill_not_in_scope", "relabel"

    return current_tool, "", "keep"


def fix_row(row: dict) -> tuple[dict, list[dict]]:
    """Apply all fix patterns to a single row.

    Returns (possibly_modified_row, list_of_changes).
    """
    query = get_user_input(row)
    current_tool = row.get("expected_tool", "")
    changes = []

    fixers = [
        ("calibrate_sensitivity", fix_calibrate_sensitivity),
        ("confirm_vs_check", fix_confirm_vs_check),
        ("query_edge", fix_query_edge),
    ]

    new_tool = current_tool
    for fixer_name, fixer_fn in fixers:
        proposed, reason, action = fixer_fn(query, new_tool)
        if action == "relabel":
            changes.append({
                "fixer": fixer_name,
                "from": new_tool,
                "to": proposed,
                "reason": reason,
                "query": query[:80],
            })
            new_tool = proposed

    if changes:
        row = dict(row)
        row["expected_tool"] = new_tool
        row["messages"] = list(row.get("messages", []))
        # Update assistant content to match new tool
        for i, m in enumerate(row["messages"]):
            if m.get("role") == "assistant":
                content = m.get("content", "")
                try:
                    parsed = json.loads(content.strip().strip("`").strip())
                    parsed["tool"] = new_tool
                    row["messages"][i] = dict(m)
                    row["messages"][i]["content"] = json.dumps(parsed, ensure_ascii=False)
                except (json.JSONDecodeError, TypeError):
                    pass
                break

    return row, changes


def main():
    parser = argparse.ArgumentParser(description="Relabel mislabeled training samples")
    parser.add_argument("--input", type=Path,
                        default=Path(r"G:\我的雲端硬碟\energy_lora_router_v02\data\harness_v02_train.jsonl"),
                        help="Input JSONL")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSONL (default: <input>_relabeled.jsonl)")
    parser.add_argument("--dry-run", action="store_true", help="Print changes only, don't write")
    parser.add_argument("--also-val", action="store_true", help="Also fix val/smoke JSONL")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.parent / (args.input.stem + "_relabeled.jsonl")

    files_to_fix = [args.input]
    if args.also_val:
        parent = args.input.parent
        for name in ["harness_v02_val.jsonl", "harness_v02_smoke.jsonl"]:
            p = parent / name
            if p.exists():
                files_to_fix.append(p)

    total_changes = 0
    total_rows = 0
    change_by_fixer: Counter = Counter()
    change_by_tool: Counter = Counter()

    for fp in files_to_fix:
        print(f"\n{'='*60}")
        print(f"Scanning: {fp.name}")
        print(f"{'='*60}")

        rows = []
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

        fixed_rows = []
        file_changes = 0

        for row in rows:
            fixed, changes = fix_row(row)
            fixed_rows.append(fixed)
            if changes:
                file_changes += 1
                total_changes += 1
                for c in changes:
                    change_by_fixer[c["fixer"]] += 1
                    change_by_tool[f"{c['from']} -> {c['to']}"] += 1
                    print(f"  [{c['fixer']}] {c['from']} -> {c['to']}")
                    print(f"    query: {c['query']}")

        total_rows += len(rows)

        if not args.dry_run and file_changes > 0:
            out_path = fp.parent / (fp.stem + "_relabeled.jsonl")
            with out_path.open("w", encoding="utf-8") as f:
                for row in fixed_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  Written: {out_path} ({file_changes} rows changed)")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Total rows scanned : {total_rows}")
    print(f"  Total rows changed : {total_changes}")
    print(f"\n  By fixer:")
    for fixer, cnt in change_by_fixer.most_common():
        print(f"    {fixer}: {cnt}")
    print(f"\n  By tool transition:")
    for trans, cnt in change_by_tool.most_common():
        print(f"    {trans}: {cnt}")


if __name__ == "__main__":
    main()
