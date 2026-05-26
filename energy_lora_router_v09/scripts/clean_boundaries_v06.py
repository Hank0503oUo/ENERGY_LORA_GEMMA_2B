"""V6 Phase 2B: Clean answerability / dispatch boundary.

Fixes: rows where expected_dispatch_type=no_evidence AND
expected_answerability=unsupported_scope should be changed to
dispatch_type=refusal (unsupported scope is a refusal, not no-evidence).

Also validates consistency of the ANSWERABILITY_DISPATCH_MAP.

Usage:
    python clean_boundaries_v06.py
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import importlib.util

_here = Path(__file__).resolve().parent
cfg_path = _here / "00_config_v06.py"
spec = importlib.util.spec_from_file_location("v06cfg", cfg_path)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

ANSWERABILITY_DISPATCH_MAP = cfg.ANSWERABILITY_DISPATCH_MAP


def clean_file(src: Path, dst: Path, fix: bool = True) -> None:
    total = 0
    fixed = 0
    warnings = 0
    fixes_log: list[dict] = []
    inconsistency_counts: Counter = Counter()

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARN L{line_no}: JSON error: {e}", file=sys.stderr)
                fout.write(line + "\n")
                continue

            expected_dt = row.get("expected_dispatch_type", "")
            expected_ans = row.get("expected_answerability", "")

            correct_dt = ANSWERABILITY_DISPATCH_MAP.get(expected_ans, expected_dt)
            if correct_dt != expected_dt:
                inconsistency_counts[f"{expected_dt}+{expected_ans}"] += 1
                if fix:
                    row["expected_dispatch_type"] = correct_dt
                    messages = row.get("messages", [])
                    for m in messages:
                        if m.get("role") == "assistant":
                            try:
                                parsed = json.loads(m["content"])
                                parsed["dispatch_type"] = correct_dt
                                m["content"] = json.dumps(parsed, ensure_ascii=False)
                            except (json.JSONDecodeError, TypeError):
                                pass
                            break
                    fixes_log.append({
                        "line": line_no,
                        "old_dispatch_type": expected_dt,
                        "new_dispatch_type": correct_dt,
                        "answerability": expected_ans,
                    })
                    fixed += 1

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  {src.name} -> {dst.name}: {total} rows, {fixed} fixed, {len(inconsistency_counts)} inconsistency types")
    for key, count in inconsistency_counts.most_common():
        print(f"    {key}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Clean v06 dispatch boundary inconsistencies")
    parser.add_argument("--src-dir", type=Path,
                        default=Path(r"G:\我的雲端硬碟\energy_lora_router_v06\data\processed"),
                        help="Directory with v06 reprompted files")
    parser.add_argument("--no-fix", action="store_true", help="Only report, don't fix")
    args = parser.parse_args()

    data_dir = args.src_dir
    fix = not args.no_fix

    files = [
        "train_v06_dispatch.jsonl",
        "val_v06_dispatch.jsonl",
        "smoke_v06_dispatch.jsonl",
    ]

    for fname in files:
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  SKIP {fname} (not found)")
            continue
        print(f"\nProcessing: {fname}")
        backup = data_dir / f"{fname}.pre_boundary_clean.bak"
        if not backup.exists():
            import shutil
            shutil.copy2(str(fpath), str(backup))
        clean_file(fpath, fpath, fix=fix)

    print("\nDone.")


if __name__ == "__main__":
    main()
