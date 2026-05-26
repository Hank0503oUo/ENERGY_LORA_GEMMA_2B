"""V6 Phase 2A: Re-apply new strict dispatch system prompt to all data files.

Reads v05 dispatch chat JSONL files, replaces messages[0].content (system prompt)
with the new v06 strict dispatch prompt, writes v06 versions.

Usage:
    python reprompt_v06_data.py
    python reprompt_v06_data.py --src-dir "G:\\我的雲端硬碟\\energy_lora_router_v05\\data\\processed"
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import importlib.util

_here = Path(__file__).resolve().parent
cfg_path = _here / "00_config_v06.py"
spec = importlib.util.spec_from_file_location("v06cfg", cfg_path)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

NEW_SYSTEM_PROMPT = cfg.render_system_prompt()


def reprompt_file(src: Path, dst: Path) -> None:
    total = 0
    patched = 0
    dst.parent.mkdir(parents=True, exist_ok=True)

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
                continue

            messages = row.get("messages", [])
            if messages and messages[0].get("role") == "system":
                old_prompt = messages[0]["content"]
                messages[0]["content"] = NEW_SYSTEM_PROMPT
                patched += 1
            else:
                print(f"  WARN L{line_no}: first message is not system role", file=sys.stderr)

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  {src.name} -> {dst.name}: {patched}/{total} rows patched")


def main():
    parser = argparse.ArgumentParser(description="Re-apply v06 strict dispatch prompt to v05 data")
    parser.add_argument("--src-dir", type=Path,
                        default=Path(r"G:\我的雲端硬碟\energy_lora_router_v05\data\processed"),
                        help="Source directory with v05 dispatch files")
    parser.add_argument("--dst-dir", type=Path,
                        default=Path(r"G:\我的雲端硬碟\energy_lora_router_v06\data\processed"),
                        help="Destination directory for v06 files")
    args = parser.parse_args()

    src = args.src_dir
    dst = args.dst_dir
    dst.mkdir(parents=True, exist_ok=True)

    files = [
        ("train_v05_dispatch.jsonl", "train_v06_dispatch.jsonl"),
        ("val_v05_dispatch.jsonl", "val_v06_dispatch.jsonl"),
        ("smoke_v05_dispatch.jsonl", "smoke_v06_dispatch.jsonl"),
    ]

    for src_name, dst_name in files:
        src_path = src / src_name
        dst_path = dst / dst_name
        if not src_path.exists():
            print(f"  SKIP {src_name} (not found)")
            continue
        print(f"\nProcessing: {src_name}")
        reprompt_file(src_path, dst_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
