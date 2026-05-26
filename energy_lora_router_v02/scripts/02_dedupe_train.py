"""Remove duplicate user prompts from train.jsonl.

Audit reported 6 duplicates. We keep the first occurrence of each unique
(user_content, expected_tool) pair. If the same user prompt maps to
DIFFERENT tools across rows, that's a labelling conflict — we log it but
keep only the first to avoid teaching the model contradictory mappings.

Run:
    python scripts/02_dedupe_train.py
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util as _ilu
_cfg_spec = _ilu.spec_from_file_location("cfg00", Path(__file__).resolve().parent / "00_config.py")
_cfg = _ilu.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg)

TRAIN_FILE = _cfg.TRAIN_FILE
BACKUP_DIR = _cfg.BACKUP_DIR


def main() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bk = BACKUP_DIR / f"{TRAIN_FILE.stem}_predupe_{stamp}{TRAIN_FILE.suffix}"
    shutil.copy2(TRAIN_FILE, bk)

    rows = []
    with TRAIN_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))

    seen: dict[str, str] = {}
    conflicts: dict[str, set[str]] = defaultdict(set)
    deduped = []
    dropped_dupes = 0

    for r in rows:
        msgs = r.get("messages", [])
        user = next((m["content"] for m in msgs if m.get("role") == "user"), None)
        tool = r.get("expected_tool")
        if user is None:
            deduped.append(r)
            continue
        key = user.strip()
        if key in seen:
            if seen[key] != tool:
                conflicts[key].add(seen[key])
                conflicts[key].add(tool)
            dropped_dupes += 1
            continue
        seen[key] = tool
        deduped.append(r)

    with TRAIN_FILE.open("w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"# scripts/02_dedupe_train.py")
    print(f"  before: {len(rows)} rows")
    print(f"  after:  {len(deduped)} rows  (dropped {dropped_dupes})")
    print(f"  backup: {bk.name}")
    if conflicts:
        print()
        print(f"  ⚠ label conflicts ({len(conflicts)}): same user prompt mapped to multiple tools")
        for user, tools in conflicts.items():
            print(f"    - {sorted(tools)}: {user[:80]!r}")
        print("  → kept the first occurrence; review backup if you want to re-label.")


if __name__ == "__main__":
    main()
