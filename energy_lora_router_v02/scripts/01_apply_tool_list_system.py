"""Step 1: rewrite the system message in train / val / smoke.

Replaces every sample's system message with the canonical tool-list version
from 00_config. Backs up originals to data/_backup/ first.

Run:
    python scripts/01_apply_tool_list_system.py
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util as _ilu
_cfg_spec = _ilu.spec_from_file_location("cfg00", Path(__file__).resolve().parent / "00_config.py")
_cfg = _ilu.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg)

TRAIN_FILE, VAL_FILE, SMOKE_FILE = _cfg.TRAIN_FILE, _cfg.VAL_FILE, _cfg.SMOKE_FILE
BACKUP_DIR = _cfg.BACKUP_DIR
NEW_SYSTEM = _cfg.render_system_prompt()


def backup(src: Path) -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"{src.stem}_{stamp}{src.suffix}"
    shutil.copy2(src, dst)
    return dst


def patch_file(path: Path) -> tuple[int, int]:
    rows = []
    patched = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            obj = json.loads(line)
            msgs = obj.get("messages", [])
            if msgs and msgs[0].get("role") == "system":
                if msgs[0]["content"] != NEW_SYSTEM:
                    msgs[0]["content"] = NEW_SYSTEM
                    patched += 1
            rows.append(obj)

    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return patched, len(rows)


def main() -> None:
    print(f"# scripts/01_apply_tool_list_system.py")
    print(f"# new system prompt: {len(NEW_SYSTEM)} chars, lists {len(_cfg.TOOLS)} tools + __refusal__")
    print()
    for fp in (TRAIN_FILE, VAL_FILE, SMOKE_FILE):
        if not fp.exists():
            print(f"skip (missing): {fp}")
            continue
        bk = backup(fp)
        patched, total = patch_file(fp)
        print(f"  {fp.name}: patched {patched}/{total}  (backup → {bk.name})")
    print()
    print("done.  Now run scripts/05_check_distribution.py to verify, then retrain.")


if __name__ == "__main__":
    main()
