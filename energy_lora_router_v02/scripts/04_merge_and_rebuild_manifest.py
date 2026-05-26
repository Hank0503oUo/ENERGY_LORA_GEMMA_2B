"""Merge an augmented JSONL of {user, expected_tool} rows into train.jsonl,
then rebuild the manifest.

Will NOT touch val.jsonl or smoke.jsonl (val/smoke must stay human-curated
to keep the eval honest).

Usage:
    python scripts/04_merge_and_rebuild_manifest.py --source data/synth/gemini_augmented_<ts>.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util as _ilu
_cfg_spec = _ilu.spec_from_file_location("cfg00", Path(__file__).resolve().parent / "00_config.py")
_cfg = _ilu.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg)

TRAIN_FILE = _cfg.TRAIN_FILE
VAL_FILE = _cfg.VAL_FILE
SMOKE_FILE = _cfg.SMOKE_FILE
MANIFEST_FILE = _cfg.MANIFEST_FILE
BACKUP_DIR = _cfg.BACKUP_DIR
SYSTEM_PROMPT = _cfg.render_system_prompt()
VALID_TOOL_NAMES = _cfg.VALID_TOOL_NAMES


def make_synth_row(user: str, tool: str, idx: int, source_label: str, difficulty: str = "easy") -> dict:
    """Construct a full SFT row matching the existing harness_v02 schema."""
    h = hashlib.sha1(f"{source_label}|{user}|{tool}".encode()).hexdigest()[:10]
    sample_id = f"harness_v02_synth_{h}"
    args = {}
    # cheap arguments inference for the most common tools, can be refined later
    if tool == "query_energy_records":
        # try to extract a building name from the user query (very rough)
        args = {"buildings": [user[:30]]} if user else {}
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps({"tool": tool, "arguments": args}, ensure_ascii=False)},
        ],
        "expected_tool": tool,
        "difficulty": difficulty,
        "category": "routing",
        "sample_id": sample_id,
        "source_file": source_label,
    }


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def existing_user_set(path: Path) -> set[str]:
    users = set()
    for r in load_jsonl(path):
        for m in r.get("messages", []):
            if m.get("role") == "user":
                users.add(m["content"].strip())
    return users


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="path to augmented {user, expected_tool} jsonl")
    p.add_argument("--source-label", default=None, help="manifest source_file label (defaults to filename)")
    args = p.parse_args()

    source = Path(args.source)
    if not source.is_absolute():
        source = (_cfg.DRIVE_ROOT / source).resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    label = args.source_label or source.stem  # e.g. gemini_augmented_20260509_120000

    # Read augmented rows
    raw = load_jsonl(source)
    print(f"# source: {source}  rows={len(raw)}")

    # Validate
    existing_users = existing_user_set(TRAIN_FILE) | existing_user_set(VAL_FILE) | existing_user_set(SMOKE_FILE)
    accepted: list[dict] = []
    skipped = Counter()
    for i, obj in enumerate(raw):
        user = (obj.get("user") or "").strip()
        tool = obj.get("expected_tool")
        difficulty = obj.get("difficulty", "easy")  # read from source if provided
        if not user:
            skipped["empty_user"] += 1
            continue
        if tool not in VALID_TOOL_NAMES:
            skipped[f"unknown_tool:{tool}"] += 1
            continue
        if user in existing_users:
            skipped["duplicate_against_existing"] += 1
            continue
        existing_users.add(user)
        accepted.append(make_synth_row(user, tool, i, label, difficulty))

    print(f"# accepted: {len(accepted)}  skipped: {dict(skipped) or '{}'}")

    if not accepted:
        print("# nothing to merge.")
        return

    # Backup train + manifest
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(TRAIN_FILE, BACKUP_DIR / f"{TRAIN_FILE.stem}_premerge_{stamp}{TRAIN_FILE.suffix}")
    shutil.copy2(MANIFEST_FILE, BACKUP_DIR / f"{MANIFEST_FILE.stem}_premerge_{stamp}{MANIFEST_FILE.suffix}")

    # Append
    with TRAIN_FILE.open("a", encoding="utf-8") as f:
        for r in accepted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"# appended {len(accepted)} rows → {TRAIN_FILE.name}")

    # Rebuild manifest
    rebuild_manifest(extra_source={label: len(accepted)})


def rebuild_manifest(extra_source: dict[str, int] | None = None) -> None:
    train_rows = load_jsonl(TRAIN_FILE)
    val_rows = load_jsonl(VAL_FILE)
    smoke_rows = load_jsonl(SMOKE_FILE)

    sources_count = Counter(r.get("source_file", "?") for r in train_rows + val_rows + smoke_rows)

    tool_dist = Counter(r.get("expected_tool") for r in train_rows + val_rows + smoke_rows)
    cat_dist = Counter(r.get("category") for r in train_rows + val_rows + smoke_rows)
    diff_dist = Counter(r.get("difficulty") for r in train_rows + val_rows + smoke_rows)
    train_cat = Counter(r.get("category") for r in train_rows)
    val_cat = Counter(r.get("category") for r in val_rows)

    manifest = {
        "version": "0.3",
        "profile": "router_strict",
        "total": len(train_rows) + len(val_rows) + len(smoke_rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "smoke": len(smoke_rows),
        "val_ratio": round(len(val_rows) / max(1, len(train_rows) + len(val_rows)), 4),
        "split_method": "v0.2 deterministic + appended synth on train only",
        "sources": dict(sources_count),
        "tool_distribution": dict(tool_dist.most_common()),
        "category_distribution": dict(cat_dist),
        "difficulty_distribution": dict(diff_dist),
        "train_category_distribution": dict(train_cat),
        "val_category_distribution": dict(val_cat),
        "regenerated_at": datetime.now(timezone.utc).isoformat(),
        "system_prompt_chars": len(SYSTEM_PROMPT),
        "system_prompt_lists_tool_catalogue": True,
        "eval_gates": {
            "tool_accuracy": ">=80%",
            "malformed_json": "<5%",
            "hard_trap_accuracy": ">=60%",
        },
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"# rebuilt manifest:")
    print(f"    train={manifest['train']}  val={manifest['val']}  smoke={manifest['smoke']}  total={manifest['total']}")
    print(f"    top tools: {list(tool_dist.most_common(10))}")


if __name__ == "__main__":
    main()
