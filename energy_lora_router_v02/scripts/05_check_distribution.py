"""Quick read-only diagnostic — print tool distribution + warnings.

Run after every data change (apply_tool_list / dedupe / merge) to confirm
the dataset is in the shape we expect before spending GPU on retraining.

Usage:
    python scripts/05_check_distribution.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
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
VAL_FILE = _cfg.VAL_FILE
SMOKE_FILE = _cfg.SMOKE_FILE
TARGET_PER_TOOL = 50  # what we'd like every tool to have in train


def load(p: Path) -> list[dict]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    train = load(TRAIN_FILE)
    val = load(VAL_FILE)
    smoke = load(SMOKE_FILE)

    print(f"# distribution check")
    print(f"  train={len(train)}  val={len(val)}  smoke={len(smoke)}")
    print()

    # System prompt status
    sys_msgs = {r["messages"][0]["content"] for r in train if r["messages"][0]["role"] == "system"}
    print(f"  unique system prompts in train: {len(sys_msgs)}")
    expected = _cfg.render_system_prompt()
    if expected in sys_msgs:
        print(f"  ✓ canonical system prompt (with tool list) is in train")
    else:
        print(f"  ✗ canonical system prompt NOT applied yet — run scripts/01_apply_tool_list_system.py")
    if len(sys_msgs) > 1:
        print(f"  ⚠ multiple system prompts found in train — re-run 01_apply to unify")
    print()

    # Tool distribution
    train_dist = Counter(r["expected_tool"] for r in train)
    val_dist = Counter(r["expected_tool"] for r in val)

    refusal_pct = 100 * train_dist.get("__refusal__", 0) / max(1, len(train))
    print(f"  train tool distribution (target {TARGET_PER_TOOL}+ per non-refusal tool):")
    for tool, _desc in _cfg.TOOLS:
        n = train_dist.get(tool, 0)
        v = val_dist.get(tool, 0)
        flag = ""
        if n < 10:
            flag = "  ← CRITICAL (<10)"
        elif n < TARGET_PER_TOOL:
            flag = f"  ← under target ({TARGET_PER_TOOL})"
        print(f"    {tool:42s}  train={n:>3d}  val={v:>2d}{flag}")
    n = train_dist.get("__refusal__", 0)
    print(f"    {'__refusal__':42s}  train={n:>3d}  val={val_dist.get('__refusal__', 0):>2d}  "
          f"({refusal_pct:.0f}% of train) {'← over-represented' if refusal_pct > 25 else ''}")
    print()

    # Coverage gap
    val_only_tools = set(val_dist) - set(train_dist)
    if val_only_tools:
        print(f"  ⚠ tools in val but missing from train: {sorted(val_only_tools)}")
    rare = [t for t, n in train_dist.items() if n < 10 and t != "__refusal__"]
    if rare:
        print(f"  ⚠ {len(rare)} tools have <10 train samples — they'll likely fail in val:")
        for t in rare:
            print(f"      - {t}: train={train_dist[t]}, val={val_dist.get(t, 0)}")
    print()

    # Estimated training step count for next run
    for bs, ga in [(8, 2), (8, 4), (16, 2)]:
        eff = bs * ga
        steps_per_epoch = (len(train) + eff - 1) // eff
        for ep in [3, 4, 5]:
            print(f"    @ batch {bs}×{ga}={eff}, {ep} epochs → {steps_per_epoch * ep} steps "
                  f"(~{steps_per_epoch * ep * 2.5:.0f}s on A100, ~{steps_per_epoch * ep * 8:.0f}s on T4)")


if __name__ == "__main__":
    main()
