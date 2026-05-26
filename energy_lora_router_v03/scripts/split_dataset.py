"""v03 stratified dataset splitter.

Splits a merged JSONL into train/val/smoke with stratified sampling.
Ensures val >= 300 with balanced difficulty/category/tool distribution.

Usage:
    python split_dataset.py /path/to/merged_v03_all.jsonl --output-dir /path/to/data/processed/
    python split_dataset.py /path/to/merged_v03_all.jsonl --output-dir ... --val-ratio 0.12 --val-min 300
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SEED = 42


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] {path}:{line_no}: {e}", file=sys.stderr)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Written: {path} ({len(rows)} rows)")


def get_stratify_key(row: dict[str, Any]) -> str:
    """Composite key for stratified splitting."""
    difficulty = row.get("difficulty", "unknown")
    category = row.get("category", "unknown")
    tool = row.get("expected_tool", "unknown")
    return f"{difficulty}|{category}|{tool}"


def extract_smoke_set(
    all_rows: list[dict[str, Any]],
    smoke_size: int = 16,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract a fixed smoke test set covering key categories.

    Smoke set should include at least one of each:
    - chart question (generate_meter_chart)
    - docs question (search_docs)
    - query vs list_campus_stats confusion pair
    - counterfactual question
    - OpenBSE question
    - safety refusal
    - malformed input
    - trap question
    """
    rng = random.Random(seed)

    # Group by expected_tool for targeted selection
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        by_tool[row.get("expected_tool", "unknown")].append(row)

    smoke_candidates: list[dict] = []
    seen_indices: set[int] = set()

    # Required coverage targets — names must match v02 actual tool catalogue.
    # The smoke set should hit every tool that historically had eval failures,
    # plus a couple of safety / refusal anchors.
    required_tools = [
        ("generate_meter_chart", 2),                  # v0.4 sample 2 regression
        ("search_docs", 2),                           # v0.4 sample 6/7/8 fail
        ("query_energy_records", 2),                  # v0.4 sample 24/26/32 confused with list_campus_stats
        ("list_campus_stats", 1),                     # the foil for above
        ("__refusal__", 2),                           # safety / trap anchor
        ("validate_strategy_openbse", 1),
        ("run_counterfactual_for_building", 1),       # was wrongly "simulate_counterfactual_replacement"
        ("detect_energy_anomalies", 1),               # was wrongly "detect_anomaly_iqr"
        ("compare_building_trends", 1),
        ("recommend_adaptive_strategies", 1),         # foil for seasonal_strategies
    ]

    for tool, count in required_tools:
        pool = [r for r in by_tool.get(tool, []) if id(r) not in seen_indices]
        selected = rng.sample(pool, min(count, len(pool)))
        smoke_candidates.extend(selected)
        seen_indices.update(id(r) for r in selected)

    # Fill remaining slots with random diverse samples
    remaining_pool = [r for r in all_rows if id(r) not in seen_indices]
    needed = smoke_size - len(smoke_candidates)
    if needed > 0 and remaining_pool:
        extra = rng.sample(remaining_pool, min(needed, len(remaining_pool)))
        smoke_candidates.extend(extra)
        seen_indices.update(id(r) for r in extra)

    # Shuffle smoke set order
    rng.shuffle(smoke_candidates)

    # Remove smoke from main pool
    remaining = [r for r in all_rows if id(r) not in seen_indices]

    return smoke_candidates, remaining


def stratified_split(
    rows: list[dict[str, Any]],
    val_ratio: float = 0.12,
    val_min_size: int = 300,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratified train/val split.

    Ensures each (difficulty, category, tool) stratum is represented
    proportionally in both splits.
    """
    rng = random.Random(seed)

    # Group by stratum
    strata: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = get_stratify_key(row)
        strata[key].append(row)

    train_rows: list[dict] = []
    val_rows: list[dict] = []

    print(f"  Stratifying {len(rows)} rows into {len(strata)} strata...")

    for key, group in sorted(strata.items()):
        n = len(group)
        n_val = max(1, round(n * val_ratio))
        rng.shuffle(group)
        val_rows.extend(group[:n_val])
        train_rows.extend(group[n_val:])

    # If val too small, upsample from large strata
    if len(val_rows) < val_min_size:
        shortfall = val_min_size - len(val_rows)
        print(f"  Val set ({len(val_rows)}) below minimum ({val_min_size}), upsampling {shortfall}...")
        # Sort train strata by size descending, take from largest
        train_strata: dict[str, list[dict]] = defaultdict(list)
        for row in train_rows:
            train_strata[get_stratify_key(row)].append(row)

        sorted_strata = sorted(train_strata.items(), key=lambda x: -len(x[1]))
        extra_val: list[dict] = []
        for key, group in sorted_strata:
            if len(extra_val) >= shortfall:
                break
            take = min(len(group), shortfall - len(extra_val))
            extra_val.extend(rng.sample(group, take))

        # Move extra from train to val
        extra_set = set(id(r) for r in extra_val)
        val_rows.extend(extra_val)
        train_rows = [r for r in train_rows if id(r) not in extra_set]

    return train_rows, val_rows


def build_manifest(
    train_path: Path,
    val_path: Path,
    smoke_path: Path,
    output_dir: Path,
    sources: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Generate v03 manifest.json."""
    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)
    smoke_rows = read_jsonl(smoke_path)

    total = len(train_rows) + len(val_rows) + len(smoke_rows)

    tool_counts: dict[str, int] = defaultdict(int)
    diff_counts: dict[str, int] = defaultdict(int)
    cat_counts: dict[str, int] = defaultdict(int)

    for row in train_rows + val_rows + smoke_rows:
        tool_counts[row.get("expected_tool", "?")] += 1
        diff_counts[row.get("difficulty", "?")] += 1
        cat_counts[row.get("category", "?")] += 1

    manifest = {
        "version": "0.4",
        "profile": "router_strict",
        "total": total,
        "train": len(train_rows),
        "val": len(val_rows),
        "smoke": len(smoke_rows),
        "val_ratio": round(len(val_rows) / max(total - len(smoke_rows), 1), 4),
        "split_method": "stratified (difficulty|category|tool) + fixed smoke",
        "sources": sources or {},
        "tool_distribution": dict(sorted(tool_counts.items(), key=lambda x: -x[1])),
        "difficulty_distribution": dict(sorted(diff_counts.items())),
        "category_distribution": dict(sorted(cat_counts.items(), key=lambda x: -x[1])),
        "eval_gates": {
            "overall_accuracy": ">=90%",
            "safety_accuracy": ">=95%",
            "trap_accuracy": ">=85%",
            "search_docs_accuracy": ">=90%",
            "chart_accuracy": ">=90%",
            "parse_error_rate": "<=3%",
            "over_refusal_rate": "<=5%",
            "stats_query_confusion": "<=3%",
        },
    }

    manifest_path = output_dir / "harness_v03_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Manifest written: {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Split v03 dataset into train/val/smoke")
    parser.add_argument("input_jsonl", type=Path, help="Merged input JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for split files")
    parser.add_argument("--val-ratio", type=float, default=0.12, help="Validation ratio (default: 0.12)")
    parser.add_argument("--val-min", type=int, default=300, help="Minimum validation size (default: 300)")
    parser.add_argument("--smoke-size", type=int, default=16, help="Smoke test set size (default: 16)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    args = parser.parse_args()

    if not args.input_jsonl.exists():
        print(f"[ERROR] Input file not found: {args.input_jsonl}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading: {args.input_jsonl}")
    all_rows = read_jsonl(args.input_jsonl)
    print(f"  Total rows: {len(all_rows)}")

    # Step 1: Extract smoke set
    print("\n--- Smoke Set ---")
    smoke_rows, remaining = extract_smoke_set(all_rows, smoke_size=args.smoke_size, seed=args.seed)
    print(f"  Smoke: {len(smoke_rows)} rows")

    # Step 2: Stratified train/val split
    print("\n--- Train/Val Split ---")
    train_rows, val_rows = stratified_split(
        remaining, val_ratio=args.val_ratio, val_min_size=args.val_min, seed=args.seed
    )
    print(f"  Train: {len(train_rows)} rows")
    print(f"  Val:   {len(val_rows)} rows")

    # Step 3: Write files
    print("\n--- Writing Output Files ---")
    out = args.output_dir
    train_path = out / "train_v03.jsonl"
    val_path = out / "val_v03.jsonl"
    smoke_path = out / "smoke_v03.jsonl"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(smoke_path, smoke_rows)

    # Also write merged copy
    merged_path = out / "merged_v03_all.jsonl"
    write_jsonl(merged_path, all_rows)

    # Step 4: Build manifest
    print("\n--- Manifest ---")
    manifest = build_manifest(train_path, val_path, smoke_path, out)

    # Summary
    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    print(f"  Input     : {args.input_jsonl.name} ({len(all_rows)} rows)")
    print(f"  Train     : {len(train_rows)} ({len(train_rows)/len(all_rows):.1%})")
    print(f"  Val       : {len(val_rows)} ({len(val_rows)/len(all_rows):.1%})")
    print(f"  Smoke     : {len(smoke_rows)}")
    print(f"  Val tools : {dict(Counter(r['expected_tool'] for r in val_rows))}")
    print(f"  Val diff  : {dict(Counter(r['difficulty'] for r in val_rows))}")
    print("=" * 60)


if __name__ == "__main__":
    from collections import Counter
    main()
