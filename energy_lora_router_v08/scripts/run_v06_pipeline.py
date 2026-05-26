"""V6 full pipeline runner.

Executes all phases in order:
  Phase 1: Config already done (00_config_v06.py)
  Phase 2A: Re-prompt data with new system prompt
  Phase 2B: Clean boundary inconsistencies
  Phase 3: Generate format curriculum and merge into train
  Phase 4: Generate format smoke test

Usage:
    python run_v06_pipeline.py
    python run_v06_pipeline.py --skip-reprompt  # if already reprompted
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

V06_ROOT = Path(r"G:\我的雲端硬碟\energy_lora_router_v06")
V05_ROOT = Path(r"G:\我的雲端硬碟\energy_lora_router_v05")
DATA_DIR = V06_ROOT / "data" / "processed"
SYNTH_DIR = V06_ROOT / "data" / "synth"


def run_script(name: str) -> None:
    import subprocess
    script = V06_ROOT / "scripts" / name
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(V06_ROOT / "scripts"),
    )
    if result.returncode != 0:
        print(f"ERROR: {name} failed with return code {result.returncode}")
        sys.exit(1)


def merge_curriculum_into_train() -> None:
    train_path = DATA_DIR / "train_v06_dispatch.jsonl"
    curriculum_path = SYNTH_DIR / "v06_format_curriculum.jsonl"

    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run reprompt first.")
        sys.exit(1)
    if not curriculum_path.exists():
        print(f"ERROR: {curriculum_path} not found. Run format_curriculum first.")
        sys.exit(1)

    train_rows = []
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_rows.append(line)

    curriculum_rows = []
    with curriculum_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                curriculum_rows.append(line)

    backup = train_path.with_suffix(".jsonl.pre_curriculum_merge.bak")
    if not backup.exists():
        import shutil
        shutil.copy2(str(train_path), str(backup))

    with train_path.open("w", encoding="utf-8") as f:
        for line in curriculum_rows:
            f.write(line)
        for line in train_rows:
            f.write(line)

    print(f"Merged {len(curriculum_rows)} curriculum samples into train ({len(train_rows)} original)")
    print(f"  New train total: {len(curriculum_rows) + len(train_rows)}")
    print(f"  Backup: {backup}")


def generate_manifest() -> None:
    train_path = DATA_DIR / "train_v06_dispatch.jsonl"
    val_path = DATA_DIR / "val_v06_dispatch.jsonl"
    smoke_path = DATA_DIR / "smoke_v06_dispatch.jsonl"
    format_smoke_path = DATA_DIR / "format_smoke_v06.jsonl"

    def count_rows(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(1 for line in path.open("r", encoding="utf-8") if line.strip())

    train_n = count_rows(train_path)
    val_n = count_rows(val_path)
    smoke_n = count_rows(smoke_path)
    format_smoke_n = count_rows(format_smoke_path)

    manifest = {
        "version": "0.6",
        "profile": "agent_dispatch_training",
        "schema": "dispatch_chat_messages",
        "total": train_n + val_n + smoke_n,
        "train": train_n,
        "val": val_n,
        "smoke": smoke_n,
        "format_smoke": format_smoke_n,
        "split_method": "v05 reprompted + boundary cleaned + format curriculum merged",
        "v06_changes": [
            "strict dispatch-only system prompt (no assistant persona)",
            "DCI tools formally declared in prompt",
            "boundary fix: unsupported_scope -> refusal",
            "format curriculum samples added to train",
            "format smoke test added",
        ],
        "notes": [
            "V6 = dispatch contract recovery round.",
            "Parse error rate target: <10%.",
            "See V6_DISPATCH_RECOVERY_PLAN.md for full rationale.",
        ],
    }

    manifest_path = DATA_DIR / "harness_v06_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest written: {manifest_path}")
    print(f"  train: {train_n}, val: {val_n}, smoke: {smoke_n}, format_smoke: {format_smoke_n}")


def main():
    parser = argparse.ArgumentParser(description="Run full V6 pipeline")
    parser.add_argument("--skip-reprompt", action="store_true")
    parser.add_argument("--skip-boundary", action="store_true")
    parser.add_argument("--skip-curriculum", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("V6 DISPATCH CONTRACT RECOVERY PIPELINE")
    print("=" * 60)

    if not args.skip_reprompt:
        run_script("reprompt_v06_data.py")

    if not args.skip_boundary:
        run_script("clean_boundaries_v06.py")

    if not args.skip_curriculum:
        run_script("format_curriculum_v06.py")

    if not args.skip_merge:
        merge_curriculum_into_train()

    if not args.skip_smoke:
        run_script("format_smoke_v06.py")

    generate_manifest()

    print("\n" + "=" * 60)
    print("V6 PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    print("Next steps:")
    print("  1. Run train_lora.py on Colab")
    print("  2. Run format_smoke eval first (gate: parse error < 10%)")
    print("  3. If gate passes, run full val eval")


if __name__ == "__main__":
    main()
