"""v03 multi-adapter comparison tool.

Evaluates multiple adapters (base, best checkpoint, final) and produces:
- Combined comparison table
- Best adapter recommendation based on selection score
- Confusion matrix comparison

Usage:
    python compare_adapters.py \
        --base-model \
        --adapter outputs/gemma_router_strict_v02/adapter \
        --ckpt outputs/gemma_router_strict_v02/checkpoints/checkpoint-156 \
        --val-file data/processed/val_v03.jsonl \
        --smoke-file data/processed/smoke_v03.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Import evaluate functions from sibling module
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_router import (  # noqa: E402
    evaluate_split,
    compute_selection_score,
    check_gates,
    MODEL_ID,
    MAX_SEQ_LENGTH,
    OUTPUT_DIR as EVAL_OUTPUT_DIR,
)


def load_model_with_adapter(adapter_path: Path | None = None, base_only: bool = False):
    """Load base model + optional adapter."""
    import torch
    from unsloth import FastModel

    print(f"[model] Loading {MODEL_ID} ...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=False,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path and not base_only:
        from peft import PeftModel
        print(f"[model] Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))

    return model, tokenizer


def run_comparison(
    val_file: Path,
    smoke_file: Path | None,
    adapters: list[tuple[str, Path | None]],
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Evaluate all adapters and produce comparison report."""
    results: dict[str, dict[str, Any]] = {}
    t_total = time.time()

    for label, adapter_path in adapters:
        print(f"\n{'='*60}")
        print(f"EVALUATING: {label}")
        print(f"{'='*60}")

        is_base = adapter_path is None
        try:
            model, tokenizer = load_model_with_adapter(
                adapter_path=None if is_base else adapter_path,
                base_only=is_base,
            )

            # Smoke eval
            smoke_rpt = None
            if smoke_file and smoke_file.exists():
                prefix_smoke = label.replace("/", "_").replace(" ", "_")
                smoke_rpt = evaluate_split(
                    model, tokenizer, smoke_file,
                    f"{prefix_smoke}_smoke", max_samples,
                )

            # Val eval
            prefix_val = label.replace("/", "_").replace(" ", "_")
            val_rpt = evaluate_split(
                model, tokenizer, val_file,
                f"{prefix_val}_val", max_samples,
            )

            score = compute_selection_score(val_rpt)
            gates = check_gates(val_rpt)

            results[label] = {
                "adapter_path": str(adapter_path) if adapter_path else "base_model",
                "selection_score": score,
                "gates": gates,
                "gates_all_passed": all(gates.values()),
                "smoke_accuracy": smoke_rpt.get("accuracy") if smoke_rpt else None,
                "val_accuracy": val_rpt.get("accuracy"),
                "val_malformed_rate": val_rpt.get("malformed_rate", 0),
                "val_over_refusal_rate": val_rpt.get("over_refusal_rate", 0),
                "by_difficulty": val_rpt.get("by_difficulty", {}),
                "by_tool": val_rpt.get("by_expected_tool", {}),
                "v02_tracked_errors": val_rpt.get("v02_tracked_errors", {}),
            }

            # Free GPU memory
            del model
            import torch
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[ERROR] Failed to evaluate {label}: {e}", file=sys.stderr)
            results[label] = {"error": str(e)}

    # Determine best
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if valid_results:
        best_label = max(valid_results, key=lambda k: valid_results[k]["selection_score"])
        best_score = valid_results[best_label]["selection_score"]
    else:
        best_label = None
        best_score = 0.0

    elapsed = time.time() - t_total

    # Build comparison report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_elapsed_seconds": round(elapsed, 1),
        "best_adapter": best_label,
        "best_selection_score": best_score,
        "adapters_evaluated": len(adapters),
        "adapters_succeeded": len(valid_results),
        "results": results,
        # Summary table for quick reading
        "summary_table": _build_summary_table(valid_results),
    }

    # Write report
    out_path = EVAL_OUTPUT_DIR / "comparison_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print summary
    _print_comparison_summary(report)

    return report


def _build_summary_table(results: dict[str, dict]) -> str:
    """Build a markdown comparison table."""
    lines = [
        "| Adapter | Score | Val Acc | Malformed | Over-Ref | Safety | Trap | Gates Pass |",
        "|---------|-------|---------|-----------|----------|--------|------|------------|",
    ]
    for label, r in sorted(results.items(), key=lambda x: -x[1].get("selection_score", 0)):
        gates_pass = "YES" if r.get("gates_all_passed") else "NO"
        by_d = r.get("by_difficulty", {})
        safety_acc = _pct(by_d.get("safety", {}))
        trap_acc = _pct(by_d.get("trap", {}))
        lines.append(
            f"| {label} | {r.get('selection_score', 0):.4f} "
            f"| {r.get('val_accuracy', 0):.1%} "
            f"| {r.get('val_malformed_rate', 0):.1%} "
            f"| {r.get('val_over_refusal_rate', 0):.1%} "
            f"| {safety_acc} "
            f"| {trap_acc} "
            f"| {gates_pass} |"
        )
    return "\n".join(lines)


def _pct(bucket: dict) -> str:
    t = bucket.get("total", 0)
    c = bucket.get("correct", 0)
    return f"{c/max(t,1):.0%}"


def _print_comparison_summary(report: dict) -> None:
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(report.get("summary_table", "(no data)"))
    print()
    print(f"Best adapter : {report['best_adapter']} (score={report['best_selection_score']:.4f})")
    print(f"Evaluated     : {report['adapters_evaluated']}/{report['adapters_succeeded']} succeeded")
    print(f"Total time   : {report['total_elapsed_seconds']:.0f}s")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compare multiple v03 router adapters")
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--smoke-file", type=Path, default=None)
    parser.add_argument("--base-model", action="store_true", help="Include base model (no LoRA)")
    parser.add_argument("--adapter", type=Path, action="append", dest="adapters",
                        help="Adapter directory (can specify multiple times)")
    parser.add_argument("--ckpt", type=Path, action="append", dest="checkpoints",
                        help="Checkpoint directory (can specify multiple times)")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    # Build adapter list: [(label, path), ...]
    adapters: list[tuple[str, Path | None]] = []

    if args.base_model:
        adapters.append(("base_model", None))

    if args.adapters:
        for p in args.adapters:
            name = p.name
            adapters.append((name, p))

    if args.checkpoints:
        for p in args.checkpoints:
            name = p.name
            adapters.append((name, p))

    if not adapters:
        print("[ERROR] Specify at least --base-model or --adapter/--ckpt", file=sys.stderr)
        sys.exit(1)

    run_comparison(args.val_file, args.smoke_file, adapters, args.max_samples)


if __name__ == "__main__":
    main()
