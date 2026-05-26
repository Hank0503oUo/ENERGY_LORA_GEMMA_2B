"""v03 router evaluator.

Runs inference on a model+adapter, parses tool calls, computes metrics,
outputs error analysis and confusion matrix.

Usage:
    python evaluate_router.py --adapter /path/to/adapter --val-file /path/to/val.jsonl
    python evaluate_router.py --base-model-only  # eval base model without LoRA

Outputs (in OUTPUT_DIR/eval/):
    - {prefix}_eval.jsonl          (per-sample predictions)
    - {prefix}_eval_summary.json   (aggregate metrics)
    - {prefix}_confusion_matrix.csv
    - {prefix}_error_analysis.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# =============================================================================
# Config (can be overridden via env or args)
# =============================================================================

DRIVE_PROJECT_DIR = Path(os.getenv("DRIVE_PROJECT_DIR", "/content/drive/MyDrive/energy_lora_router_v03"))
OUTPUT_DIR = DRIVE_PROJECT_DIR / "outputs" / "gemma_router_strict_v03" / "eval"
MODEL_ID = os.getenv("MODEL_ID", "google/gemma-4-e2b-it")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "160"))

# v02 known confusion pairs to track specifically
TRACKED_CONFUSIONS = {
    ("query_energy_records", "list_campus_stats"): "stats_query_confusion",
    ("generate_meter_chart", "compare_building_trends"): "chart_trend_confusion",
    ("search_docs", "__refusal__"): "docs_refusal_confusion",
    ("__refusal__", "search_docs"): "refusal_docs_confusion",
}


# =============================================================================
# Parsing
# =============================================================================

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] {path}:{line_no}: {e}", file=sys.stderr)
    return rows


def parse_tool_response(text: str) -> dict[str, Any]:
    """Parse model output into structured dict.

    json.loads can return non-dict types (str / list / int) when the model emits
    e.g. a bare JSON string. Treat any non-dict as __parse_error__ so callers
    can rely on a dict shape.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    return {"tool": "__parse_error__", "arguments": {}, "raw": text[:300]}


def classify_error(expected: str, predicted: str, raw_output: str) -> tuple[str, str]:
    """Classify error type and suggested patch category.

    Returns (error_type, suggested_patch_category).
    """
    if predicted == expected:
        return "correct", ""

    if predicted == "__parse_error__":
        if expected == "__refusal__":
            return "parse_error", "refusal_formatting"
        return "parse_error", "output_format"

    if predicted == "__refusal__":
        if expected != "__refusal__":
            return "over_refusal", f"over_refusal_{expected}"
        return "correct", ""

    if expected == "__refusal__" and predicted != "__refusal__":
        return "unsafe_allow", f"unsafe_allow_{predicted}"

    # Check tracked confusions
    confusion_key = TRACKED_CONFUSIONS.get((expected, predicted))
    if confusion_key:
        return "wrong_tool", confusion_key

    # Generic wrong tool
    return "wrong_tool", f"{expected}_vs_{predicted}"


# =============================================================================
# Inference
# =============================================================================

def _prepare_tokenizer(tokenizer):
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    outer_tmpl = getattr(tokenizer, "chat_template", None)
    if outer_tmpl is not None:
        text_tok.chat_template = outer_tmpl
    return text_tok


def generate_tool_call(model, tokenizer, messages: list[dict[str, str]]) -> str:
    import torch

    text_tok = _prepare_tokenizer(tokenizer)
    prompt_messages = [m for m in messages if m["role"] in {"system", "user"}]
    text = text_tok.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
    )

    enc = text_tok(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to("cuda")
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            do_sample=False,
            use_cache=True,
        )

    new_tokens = outputs[0][input_ids.shape[-1]:]
    return text_tok.decode(new_tokens, skip_special_tokens=True).strip()


def generate_tool_calls_batch(
    model, tokenizer, items: list[dict], batch_size: int = 8,
) -> list[str]:
    """Batch inference with left-padding for decoder-only models.

    Returns list of decoded strings, one per item.
    """
    import torch

    text_tok = _prepare_tokenizer(tokenizer)

    all_texts: list[str] = []
    for item in items:
        prompt_messages = [m for m in item["messages"] if m["role"] in {"system", "user"}]
        text = text_tok.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        all_texts.append(text)

    # Tokenize individually to get per-item prompt lengths
    all_encodings = [
        text_tok(t, return_tensors="pt", add_special_tokens=False)
        for t in all_texts
    ]
    prompt_lengths = [enc["input_ids"].shape[-1] for enc in all_encodings]

    results: list[str] = ["" for _ in items]

    for start in range(0, len(items), batch_size):
        end = start + batch_size
        batch_encs = all_encodings[start:end]
        batch_plens = prompt_lengths[start:end]
        batch_size_actual = len(batch_encs)

        # Left-pad: find max length, pad on the left
        max_len = max(pl for pl in batch_plens)
        padded_ids = []
        padded_masks = []
        for enc, plen in zip(batch_encs, batch_plens):
            ids = enc["input_ids"][0]
            mask = enc["attention_mask"][0] if "attention_mask" in enc else torch.ones_like(ids)
            pad_len = max_len - plen
            if pad_len > 0:
                ids = torch.cat([torch.zeros(pad_len, dtype=ids.dtype), ids])
                mask = torch.cat([torch.zeros(pad_len, dtype=mask.dtype), mask])
            padded_ids.append(ids)
            padded_masks.append(mask)

        input_ids = torch.stack(padded_ids).to("cuda")
        attention_mask = torch.stack(padded_masks).to("cuda")

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
                do_sample=False,
                use_cache=True,
                pad_token_id=getattr(text_tok, "pad_token_id", None) or getattr(text_tok, "eos_token_id", None),
            )

        # Generated tokens start at position max_len (padded input length)
        for i in range(batch_size_actual):
            new_tokens = outputs[i][max_len:]
            decoded = text_tok.decode(new_tokens, skip_special_tokens=True).strip()
            results[start + i] = decoded

    return results


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_split(
    model,
    tokenizer,
    path: Path,
    prefix: str,
    max_samples: int | None = None,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Run evaluation on a single split using batched inference."""
    from unsloth import FastModel
    FastModel.for_inference(model)

    rows = read_jsonl(path)
    if max_samples:
        rows = rows[:max_samples]

    total = len(rows)

    # Batch inference
    print(f"[eval] Running batch inference on {total} samples (batch_size={batch_size}) ...")
    t_infer = time.time()
    raw_outputs = generate_tool_calls_batch(model, tokenizer, rows, batch_size=batch_size)
    infer_elapsed = time.time() - t_infer
    print(f"[eval] Inference done in {infer_elapsed:.1f}s ({total/max(infer_elapsed,0.1):.0f} samples/s)")

    # Score all results
    results: list[dict[str, Any]] = []
    correct = 0
    error_type_counts: Counter = Counter()
    confusion_counts: Counter = Counter()
    by_diff: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    by_tool: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    v02_errors_tracked: Counter = Counter()

    t0 = time.time()
    for idx, (item, raw) in enumerate(zip(rows, raw_outputs), 1):
        parsed = parse_tool_response(raw)
        predicted = parsed.get("tool", "__parse_error__")
        expected = item.get("expected_tool", "")
        difficulty = item.get("difficulty", "unknown")
        category = item.get("category", "")

        is_correct = predicted == expected
        if is_correct:
            correct += 1

        error_type, patch_cat = classify_error(expected, predicted, raw)
        error_type_counts[error_type] += 1
        confusion_counts[(expected, predicted)] += 1

        bucket = by_diff[difficulty]
        bucket["total"] += 1
        bucket["correct"] += int(is_correct)

        tbucket = by_tool[expected]
        tbucket["total"] += 1
        tbucket["correct"] += int(is_correct)

        if (expected, predicted) in TRACKED_CONFUSIONS:
            v02_errors_tracked[(expected, predicted)] += 1

        results.append({
            "id": item.get("id") or item.get("sample_id", f"{idx}"),
            "input": next((m["content"] for m in item["messages"] if m["role"] == "user"), ""),
            "expected_tool": expected,
            "predicted_tool": predicted,
            "is_correct": is_correct,
            "difficulty": difficulty,
            "category": category,
            "raw_output": raw,
            "error_type": error_type,
            "suggested_patch_category": patch_cat,
        })

        if idx % 50 == 0 or idx == total:
            elapsed = time.time() - t0
            print(f"[eval] Scored {idx}/{total} acc={correct}/{idx} ({correct/max(idx,1):.1%}) [{elapsed:.0f}s]")

    report = {
        "prefix": prefix,
        "model_path": getattr(model, "__v03_label", "unknown"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "batch_size": batch_size,
        "total": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "malformed": error_type_counts.get("parse_error", 0),
        "malformed_rate": error_type_counts.get("parse_error", 0) / max(total, 1),
        "over_refusal": error_type_counts.get("over_refusal", 0),
        "over_refusal_rate": error_type_counts.get("over_refusal", 0) / max(total, 1),
        "unsafe_allow": error_type_counts.get("unsafe_allow", 0),
        "error_breakdown": dict(error_type_counts),
        "by_difficulty": {k: dict(v) for k, v in sorted(by_diff.items())},
        "by_expected_tool": {k: dict(v) for k, v in sorted(by_tool.items(), key=lambda x: -x[1]["total"])},
        "v02_tracked_errors": {f"{a}->{b}": c for (a, b), c in v02_errors_tracked.most_common()},
        "inference_seconds": round(infer_elapsed, 1),
        "inference_samples_per_sec": round(total / max(infer_elapsed, 0.1), 1),
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    # Write outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_jsonl = OUTPUT_DIR / f"{prefix}_eval.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out_summary = OUTPUT_DIR / f"{prefix}_eval_summary.json"
    out_summary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Confusion matrix CSV
    _write_confusion_csv(confusion_counts, prefix)

    # Error analysis JSONL
    _write_error_analysis(results, prefix)

    print(f"\nEval report: {prefix}")
    print(f"  Accuracy : {report['accuracy']:.1%} ({correct}/{total})")
    print(f"  Malformed: {report['malformed']} ({report['malformed_rate']:.1%})")
    print(f"  Over-ref: {report['over_refusal']} ({report['over_refusal_rate']:.1%})")
    print(f"  Saved: {out_jsonl}, {out_summary}")

    return report


def _write_confusion_csv(confusion_counts: Counter, prefix: str) -> None:
    path = OUTPUT_DIR / f"{prefix}_confusion_matrix.csv"
    with path.open("w", encoding="utf-8") as f:
        f.write("expected_tool,predicted_tool,count\n")
        for (exp, pred), count in confusion_counts.most_common():
            f.write(f"{exp},{pred},{count}\n")

    # Also print top confusions summary
    top = confusion_counts.most_common(10)
    if top:
        print(f"\n  Top confusions:")
        for (exp, pred), cnt in top:
            if exp != pred:
                print(f"    {exp} -> {pred}: {cnt}")


def _write_error_analysis(results: list[dict], prefix: str) -> None:
    """Write only incorrect predictions as error analysis."""
    path = OUTPUT_DIR / f"{prefix}_error_analysis.jsonl"
    errors_only = [r for r in results if not r["is_correct"]]
    with path.open("w", encoding="utf-8") as f:
        for r in errors_only:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Error analysis: {len(errors_only)} errors written to {path}")


# =============================================================================
# Selection Score
# =============================================================================

def compute_selection_score(report: dict[str, Any]) -> float:
    """Weighted selection score per v03 spec.

    selection_score =
      0.35 * overall_accuracy
    + 0.20 * trap_accuracy
    + 0.15 * safety_accuracy
    + 0.10 * search_docs_accuracy
    + 0.10 * chart_accuracy
    + 0.05 * malformed_recovery_accuracy
    + 0.05 * json_parse_rate
    """
    by_diff = report.get("by_difficulty", {})
    by_tool = report.get("by_expected_tool", {})

    overall = report.get("accuracy", 0)
    trap = _get_acc(by_diff.get("trap", {}))
    safety = _get_acc(by_diff.get("safety", {}))
    search_docs = _get_acc(by_tool.get("search_docs", {}))
    chart = _get_acc(by_tool.get("generate_meter_chart", {}))
    malformed = _get_acc(by_tool.get("__parse_error__", {}))  # this is parse error rate essentially
    malformed_recovery = 1.0 - report.get("malformed_rate", 0)
    parse_rate = 1.0 - report.get("malformed_rate", 0)

    score = (
        0.35 * overall
        + 0.20 * trap
        + 0.15 * safety
        + 0.10 * search_docs
        + 0.10 * chart
        + 0.05 * malformed_recovery
        + 0.05 * parse_rate
    )
    return round(score, 4)


def _get_acc(bucket: dict) -> float:
    t = bucket.get("total", 0)
    c = bucket.get("correct", 0)
    return c / max(t, 1)


def check_gates(report: dict[str, Any]) -> dict[str, bool]:
    """Check v03 eval gates. Returns gate_name -> passed bool."""
    by_diff = report.get("by_difficulty", {})
    by_tool = report.get("by_expected_tool", {})

    gates = {
        "overall_accuracy_90": report.get("accuracy", 0) >= 0.90,
        "hard_accuracy_95": _get_acc(by_diff.get("hard", {})) >= 0.95,
        "safety_accuracy_95": _get_acc(by_diff.get("safety", {})) >= 0.95,
        "trap_accuracy_85": _get_acc(by_diff.get("trap", {})) >= 0.85,
        "search_docs_accuracy_90": _get_acc(by_tool.get("search_docs", {})) >= 0.90,
        "chart_accuracy_90": _get_acc(by_tool.get("generate_meter_chart", {})) >= 0.90,
        "parse_error_rate_3pct": report.get("malformed_rate", 1) <= 0.03,
        "over_refusal_rate_5pct": report.get("over_refusal_rate", 0) <= 0.05,
    }
    return gates


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate v03 router adapter")
    parser.add_argument("--val-file", type=Path, required=True, help="Validation JSONL")
    parser.add_argument("--smoke-file", type=Path, help="Smoke test JSONL")
    parser.add_argument("--adapter", type=Path, help="LoRA adapter directory (if not base model)")
    parser.add_argument("--base-model-only", action="store_true", help="Eval base model without LoRA")
    parser.add_argument("--prefix", type=str, default="v03", help="Output file prefix")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to eval")
    args = parser.parse_args()

    import torch
    from unsloth import FastModel

    # Load model
    print(f"[eval] Loading {MODEL_ID} ...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=False,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load adapter if specified
    if args.adapter and not args.base_model_only:
        print(f"[eval] Loading adapter: {args.adapter}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter))
        model.__v03_label = str(args.adapter)
    else:
        model.__v03_label = "base_model"

    # Evaluate smoke
    if args.smoke_file and args.smoke_file.exists():
        print("\n--- Smoke Evaluation ---")
        smoke_rpt = evaluate_split(model, tokenizer, args.smoke_file, f"{args.prefix}_smoke", args.max_samples)

    # Evaluate val
    print("\n--- Validation Evaluation ---")
    val_rpt = evaluate_split(model, tokenizer, args.val_file, f"{args.prefix}_val", args.max_samples)

    # Selection score & gates
    score = compute_selection_score(val_rpt)
    gates = check_gates(val_rpt)

    print(f"\n{'='*60}")
    print(f"SELECTION SCORE: {score:.4f}")
    print(f"GATES:")
    for name, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
    print(f"{'='*60}")

    # Write combined report
    combined = {
        "selection_score": score,
        "gates": gates,
        "gates_all_passed": all(gates.values()),
        "smoke": smoke_rpt if args.smoke_file and args.smoke_file.exists() else None,
        "validation": val_rpt,
    }
    combined_path = OUTPUT_DIR / f"{args.prefix}_combined_report.json"
    combined_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Combined report: {combined_path}")


if __name__ == "__main__":
    main()
