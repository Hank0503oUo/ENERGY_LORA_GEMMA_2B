"""v05 dispatch dataset validator.
Checks schema compliance for agent dispatch training data.

Usage: python validate_dispatch_dataset.py <path.jsonl>
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Import V5 config enums
import importlib.util
cfg_path = Path(__file__).resolve().parent / "00_config_v05.py"
spec = importlib.util.spec_from_file_location("v05cfg", cfg_path)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

VALID_DISPATCH_TYPES = cfg.VALID_DISPATCH_TYPES
VALID_WORKFLOW_IDS = cfg.VALID_WORKFLOW_IDS
VALID_ANSWERABILITY = cfg.VALID_ANSWERABILITY
VALID_DIFFICULTIES = cfg.VALID_DIFFICULTIES
VALID_TOOL_NAMES = getattr(cfg, "ALL_VALID_TOOLS", cfg.VALID_TOOL_NAMES)

class DispatchReport:
    def __init__(self, path: Path):
        self.path = path
        self.errors: list[dict] = []
        self.warnings: list[dict] = []
        self.total = 0
        self.dispatch_counts: Counter = Counter()
        self.workflow_counts: Counter = Counter()
        self.answerability_counts: Counter = Counter()

    def add_error(self, line_no, code, msg, row=None):
        self.errors.append({"line": line_no, "code": code, "message": msg})
    def add_warning(self, line_no, code, msg, row=None):
        self.warnings.append({"line": line_no, "code": code, "message": msg})

    def summary(self):
        lines = [
            f"{'='*60}", f"Validation: {self.path.name}",
            f"  Total: {self.total}  Errors: {len(self.errors)}  Warnings: {len(self.warnings)}",
        ]
        if self.errors:
            lines.append("  ERRORS:")
            for e in self.errors[:15]:
                lines.append(f"    L{e['line']:>5} [{e['code']}] {e['message']}")
            if len(self.errors) > 15:
                lines.append(f"    ... +{len(self.errors)-15} more")
        if self.warnings:
            lines.append("  WARNINGS:")
            for w in self.warnings[:10]:
                lines.append(f"    L{w['line']:>5} [{w['code']}] {w['message']}")
        lines.extend([
            f"  dispatch_type: {dict(self.dispatch_counts)}",
            f"  workflow_id:   {dict(self.workflow_counts)}",
            f"  answerability: {dict(self.answerability_counts)}",
            f"{'='*60}",
        ])
        return "\n".join(lines)

def validate_file(path: Path) -> DispatchReport:
    report = DispatchReport(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            report.total += 1
            try: row = json.loads(line)
            except json.JSONDecodeError as e:
                report.add_error(line_no, "JSON_PARSE", str(e)); continue

            # Required top-level fields
            for field in ["sample_id", "user_role", "user_query", "expected"]:
                if field not in row:
                    report.add_error(line_no, "MISSING_FIELD", f"'{field}' missing"); continue

            exp = row.get("expected", {})

            # dispatch_type
            dt = exp.get("dispatch_type", "")
            if dt not in VALID_DISPATCH_TYPES:
                report.add_error(line_no, "INVALID_DISPATCH_TYPE", f"'{dt}' not in {VALID_DISPATCH_TYPES}")
            report.dispatch_counts[dt] += 1

            # workflow_id
            wid = exp.get("workflow_id", "")
            if wid not in VALID_WORKFLOW_IDS:
                report.add_error(line_no, "INVALID_WORKFLOW_ID", f"'{wid}' not in {VALID_WORKFLOW_IDS}")
            report.workflow_counts[wid] += 1

            # answerability
            ans = exp.get("answerability", "")
            if ans not in VALID_ANSWERABILITY:
                report.add_error(line_no, "INVALID_ANSWERABILITY", f"'{ans}' not in {VALID_ANSWERABILITY}")
            report.answerability_counts[ans] += 1

            # difficulty
            diff = row.get("difficulty", "")
            if diff not in VALID_DIFFICULTIES:
                report.add_error(line_no, "INVALID_DIFFICULTY", f"'{diff}'")

            # locked_entities must be a dict with building_names, years, metrics
            le = exp.get("locked_entities", {})
            if not isinstance(le, dict):
                report.add_error(line_no, "BAD_LOCKED_ENTITIES", "locked_entities must be dict")
            for key in ["building_names", "years", "metrics"]:
                if key not in le:
                    report.add_warning(line_no, "MISSING_ENTITY_KEY", f"locked_entities missing '{key}'")

            # required_tools
            rt = exp.get("required_tools", [])
            if isinstance(rt, list):
                for ti, t in enumerate(rt):
                    if not isinstance(t, dict):
                        report.add_error(line_no, "BAD_TOOL_ENTRY", f"required_tools[{ti}] not dict")
                        continue
                    tool_name = t.get("tool", "")
                    if tool_name and tool_name not in VALID_TOOL_NAMES:
                        report.add_error(line_no, "UNKNOWN_TOOL", f"required_tools[{ti}].tool='{tool_name}'")
                    if "purpose" not in t:
                        report.add_warning(line_no, "MISSING_PURPOSE", f"required_tools[{ti}] missing purpose")
            else:
                report.add_error(line_no, "BAD_REQUIRED_TOOLS", "required_tools must be list")

            # stop_conditions
            sc = exp.get("stop_conditions", [])
            if not sc:
                report.add_warning(line_no, "EMPTY_STOP_CONDITIONS", "stop_conditions empty")

            # intent_tool check
            it = exp.get("intent_tool")
            if it and it not in VALID_TOOL_NAMES:
                report.add_warning(line_no, "UNKNOWN_INTENT_TOOL", f"intent_tool='{it}'")

            # Consistency: dispatch_type alignment
            if dt == "refusal" and ans not in ("unsupported_scope","unsupported_capability","unsafe_operation"):
                report.add_warning(line_no, "REFUSAL_ANSWERABILITY_MISMATCH", f"refusal but answerability={ans}")
            if dt == "single_tool" and ans not in ("answerable_single_tool",):
                report.add_warning(line_no, "SINGLE_TOOL_ANSWERABILITY_MISMATCH", f"single_tool but answerability={ans}")
            if dt == "clarify_needed" and ans not in ("missing_required_arguments","ambiguous_reference"):
                report.add_warning(line_no, "CLARIFY_ANSWERABILITY_MISMATCH", f"clarify_needed but answerability={ans}")
            if dt in ("single_tool","workflow_chain") and not rt:
                report.add_warning(line_no, "NO_TOOLS_FOR_ACTIONABLE", f"{dt} but required_tools empty")

            # final_behavior
            if "final_behavior" not in exp:
                report.add_warning(line_no, "MISSING_FINAL_BEHAVIOR", "no final_behavior")

    return report

def main():
    parser = argparse.ArgumentParser(description="Validate v05 dispatch JSONL")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    all_valid = True
    for fp in args.files:
        if not fp.exists():
            print(f"ERROR: {fp} not found", file=sys.stderr)
            continue
        rpt = validate_file(fp)
        print(rpt.summary())
        if rpt.errors:
            all_valid = False

    if args.strict and not all_valid:
        sys.exit(1)

if __name__ == "__main__":
    main()
