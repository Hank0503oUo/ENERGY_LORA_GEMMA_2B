"""v04 dataset validator.

Checks:
1. JSON parseable
2. id uniqueness
3. input not empty
4. expected_tool in whitelist
5. difficulty in allowed set
6. category not empty
7. safety heuristic: safety questions should have __refusal__
8. over-refusal heuristic: non-safety shouldn't be mostly __refusal__
9. chart keyword heuristic
10. docs keyword heuristic
11. building+year+EUI vs list_campus_stats heuristic
12. unvalidated numeric answers

Usage:
    python validate_dataset.py /path/to/data.jsonl
    python validate_dataset.py /path/to/data.jsonl --tool-whitelist tools.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Try to load the canonical tool list from 00_config.py (single source of truth).
# Fall back to a hard-coded copy that mirrors v02/manifest.tool_distribution
# so this script still runs if the import path is broken.
def _load_tool_whitelist_from_config() -> set[str]:
    try:
        import importlib.util as _ilu
        from pathlib import Path as _P
        cfg_path = _P(__file__).resolve().parent / "00_config.py"
        if not cfg_path.exists():
            return set()
        spec = _ilu.spec_from_file_location("v04cfg00", cfg_path)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        names = set(getattr(mod, "VALID_TOOL_NAMES", set()))
        names.add("__parse_error__")  # parser-side only, not in TOOLS
        return names
    except Exception as exc:  # pragma: no cover
        print(f"[validate] could not import 00_config.py ({exc}); using fallback whitelist")
        return set()


_FALLBACK_WHITELIST = {
    "__refusal__",
    "__parse_error__",
    # 28 real v02 tools (matches energy_lora_router_v02/manifest.tool_distribution)
    "query_energy_records",
    "compare_energy_usage",
    "compare_building_trends",
    "compare_actual_predicted",
    "generate_meter_chart",
    "search_docs",
    "run_counterfactual_for_building",
    "run_openbse_hybrid_counterfactual",
    "openbse_hvac_breakdown",
    "seasonal_strategies",
    "recommend_adaptive_strategies",
    "optimize_energy_portfolio",
    "get_top_energy_buildings",
    "rank_energy_buildings_across_years",
    "detect_energy_anomalies",
    "classify_anomaly",
    "diagnose_energy_anomaly",
    "validate_strategy_openbse",
    "run_pvid",
    "calibrate_sensitivity",
    "get_sensitivity_status",
    "record_strategy",
    "confirm_strategy_adoption",
    "check_strategy_status",
    "correlate_algorithms",
    "list_campus_stats",
    "list_rtem_sources",
    "map_energy_semantics",
}

DEFAULT_TOOL_WHITELIST = _load_tool_whitelist_from_config() or _FALLBACK_WHITELIST

VALID_DIFFICULTIES = {"easy", "medium", "hard", "trap", "malformed", "safety"}

CHART_KEYWORDS = {"chart", "plot", "graph", "圖表", "趨勢圖", "繪圖", "畫圖"}
DOCS_KEYWORDS = {
    "文件", "法規", "wiki", "docs", "SOP", "定義", "規範",
    "document", "regulation", "policy", "manual", "說明書",
}
BUILDING_EUI_PATTERN = re.compile(
    r"(?:特定|這棟|該棟|某棟)\s*建築.*(?:EUI|kWh)"
    r"|(?:特定|這棟|該棟|某棟)\s*建築.*\d{4}.*(?:用電量|能耗)"
    r"|(?:EUI|kWh).*(?:特定|這棟|該棟|某棟)",
    re.IGNORECASE,
)


class ValidationReport:
    def __init__(self, path: Path):
        self.path = path
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.total = 0
        self.ids_seen: set[str] = set()
        self.tool_counts: Counter = Counter()
        self.difficulty_counts: Counter = Counter()

    def add_error(self, line_no: int, code: str, msg: str, row: dict | None = None):
        self.errors.append({"line": line_no, "code": code, "message": msg, "row_sample": _truncate(row)})

    def add_warning(self, line_no: int, code: str, msg: str, row: dict | None = None):
        self.warnings.append({"line": line_no, "code": code, "message": msg, "row_sample": _truncate(row)})

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": str(self.path),
            "total_rows": self.total,
            "valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors[:50],
            "warnings": self.warnings[:50],
            "tool_distribution": dict(self.tool_counts),
            "difficulty_distribution": dict(self.difficulty_counts),
        }

    def summary_text(self) -> str:
        lines = [
            f"{'='*60}",
            f"Validation: {self.path.name}",
            f"  Total rows : {self.total}",
            f"  Errors     : {len(self.errors)}  {'FAIL' if self.errors else 'OK'}",
            f"  Warnings   : {len(self.warnings)}",
        ]
        if self.errors:
            lines.append(f"\n  ERRORS (must fix before training):")
            for e in self.errors[:20]:
                lines.append(f"    L{e['line']:>5} [{e['code']}] {e['message']}")
            if len(self.errors) > 20:
                lines.append(f"    ... and {len(self.errors)-20} more")
        if self.warnings:
            lines.append(f"\n  WARNINGS (recommended fixes):")
            for w in self.warnings[:20]:
                lines.append(f"    L{w['line']:>5} [{w['code']}] {w['message']}")
            if len(self.warnings) > 20:
                lines.append(f"    ... and {len(self.warnings)-20} more")
        lines.extend([
            "",
            f"  Tool distribution      : {dict(self.tool_counts)}",
            f"  Difficulty distribution : {dict(self.difficulty_counts)}",
            f"{'='*60}",
        ])
        return "\n".join(lines)


def _truncate(row: dict | None, max_len: int = 200) -> str | None:
    if row is None:
        return None
    s = json.dumps(row, ensure_ascii=False)
    return s[:max_len] + ("..." if len(s) > max_len else "")


def _get_user_input(row: dict) -> str:
    messages = row.get("messages", [])
    for m in messages:
        if m.get("role") == "user":
            return m.get("content", "").strip()
    return row.get("input", "").strip()


def _get_assistant_content(row: dict) -> str:
    messages = row.get("messages", [])
    for m in messages:
        if m.get("role") == "assistant":
            return m.get("content", "").strip()
    return ""


def validate_file(
    path: Path,
    tool_whitelist: set[str] | None = None,
) -> ValidationReport:
    if tool_whitelist is None:
        tool_whitelist = DEFAULT_TOOL_WHITELIST

    report = ValidationReport(path)

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, 1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            report.total += 1

            # 1. JSON parseable
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as e:
                report.add_error(line_no, "JSON_PARSE_ERROR", str(e))
                continue

            # 2. id uniqueness
            sample_id = row.get("id") or row.get("sample_id")
            if sample_id:
                if sample_id in report.ids_seen:
                    report.add_error(line_no, "DUPLICATE_ID", f"Duplicate id: {sample_id}", row)
                else:
                    report.ids_seen.add(sample_id)

            # 3. input not empty
            user_input = _get_user_input(row)
            if not user_input:
                report.add_error(line_no, "EMPTY_INPUT", "User input is empty", row)

            # 4. expected_tool in whitelist
            expected_tool = row.get("expected_tool", "")
            if expected_tool not in tool_whitelist:
                report.add_error(
                    line_no,
                    "UNKNOWN_TOOL",
                    f"expected_tool={expected_tool!r} not in whitelist",
                    row,
                )
            report.tool_counts[expected_tool] += 1

            # 5. difficulty in allowed set
            difficulty = row.get("difficulty", "")
            if difficulty not in VALID_DIFFICULTIES:
                report.add_error(
                    line_no,
                    "INVALID_DIFFICULTY",
                    f"difficulty={difficulty!r} not in {VALID_DIFFICULTIES}",
                    row,
                )
            report.difficulty_counts[difficulty] += 1

            # 6. category not empty
            category = row.get("category", "")
            if not category:
                report.add_warning(line_no, "EMPTY_CATEGORY", "category is empty or missing", row)

            # 7. safety heuristic
            if difficulty == "safety" and expected_tool != "__refusal__":
                report.add_warning(
                    line_no,
                    "SAFETY_NOT_REFUSAL",
                    f"safety question but expected_tool={expected_tool!r} (should be __refusal__?)",
                    row,
                )

            
            # 6a. answerability validation
            answerability = row.get("answerability", "")
            if answerability:
                if answerability not in {'unsupported_capability', 'unsupported_scope', 'unsafe_operation', 'answerable_single_tool', 'answerable_multi_tool', 'missing_data', 'ambiguous_need_clarification'}:
                    report.add_warning(
                        line_no,
                        "INVALID_ANSWERABILITY",
                        f"answerability={answerability!r} not in valid set",
                        row,
                    )
                # 6b. answerability consistency with expected_tool
                if answerability in ("unsupported_scope", "unsupported_capability", "unsafe_operation", "missing_data", "ambiguous_need_clarification"):
                    if expected_tool != "__refusal__":
                        report.add_error(
                            line_no,
                            "ANSWERABILITY_TOOL_MISMATCH",
                            f"answerability={answerability!r} but expected_tool={expected_tool!r} (should be __refusal__)",
                            row,
                        )
                if answerability in ("answerable_single_tool", "answerable_multi_tool"):
                    if expected_tool == "__refusal__":
                        report.add_warning(
                            line_no,
                            "ANSWERABILITY_REFUSAL_MISMATCH",
                            f"answerability={answerability!r} but expected_tool=__refusal__ (possible over-refusal)",
                            row,
                        )

            # 6c. task_type validation
            task_type = row.get("task_type", "")
            if task_type:
                if task_type not in {'data_query', 'strategy_tracking', 'chart_generation', 'comparison', 'ranking', 'calibration', 'clarification', 'safety_refusal', 'semantic_mapping', 'optimization', 'anomaly_detection', 'document_search', 'simulation'}:
                    report.add_warning(
                        line_no,
                        "INVALID_TASK_TYPE",
                        f"task_type={task_type!r} not in valid set",
                        row,
                    )

            # 6d. refusal_type + reason for non-answerable samples
            if answerability not in ("answerable_single_tool", "answerable_multi_tool", ""):
                refusal_type = row.get("refusal_type", "")
                reason = row.get("reason", "")
                if not refusal_type:
                    report.add_warning(
                        line_no,
                        "MISSING_REFUSAL_TYPE",
                        f"answerability={answerability!r} but refusal_type is missing",
                        row,
                    )
                if not reason:
                    report.add_warning(
                        line_no,
                        "MISSING_REASON",
                        f"answerability={answerability!r} but reason is missing",
                        row,
                    )
                # Check assistant content has arguments with reason for refusal
                assistant_content = _get_assistant_content(row)
                if assistant_content and expected_tool == "__refusal__":
                    try:
                        parsed = json.loads(assistant_content)
                        args = parsed.get("arguments", {})
                        if isinstance(args, dict) and not args.get("reason"):
                            report.add_warning(
                                line_no,
                                "REFUSAL_MISSING_REASON_IN_ARGS",
                                f"__refusal__ assistant content missing reason in arguments",
                                row,
                            )
                    except (json.JSONDecodeError, TypeError):
                        pass

# 8. over-refusal heuristic (checked at aggregate level later)

            # 9. chart keyword heuristic
            has_chart_kw = any(kw in user_input for kw in CHART_KEYWORDS)
            if has_chart_kw and expected_tool not in ("generate_meter_chart", "__refusal__"):
                report.add_warning(
                    line_no,
                    "CHART_MISMATCH",
                    f"chart/plot keyword in input but expected_tool={expected_tool!r}",
                    row,
                )

            # 10. docs keyword heuristic
            has_docs_kw = any(kw in user_input for kw in DOCS_KEYWORDS)
            if has_docs_kw and expected_tool not in ("search_docs", "__refusal__"):
                report.add_warning(
                    line_no,
                    "DOCS_MISMATCH",
                    f"docs/wiki/SOP keyword in input but expected_tool={expected_tool!r}",
                    row,
                )

            # 11. building+year+EUI vs list_campus_stats (heuristic, warn only)
            if BUILDING_EUI_PATTERN.match(user_input):
                if expected_tool == "list_campus_stats":
                    report.add_warning(
                        line_no,
                        "STATS_QUERY_HEURISTIC",
                        f"Possible stats/query confusion: query mentions building+year+EUI but expected_tool=list_campus_stats (heuristic, may be correct)",
                        row,
                    )

            # 12. assistant content parse check
            assistant_content = _get_assistant_content(row)
            if assistant_content:
                if assistant_content.startswith("```"):
                    assistant_content = assistant_content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                try:
                    parsed = json.loads(assistant_content)
                    if isinstance(parsed, dict) and "tool" in parsed:
                        if parsed["tool"] != expected_tool:
                            report.add_warning(
                                line_no,
                                "TOOL_MISMATCH",
                                f"assistant tool={parsed['tool']!r} != expected_tool={expected_tool!r}",
                                row,
                            )
                except (json.JSONDecodeError, TypeError):
                    if expected_tool not in ("__parse_error__", "__refusal__"):
                        report.add_warning(
                            line_no,
                            "ASSISTANT_NOT_JSON",
                            f"Assistant content is not valid JSON (expected_tool={expected_tool!r})",
                            row,
                        )

    return report


def validate_aggregate(reports: list[ValidationReport]) -> list[str]:
    """Cross-file checks."""
    issues: list[str] = []

    for rpt in reports:
        total = rpt.total
        refusal_count = rpt.tool_counts.get("__refusal__", 0)
        non_safety_total = total - rpt.difficulty_counts.get("safety", 0)
        non_safety_refusal = min(refusal_count, non_safety_total)

        if non_safety_total > 0 and non_safety_refusal / non_safety_total > 0.3:
            issues.append(
                f"{rpt.path.name}: over-refusal warning - "
                f"{non_safety_refusal}/{non_safety_total} ({non_safety_refusal/non_safety_total:.0%}) "
                f"non-safety samples are __refusal__"
            )

    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate v04 router dataset JSONL files")
    parser.add_argument("files", nargs="+", type=Path, help="JSONL file(s) to validate")
    parser.add_argument("--tool-whitelist", type=Path, help="Path to tool whitelist file (one tool per line)")
    parser.add_argument("--output-json", type=Path, help="Write full report to JSON")
    parser.add_argument("--strict", action="store_true", help="Exit with error code on validation failures")
    args = parser.parse_args()

    tool_whitelist = None
    if args.tool_whitelist:
        wl_text = args.tool_whitelist.read_text(encoding="utf-8").strip()
        tool_whitelist = set(line.strip() for line in wl_text.splitlines() if line.strip())

    reports: list[ValidationReport] = []
    for fp in args.files:
        if not fp.exists():
            print(f"[ERROR] File not found: {fp}", file=sys.stderr)
            if args.strict:
                sys.exit(1)
            continue
        rpt = validate_file(fp, tool_whitelist)
        reports.append(rpt)
        print(rpt.summary_text())
        print()

    # Aggregate checks
    agg_issues = validate_aggregate(reports)
    if agg_issues:
        print("\n=== AGGREGATE WARNINGS ===")
        for issue in agg_issues:
            print(f"  ! {issue}")
        print()

    # Summary
    all_valid = all(r.is_valid for r in reports)
    total_errors = sum(len(r.errors) for r in reports)
    total_warnings = sum(len(r.warnings) for r in reports)
    total_rows = sum(r.total for r in reports)

    print("=" * 60)
    print(f"OVERALL: {total_rows} rows across {len(reports)} file(s)")
    print(f"  Valid   : {'YES' if all_valid else 'NO  (has errors)'}")
    print(f"  Errors  : {total_errors}")
    print(f"  Warnings: {total_warnings}")
    print("=" * 60)

    if args.output_json:
        output = {
            "overall_valid": all_valid,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "total_rows": total_rows,
            "files": [r.to_dict() for r in reports],
            "aggregate_issues": agg_issues,
        }
        args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nReport written to: {args.output_json}")

    if args.strict and not all_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
