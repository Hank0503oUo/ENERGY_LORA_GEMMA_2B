"""Convert V5 dispatch JSONL -> chat messages JSONL for LoRA training.

Input:  dispatch format with user_query + expected fields
Output: chat format with system/user/assistant messages

Assistant output: compact dispatch JSON that the LoRA model learns to predict.
"""
import json, sys, argparse
from pathlib import Path
from collections import Counter

# Import V5 config
import importlib.util
# Look for config relative to this script, or in v05/scripts
_here = Path(__file__).resolve().parent
cfg_path = _here / "00_config_v05.py"
if not cfg_path.exists():
    cfg_path = Path(r"G:\我的雲端硬碟\energy_lora_router_v05\scripts\00_config_v05.py")
spec = importlib.util.spec_from_file_location("v05cfg", cfg_path)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

SYSTEM_PROMPT = cfg.render_system_prompt()

def dispatch_to_assistant(exp: dict) -> dict:
    """Extract the dispatch decision into the assistant output format."""
    return {
        "dispatch_type": exp.get("dispatch_type", ""),
        "workflow_id": exp.get("workflow_id", "none"),
        "answerability": exp.get("answerability", ""),
        "locked_entities": exp.get("locked_entities", {}),
        "required_tools": exp.get("required_tools", []),
        "stop_conditions": exp.get("stop_conditions", []),
    }

def convert_file(input_path: Path, output_path: Path):
    """Convert dispatch JSONL to chat messages JSONL."""
    total = 0
    errors = 0
    dispatch_counts = Counter()
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"L{line_no}: JSON error: {e}")
                errors += 1
                continue
            
            exp = row.get("expected", {})
            user_query = row.get("user_query", "")
            conversation_ctx = row.get("conversation_context", [])
            
            # Build messages
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            # Add conversation context (previous turns)
            for ctx_msg in conversation_ctx:
                messages.append(ctx_msg)
            
            # Add current user query
            messages.append({"role": "user", "content": user_query})
            
            # Assistant output: dispatch decision
            assistant_output = dispatch_to_assistant(exp)
            messages.append({
                "role": "assistant",
                "content": json.dumps(assistant_output, ensure_ascii=False)
            })
            
            # Build training row
            dt = exp.get("dispatch_type", "unknown")
            dispatch_counts[dt] += 1
            
            training_row = {
                "messages": messages,
                "sample_id": row.get("sample_id", f"v5_{line_no}"),
                "user_role": row.get("user_role", ""),
                "expected_dispatch_type": dt,
                "expected_workflow_id": exp.get("workflow_id", ""),
                "expected_answerability": exp.get("answerability", ""),
                "difficulty": row.get("difficulty", "unknown"),
                "tags": row.get("tags", []),
            }
            
            fout.write(json.dumps(training_row, ensure_ascii=False) + "\n")
    
    print(f"Converted: {total} rows ({errors} errors)")
    print(f"dispatch_type: {dict(dispatch_counts)}")
    print(f"Output: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert V5 dispatch JSONL to chat training format")
    parser.add_argument("input", type=Path, help="Dispatch JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output chat JSONL")
    args = parser.parse_args()
    
    convert_file(args.input, args.output)

if __name__ == "__main__":
    main()
