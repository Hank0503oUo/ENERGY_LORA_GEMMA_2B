# energy_lora_router_v05 — Agent Dispatch Training

## Status: ✅ READY TO TRAIN

1000 dispatch training samples generated, validated, converted to chat format, and split into train/val/smoke.

## Architecture

```text
V05 is NOT another expected_tool router.
It predicts: dispatch_type / workflow_id / entity lock / required_tools / stop_conditions
```

**Model output** (what LoRA learns):
```json
{
  "dispatch_type": "single_tool|workflow_chain|clarify_needed|no_evidence|refusal",
  "workflow_id": "single_building_year_status|building_hotspot_improvement|...",
  "locked_entities": {"building_names": ["<BUILDING_A>"], "years": [2017]},
  "required_tools": [{"tool": "query_energy_records", "arguments": {...}}],
  "stop_conditions": ["if_energy_values_missing_stop_before_strategy"]
}
```

## Data Pipeline

```text
Generator (gen_v05_1000)           # Template-based synthesis
  ↓
validate_dispatch_dataset.py       # Schema validation (0 errors)
  ↓
convert_dispatch_to_chat.py        # Dispatch → chat messages
  ↓
Manual split                       # Stratified 837/147/16
```

## Key Files

| File | Purpose |
|---|---|
| `data/processed/train_v05.jsonl` | 837 dispatch chat samples |
| `data/processed/val_v05.jsonl` | 147 dispatch chat samples |
| `data/processed/smoke_v05.jsonl` | 16 smoke test samples |
| `data/processed/harness_v05_manifest.json` | Data manifest |
| `data/synth/v05_dispatch_gen1000.jsonl` | Raw 1000 dispatch samples |
| `scripts/00_config_v05.py` | 7-tool config + system prompt + enums |
| `scripts/validate_dispatch_dataset.py` | Dispatch format validator |
| `scripts/convert_dispatch_to_chat.py` | Dispatch → chat converter |
| `scripts/evaluate_router.py` | Eval (dispatch-aware patched) |
| `notebooks/router_strict_lora_colab_v05.ipynb` | Colab training notebook |
| `deepseek_v05_dispatch_generation_prompt.md` | DeepSeek prompt (reference) |
| `deepseek_v05_dispatch_prompt.md` | Copied from demo/docs for reference |

## Dispatch Type Distribution

| Type | Train | Purpose |
|---|---|---|
| dispatch_workflow_chain | 279 | Multi-step query → strategy |
| dispatch_single_tool | 188 | Direct tool call |
| dispatch_no_evidence | 130 | Missing data → stop |
| dispatch_refusal | 129 | Unsafe/out-of-scope |
| dispatch_clarify | 111 | Ambiguous → ask user |

## 7-Tool Catalogue (V5 Frozen Spec)

```text
query_energy_records
list_campus_stats
get_top_energy_buildings
detect_energy_anomalies
run_openbse_hybrid_counterfactual
openbse_hvac_breakdown
recommend_adaptive_strategies
```

Plus DCI tools (for document_search_dci workflow): `find_docs`, `grep_docs`, `read_doc_chunk`, `inspect_doc_context`, `count_doc_matches`, `search_docs`

## Training Parameters

```text
MODEL_ID             = google/gemma-4-e2b-it
LOAD_IN_4BIT         = false
MAX_SEQ_LENGTH       = 2048
LORA_R / ALPHA       = 32 / 64
TRAIN_BATCH_SIZE     = 8
GRAD_ACCUM_STEPS     = 4
LEARNING_RATE        = 3e-5
NUM_TRAIN_EPOCHS     = 1
SAVE/EVAL_STEPS      = 40 / 40
```

## Excluded from V5

- ❌ calibrate_sensitivity (maintenance tool, out of scope)
- ❌ record_strategy / confirm_strategy_adoption / check_strategy_status
- ❌ map_energy_semantics / list_rtem_sources
- ❌ rank_energy_buildings_across_years / compare_energy_usage / compare_building_trends
- ❌ classify_anomaly / diagnose_energy_anomaly
- ❌ validate_strategy_openbse / run_pvid / compare_actual_predicted
- ❌ seasonal_strategies / optimize_energy_portfolio
- ❌ correlate_algorithms

## Reference

- V5 demo spec: `D:\idf優化\demo\docs\V5_DEMO_SPEC.md`
- V5 dispatch prompt: `D:\idf優化\demo\docs\V5_DEEPSEEK_DISPATCH_DATA_PROMPT.md`
