# energy_lora_router_v06 — Dispatch Contract Recovery Round

## Status: READY TO TRAIN

V6 fixes the core issue from V05: **87% parse error rate** caused by system prompt / data / output contract misalignment.

## V05 Root Cause

The v05 system prompt defined the model as a **conversational NTU energy assistant** (natural language output), but training targets were **strict JSON dispatch objects**. The model mostly chose natural language, resulting in 87% parse errors.

## V6 Key Changes

### 1. Strict Dispatch-Only System Prompt
- Model is now a **dispatch classifier**, NOT a conversational assistant
- Explicitly forbids: natural language, markdown, code fences, explanations
- Requires: single JSON object with all 6 fields
- See `scripts/00_config_v06.py`

### 2. DCI Tools Formally Declared
- All 6 DCI document tools (`find_docs`, `grep_docs`, `read_doc_chunk`, `inspect_doc_context`, `count_doc_matches`, `search_docs`) now appear in the system prompt tool whitelist
- `document_search_dci` workflow is recognized as legitimate

### 3. Boundary Fix
- `unsupported_scope` now maps to `refusal` (not `no_evidence`)
- Automated consistency checking via `clean_boundaries_v06.py`

### 4. Format Curriculum (120 samples)
- Added to training data to teach JSON-only output
- Covers: short queries, document queries, single tool, workflow, refusal, clarify, no_evidence
- See `data/synth/v06_format_curriculum.jsonl`

### 5. Format Smoke Gate (24 samples)
- Small eval set that ONLY tests JSON format compliance
- Gate: if parse error > 10%, stop and fix prompt before full training
- See `data/processed/format_smoke_v06.jsonl`

## Data Summary

| Split | Count | Source |
|-------|-------|--------|
| train | 957 | 837 reprompted v05 + 120 format curriculum |
| val | 147 | reprompted v05 |
| smoke | 16 | reprompted v05 |
| format_smoke | 24 | new format-only gate |

## Key Files

| File | Purpose |
|------|---------|
| `scripts/00_config_v06.py` | Strict dispatch config + system prompt + enums |
| `scripts/train_lora.py` | LoRA training (updated for v06) |
| `scripts/evaluate_router.py` | Eval with format smoke gate support |
| `scripts/reprompt_v06_data.py` | Re-apply v06 prompt to v05 data |
| `scripts/clean_boundaries_v06.py` | Fix answerability/dispatch boundary |
| `scripts/format_curriculum_v06.py` | Generate format curriculum samples |
| `scripts/format_smoke_v06.py` | Generate format smoke test |
| `scripts/run_v06_pipeline.py` | Full pipeline runner |
| `v6_training_config.json` | Training configuration |
| `data/processed/harness_v06_manifest.json` | Data manifest |

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
```

## Recommended Training Sequence

### Phase 4: Small-Round Validation
1. Train 0.3-0.5 epoch
2. Run `format_smoke_v06` eval
3. If parse error > 10%: stop, fix prompt

### Phase 5: Full Training
1. Full 1 epoch
2. Format smoke gate
3. Full val eval

## V6 Acceptance Criteria

### Format Gate (must pass first)
- Parse error rate < 10%
- Format smoke accuracy > 90%

### Boundary Gate
- Refusal correctness > 85%
- Clarify accuracy > 80%
- No_evidence accuracy > 80%

### Dispatch Gate
- Single_tool accuracy > 75%
- Workflow_chain accuracy > 70%
- document_search_dci workflow has viable skeleton

## Eval Command (Colab)

```bash
# Format smoke gate first
python evaluate_router.py \
    --adapter outputs/gemma_dispatch_v06/adapter \
    --val-file data/processed/format_smoke_v06.jsonl \
    --prefix v06_format_smoke

# If gate passes, full eval
python evaluate_router.py \
    --adapter outputs/gemma_dispatch_v06/adapter \
    --val-file data/processed/val_v06_dispatch.jsonl \
    --smoke-file data/processed/smoke_v06_dispatch.jsonl \
    --format-smoke-file data/processed/format_smoke_v06.jsonl \
    --prefix v06
```

## Reference

- V6 recovery plan: `D:\idf優化\demo\docs\V6_DISPATCH_RECOVERY_PLAN.md`
- V05 source: `G:\我的雲端硬碟\energy_lora_router_v05`
