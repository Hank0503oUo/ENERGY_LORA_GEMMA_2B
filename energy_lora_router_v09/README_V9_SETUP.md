# energy_lora_router_v09

## Status

V9 is copied from the full V8 folder and updates only the routing repair layer.

## Main Fixes

- `no_evidence` now has priority before `single_tool` and `workflow_chain`.
- Evaluation default `MAX_NEW_TOKENS` is raised from `160` to `512`.
- The prompt no longer uses placeholder wording such as `workflow_id=任務模式`.
- A deterministic pre-gate catches building energy queries with years outside `2017-2023`.
- V9 adds focused contrast samples for `no_evidence`, `clarify_needed`, `single_tool`, and `workflow_chain`.

## New Files

- `scripts/00_config_v09.py`
- `scripts/run_v09_pipeline.py`
- `v9_training_config.json`
- `README_V9_SETUP.md`
- `notebooks/router_strict_lora_colab_v09.ipynb`

## Generate V9 Data

```bash
python scripts/run_v09_pipeline.py
```

Expected outputs:

- `data/processed/train_v09_dispatch.jsonl`
- `data/processed/val_v09_dispatch.jsonl`
- `data/processed/smoke_v09_dispatch.jsonl`
- `data/processed/format_smoke_v09.jsonl`
- `data/processed/harness_v09_manifest.json`
- `data/synth/v09_boundary_curriculum.jsonl`

## Train

```bash
export DRIVE_PROJECT_DIR=/content/drive/MyDrive/energy_lora_router_v09
python scripts/train_lora.py
```

## Evaluate

```bash
export DRIVE_PROJECT_DIR=/content/drive/MyDrive/energy_lora_router_v09
export MAX_NEW_TOKENS=512
python scripts/evaluate_router.py \
  --adapter outputs/gemma_dispatch_v09/adapter \
  --val-file data/processed/val_v09_dispatch.jsonl \
  --smoke-file data/processed/smoke_v09_dispatch.jsonl \
  --format-smoke-file data/processed/format_smoke_v09.jsonl \
  --prefix v09_final
```

## V9 Target

The first target is not production accuracy yet. The first gate to fix is:

- parse error rate below 10%
- `no_evidence` above 80%
- format smoke above 90%

After those stabilize, tune `workflow_chain` vs `single_tool`.
