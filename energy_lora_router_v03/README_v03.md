# energy_lora_router_v03

NTU campus energy assistant tool-router LoRA v03 training pipeline.

## What Changed from v02

| Item | v02 | v03 |
|------|-----|-----|
| Validation set | 38 rows | 300+ rows (stratified) |
| Training epochs | 4 (overfit at 3-4) | 2 (conservative) |
| LoRA rank/alpha | 16/16 | 32/64 |
| Learning rate | 2e-4 | 1e-4 |
| Adapter selection | Final only | Selection score + gate checks |
| Evaluation | Accuracy only | Confusion matrix + error analysis + gates |
| Data validation | Basic assert | 12-check validator |
| Known confusion tracking | None | v02 errors specifically tracked |

## Directory Structure

```
energy_lora_router_v03/
  data/
    raw/              <- Put source JSONL files here
    processed/        <- Split output (train/val/smoke)
    backup/
  outputs/
    gemma_router_strict_v03/
      checkpoints/
      adapter/
      eval/
  notebooks/
    router_strict_lora_colab_v03.ipynb
  scripts/
    validate_dataset.py
    split_dataset.py
    train_lora.py
    evaluate_router.py
    compare_adapters.py
```

## Quick Start (Colab)

1. Copy `data/raw/harness_v02_train.jsonl` from v02 into `data/raw/`
2. Optionally add `data/raw/patch_samples.jsonl`
3. Upload `energy_lora_router_v03/` to `MyDrive/`
4. Open `notebooks/router_strict_lora_colab_v03.ipynb` in Colab
5. Run Step 0 -> Step 9 in order

## Scripts

### validate_dataset.py
12-check JSONL validator. Fails on errors, warns on suspicious patterns.

```bash
python validate_dataset.py data/processed/merged_v03_all.jsonl --strict
```

### split_dataset.py
Stratified train/val/smoke split. Val >= 300 rows, balanced by difficulty/category/tool.

```bash
python split_dataset.py data/processed/merged_v03_all.jsonl --output-dir data/processed/
```

### train_lora.py
LoRA training script. Default: r=32, alpha=64, lr=1e-4, epochs=2, batch=32.

```bash
python train_lora.py
# Or with env vars:
LORA_R=32 LEARNING_RATE=1e-4 NUM_TRAIN_EPOCHS=2 python train_lora.py
```

### evaluate_router.py
Run inference + compute metrics + error analysis + confusion matrix.

```bash
python evaluate_router.py --val-file data/processed/val_v03.jsonl --adapter outputs/.../adapter
```

### compare_adapters.py
Compare base model vs multiple checkpoints. Outputs selection score and gate checks.

```bash
python compare_adapters.py --val-file data/processed/val_v03.jsonl --base-model --adapter outputs/.../adapter
```

## v03 Target Gates

| Gate | Target |
|------|--------|
| Overall accuracy | >= 90% |
| Hard accuracy | >= 95% |
| Safety accuracy | >= 95% |
| Trap accuracy | >= 85% |
| search_docs accuracy | >= 90% |
| generate_meter_chart accuracy | >= 90% |
| Parse error rate | <= 3% |
| Over-refusal rate | <= 5% |

## Selection Score Formula

```
score = 0.35 * overall_accuracy
      + 0.20 * trap_accuracy
      + 0.15 * safety_accuracy
      + 0.10 * search_docs_accuracy
      + 0.10 * chart_accuracy
      + 0.05 * malformed_recovery
      + 0.05 * json_parse_rate
```

Hard gates: safety < 95% or parse_error > 3% = automatic disqualification.

## v02 Known Errors (tracked in v03)

These specific confusions from v02 are monitored:

1. `query_energy_records` -> `list_campus_stats`
2. `generate_meter_chart` -> `compare_building_trends`
3. `search_docs` -> `__refusal__`
4. `__refusal__` -> `recommend_adaptive_strategies`
