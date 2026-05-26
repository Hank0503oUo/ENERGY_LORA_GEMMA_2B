# ENERGY_LORA_GEMMA_2B

Gemma LoRA router training workspace for building-energy dispatch experiments.

Last updated: 2026-05-26

## Review Links

- GitHub source repository: https://github.com/Hank0503oUo/ENERGY_LORA_GEMMA_2B
- Full model artifacts and generated outputs: https://drive.google.com/drive/folders/1Bzyg8zyi_SIFMgAn6Cpg4JJqp5O2gHcj?usp=sharing

Use GitHub for source code, configuration, prompts, experiment history, and reproducible scripts. Use the Google Drive folder for LoRA adapters, checkpoints, model weights, generated datasets, notebooks with outputs, and evaluation artifacts.

## Project Contents

- `energy_lora_router_v02` through `energy_lora_router_v09`: versioned training, evaluation, and data-preparation workspaces.
- `energy_lora_router_v09`: latest routing repair round and the primary entry point.
- `scripts/`: data validation, dataset splitting, LoRA training, evaluation, and adapter comparison utilities inside each version folder.
- Colab notebooks and generated data folders are intentionally kept out of Git; use the Drive artifact folder for those files after access review.

## Data Safety Boundary

This repository is intended to contain source code, prompts, configuration, reproducible scripts, and experiment notes only. Generated JSONL datasets, notebook outputs, model artifacts, and raw or derived NTU meter-reading tables should not be committed to GitHub.

Current audit status:

- No tracked GitHub files should match the high-risk NTU meter-data patterns `powerMeter`, `NTU_powerMeter`, `metadata_uid`, `metadata_loop`, `meter_building`, `meter_id`, `ntu_energy`, or `energy.geojson`.
- Generated LoRA datasets are kept outside Git because even router/dispatch supervision can accidentally include tool-returned values such as area, mean kW, EUI, or kWh. Review those artifacts in the permissioned Drive folder before sharing.
- The intended LoRA training target is router/dispatch supervision. It should teach the model to output tool calls such as `query_energy_records` with building and year arguments; it should not train the model on real building-level kWh tables.
- Training prompts explicitly instruct that real numeric answers must come from tool results and must not be invented by the model.
- Any examples retained in Git should use placeholder entities such as `<BUILDING_A>` or clearly synthetic values. Real numeric answers belong in authorized runtime tools, not in the adapter training rows.
- The trained adapter should be treated as a dispatcher, not as the source of truth for electricity usage. Actual building electricity answers must come from the runtime database/tool layer with the proper access controls.

## Experiment Version History

| Version | Training direction | Training data generated for that round |
|---|---|---|
| v02 | First strict tool-router LoRA round. The model learned to map user requests to a fixed tool catalogue, including safety/refusal cases and hard negatives. | `harness_v02_train.jsonl`, `harness_v02_val.jsonl`, `harness_v02_smoke.jsonl`, `harness_v02_train_relabeled.jsonl`, plus synthetic supplements under `data/synth/`. Manifest: 2,518 total, 2,476 train, 38 val, 4 smoke. |
| v03 | Stabilization round after v02: larger stratified validation, stronger dataset validation, confusion tracking, checkpoint comparison, and gate-based adapter selection. | `merged_v03_all.jsonl`, `train_v03.jsonl`, `val_v03.jsonl`, `smoke_v03.jsonl`, plus `patch_samples.jsonl` and `safety_augment_v03.jsonl`. Manifest: 3,085 total, 2,635 train, 434 val, 16 smoke. |
| v04 | Router-strict augmentation round. Expanded training coverage with audited/cleaned routing samples and DeepSeek-generated augmentation to reduce tool confusion. | `merged_v04_all.jsonl`, `train_v04.jsonl`, `train_v04_audited.jsonl`, `train_v04_cleaned.jsonl`, `val_v04.jsonl`, `smoke_v04.jsonl`, plus `v04_augment_deepseek*.jsonl`. Manifest: 3,943 total, 3,400 train, 527 val, 16 smoke. |
| v05 | Architecture shift from `expected_tool` routing to agent dispatch training. The target became structured JSON with `dispatch_type`, `workflow_id`, `answerability`, locked entities, required tools, and stop conditions. | Raw dispatch data: `v05_dispatch_gen1000.jsonl` and `v05_dispatch_seed30.jsonl`. Chat-format training data: `train_v05_dispatch.jsonl`, `val_v05_dispatch.jsonl`, `smoke_v05_dispatch.jsonl`, `v05_chat_merged.jsonl`, `train_v05_chat_seed30.jsonl`. Manifest: 1,000 total, 837 train, 147 val, 16 smoke. |
| v06 | Dispatch contract recovery round. The main issue was high parse error from prompt/data mismatch, so v06 introduced strict dispatch-only prompting, formal DCI tools, boundary cleanup, and a format curriculum. | `train_v06_dispatch.jsonl`, `val_v06_dispatch.jsonl`, `smoke_v06_dispatch.jsonl`, `format_smoke_v06.jsonl`, plus `v06_format_curriculum.jsonl`. Manifest: 1,120 total, 957 train, 147 val, 16 smoke, 24 format smoke. |
| v07 | Schema clarity round. The model had started producing JSON but confused tool names, dispatch types, and workflow IDs, so v07 separated those three layers and added contrast examples. | `train_v07_dispatch.jsonl`, `val_v07_dispatch.jsonl`, `smoke_v07_dispatch.jsonl`, `format_smoke_v07.jsonl`, plus `v07_contrast_curriculum.jsonl` and JSON grammar files. Manifest: 1,194 total, 1,031 train, 147 val, 16 smoke, 16 format smoke. |
| v08 | Boundary repair round. Focused on `clarify_needed` over-triggering, `no_evidence` vs `clarify_needed`, and `workflow_chain` vs `single_tool` decision rules. | `train_v08_dispatch.jsonl`, `val_v08_dispatch.jsonl`, `smoke_v08_dispatch.jsonl`, `format_smoke_v08.jsonl`, plus `v08_boundary_curriculum.jsonl`. Manifest: 1,272 total, 1,109 train, 147 val, 16 smoke, 16 format smoke. |
| v09 | Current repair round. Puts `no_evidence` before actionable routing, removes placeholder workflow wording, adds year-range pre-gate examples, and raises evaluation generation length to avoid truncated JSON. | `train_v09_dispatch.jsonl`, `val_v09_dispatch.jsonl`, `smoke_v09_dispatch.jsonl`, `format_smoke_v09.jsonl`, plus `v09_boundary_curriculum.jsonl`. Manifest: 1,336 total, 1,173 train, 147 val, 16 smoke, 16 format smoke. |

## Current Experiment Progress

The current main track is the v09 dispatch-router repair round. V09 is built on the v08 workspace and focuses on repairing structured dispatch behavior rather than changing the full research scope.

V09 data status:

- Dataset profile: `agent_dispatch_training`
- Total examples: 1,336
- Train split: 1,173
- Validation split: 147
- Smoke split: 16
- Format smoke split: 16

Main v09 changes:

- `no_evidence` is evaluated before actionable tool routing.
- Placeholder wording such as `workflow_id=任務模式` was removed from the prompt.
- Out-of-range year cases for building-energy queries were added as deterministic pre-gate examples.
- Contrast examples were added for `no_evidence`, `clarify_needed`, `single_tool`, and `workflow_chain`.
- Evaluation default `MAX_NEW_TOKENS` was raised from `160` to `512` to reduce truncated JSON outputs.

Current evaluated baseline:

The latest local evaluation output available in this workspace is the v08 adapter copied under the v09 folder (`outputs/gemma_dispatch_v08`). V09 data has been prepared, but a final trained `gemma_dispatch_v09` adapter is not yet present in the local outputs.

V08 validation snapshot:

- Validation accuracy: 27.9% (`41 / 147`)
- Parse/malformed rate: 40.8%
- Smoke accuracy: 18.8% (`3 / 16`)
- Smoke parse/malformed rate: 50.0%
- Refusal cases: strong on explicit refusal examples (`20 / 20` in validation)
- Single-tool routing: partially working (`15 / 31` in validation)
- `no_evidence`: failing in current baseline (`0 / 25` in validation)
- `workflow_chain`: failing in current baseline (`0 / 56` in validation)
- Unsafe allow count: 0
- Over-refusal rate: 3.4%, which is under the 5% gate

Interpretation:

The model has learned part of the refusal/safety behavior and some single-tool routing, but it is still unstable as a structured dispatch router. The biggest blockers are JSON parse reliability, `no_evidence` recognition, and multi-step `workflow_chain` routing.

## Model Artifacts

Training outputs, checkpoints, adapters, and model weight files are intentionally excluded from Git by `.gitignore`.
Generated JSONL datasets and notebooks with cell outputs are also excluded from Git because they can contain tool-returned values.
The local project is about 6.4 GB because it contains generated LoRA artifacts, including files larger than GitHub's regular 100 MB single-file limit.

Full model artifacts are available via Google Drive:

- [ENERGY_LORA_GEMMA_2B artifact folder](https://drive.google.com/drive/folders/1Bzyg8zyi_SIFMgAn6Cpg4JJqp5O2gHcj?usp=sharing)

## Expected Next Adjustment Direction

The next experiment should train and evaluate the v09 adapter using the prepared v09 dataset.

Primary gates for the next run:

- Parse error rate below 10%
- Format smoke accuracy above 90%
- `no_evidence` accuracy above 80%
- Then tune `workflow_chain` vs `single_tool`

Recommended adjustment plan:

1. Train `gemma_dispatch_v09` from the v09 dataset.
2. Evaluate with `MAX_NEW_TOKENS=512` and compare against the v08 baseline above.
3. Add stricter JSON-format supervision or constrained decoding if malformed output remains high.
4. Expand contrast data for `workflow_chain` vs `single_tool`, especially cases where one user request requires multiple tools.
5. Expand `no_evidence` and out-of-range year examples so the router learns to reject unsupported evidence instead of forcing a tool call.
6. Select the best checkpoint using parse rate, format smoke, `no_evidence`, and workflow accuracy, not only overall accuracy.
7. Upload the selected adapter, checkpoint, and evaluation report to the Google Drive artifact folder.

## V9 Quick Start

```bash
cd energy_lora_router_v09
python scripts/run_v09_pipeline.py
```

Train on Colab or a GPU runtime:

```bash
export DRIVE_PROJECT_DIR=/content/drive/MyDrive/energy_lora_router_v09
python scripts/train_lora.py
```

Evaluate:

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

Before running the training or evaluation commands, restore the reviewed dataset and notebook artifacts from the permissioned Drive folder. See the version-specific README files for details about each training round.
