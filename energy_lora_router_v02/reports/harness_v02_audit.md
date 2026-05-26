# harness_v02 SFT Data-Factory Audit

- generated_at: `2026-05-07T18:46:05.683311+00:00`
- assessment: `PASS_WITH_WARNINGS`

## Counts

- train: `297`
- val: `38`
- smoke: `4`
- sources: `{'router_sft.jsonl': 300, 'safety_sft.jsonl': 35}`

## Factory Alignment

- manifest: `True`
- deterministic_split: `True`
- smoke_split: `True`
- risk_refusal_samples: `True`
- rejected_variants: `False`
- preference_pairs: `False`
- downstream_validation_report: `False`

## Issues

- `warn` `duplicate_train_prompts`: 6 duplicate user prompts in train split.
- `warn` `real_entity_names_present`: Real campus/building names are present. This is OK for a closed demo, but not for a reusable public SFT set: NTU, 台大, 保健中心, 化學工程館, 土木研究大樓, 土木大樓
- `warn` `no_rejected_variants`: harness_v02 has no rejected variants/preference pairs; keep it as SFT router data, not DPO data.

## Distributions

```json
{
  "train_category": {
    "routing": 268,
    "safety": 29
  },
  "val_category": {
    "routing": 32,
    "safety": 6
  },
  "train_difficulty": {
    "hard": 72,
    "easy": 60,
    "trap": 52,
    "medium": 49,
    "malformed": 35,
    "safety": 29
  },
  "val_difficulty": {
    "medium": 8,
    "hard": 8,
    "trap": 8,
    "safety": 6,
    "malformed": 5,
    "easy": 3
  },
  "train_tool_top": {
    "__refusal__": 83,
    "query_energy_records": 47,
    "run_counterfactual_for_building": 22,
    "compare_building_trends": 13,
    "compare_energy_usage": 13,
    "detect_energy_anomalies": 11,
    "get_top_energy_buildings": 10,
    "search_docs": 10,
    "recommend_adaptive_strategies": 9,
    "rank_energy_buildings_across_years": 8,
    "optimize_energy_portfolio": 8,
    "run_openbse_hybrid_counterfactual": 8,
    "classify_anomaly": 6,
    "run_pvid": 6,
    "generate_meter_chart": 5
  },
  "val_tool_top": {
    "__refusal__": 10,
    "query_energy_records": 8,
    "run_counterfactual_for_building": 5,
    "search_docs": 3,
    "generate_meter_chart": 2,
    "compare_building_trends": 2,
    "compare_energy_usage": 1,
    "seasonal_strategies": 1,
    "run_openbse_hybrid_counterfactual": 1,
    "openbse_hvac_breakdown": 1,
    "correlate_algorithms": 1,
    "record_strategy": 1,
    "get_top_energy_buildings": 1,
    "optimize_energy_portfolio": 1
  }
}
```
