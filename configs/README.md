# Configs

Every training config is a complete TOML file. Changing any value changes its hash, and a run
resumed under a changed config fails on purpose. Each experiment file states which base it
copies and marks the lines it changes.

| File | Role |
|---|---|
| `transformer.toml` | `small-v1`, the first small transformer on the CSV tables alone; the S/H/F comparison ran on it |
| `multimodal.toml` | `small-v1-mm`: the same plus genotype tokens and the brain-MRI token (D34); the round-1 incumbent |
| `v2_base.toml` | `small-v2-base`: `multimodal.toml` without the clock codes DALY, QALY, and QOLS (D36); the representation the submission uses |
| `experiments/mm_*.toml` | round 1 trials (`budget.toml`, D35), each `multimodal.toml` with the marked lines changed |
| `experiments/v2_*.toml` | round 2 trials (`budget_v2.toml`, D36), each `v2_base.toml` with the marked lines changed. `v2_c1.toml`, `v2_c1_seed2.toml`, and `v2_c1_seed3.toml` are the 15 submitted members (five folds each); `v2_ptw_pre.toml` is the pretraining of seed 1, and the two seed files run their own pretraining |
| `experiments/v3_l4w192.toml` | the round-3 architecture screen (`budget_v3.toml`, D38): 4 layers, width 192; it failed on fold 0 |
| `budget.toml`, `budget_v2.toml`, `budget_v3.toml` | the three tuning rounds: trial lists, adoption rules, and the evidence for each, committed before their transformer runs started (round 3's stacking and calibration rules were committed while the stacking job was already running; D38 says so) |
| `final.toml` | the frozen bundle: the members, the blend recipe, the Platt weights, the comparison files, and the val-run settings |
| `p3a.toml` | the preflight audit setting: reproduces one step-2 baseline (C = 0.003) on the saved folds |
| `smoke.toml`, `smoke_mm.toml` | tiny end-to-end runs for `make pipeline-smoke`; not results |
