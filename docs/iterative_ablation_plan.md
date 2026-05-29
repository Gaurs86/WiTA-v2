# WiTA v2 — Iterative Ablation Plan (live document)

This is the canonical in-repo version of the iterative-ablation plan.  It
captures the *current* state of the project after each completed stage so
new collaborators (and future-me) can read one file instead of stitching
together prompts and notebooks.  Update it whenever a stage's verdict
lands.

Last updated: **after Stage 1 v3** (DANN falsified, 5-fold CV protocol locked).

---

## 0. Operating principles (non-negotiable)

1. **Single-variable-per-step contract.**  Between any two consecutive
   stages, exactly one input or one architectural choice changes; every
   other knob (optimiser, scheduler, seed, augmentation, Conformer config,
   ConvTranspose1d upsample, T_native, batch size, num_epochs) is locked
   from the previous stage.  This makes &Delta;-CER attribution clean.
2. **5-fold subject-disjoint CV mandatory.**  Every stage reports
   `mean ± std` over five folds of `manifests/subject_cv5.json`
   (built deterministically from `build_cv5_manifest(samples, n_folds=5,
   seed=42)`).  Single-split numbers are exploratory only and labelled as
   such.
3. **Per-signer scatter on every stage.**  Hard-regime tail
   (PHW, PJH, SYB, KJM, KNY, LKS, KIM, YMG) is the visual sanity check on
   whether the change addresses the tail or just lifts the easy regime.

---

## 1. Stage status

| Stage | Description                                            | Status      | 5-fold mean ± std | Notes |
|-------|--------------------------------------------------------|-------------|--------------------|-------|
| 1 v1  | Landmarks, no augment, 200 epochs                      | superseded  | —                  | single-split 0.7060; collapsed train NLL |
| 1 v2  | + augmentation (temporal warp/jitter/crop) + dropout/wd | superseded  | —                  | single-split 0.6870 |
| **1 v3** | Stage 1 v2 + DANN signer-adversarial (3 alpha × 5 folds) | **complete** | **0.6448 ± 0.052** (no_dann) | DANN falsified |
| 2     | DINOv2-S mean-pool over all 256 patches                | complete    | *(single-split)* 0.8601 | mean-pool destroys spatial focus |
| 3     | DINOv2-S **fingertip 3x3 bell pool** + ±1 temporal context + visibility gate | **in progress** | target 0.70–0.78 | see `configs/stage3.yaml` |
| 3 mj  | Stage 3 ablation: 5-fingertip pool                     | pending     | —                  | `configs/stage3_multijoint.yaml` |
| 4     | Fusion: landmark stream + DINOv2 fingertip stream      | pending     | target 0.46–0.58 | early & late variants |
| 5     | Swin-T + landmarks                                     | pending     | prior 0.58–0.65 |   |
| 6     | VideoMAE + landmarks                                   | pending     | prior 0.55–0.63 |   |
| 7     | CLIP/SigLIP + landmarks                                | pending     | prior 0.57–0.64 |   |
| 8     | X-CLIP + landmarks                                     | pending     | prior 0.58–0.66 |   |
| 9     | Joint CTC + attention + KenLM on row-4 winner          | pending     | prior 0.36–0.48 |   |
| 10    | Progressive unfreeze of winner                         | pending     | prior 0.32–0.44 |   |

---

## 2. Decisions baked in by completed stages

### From Stage 1 v3 (DANN sweep)
- **DANN is dead.**  Variant-mean delta = +0.020 (DANN hurts).  Per plan §5
  decision tree, do not bake DANN into any subsequent stage.
- **Single-split CER is statistically fragile** (std = 0.052 over 5 folds,
  > 0.04 threshold).  CV mandatory from here on.
- The Stage 1 baseline is **0.6448 ± 0.052** (5-fold no_dann), not 0.687.
  Update all priors.

### From Stage 2 (DINOv2 mean-pool)
- Frozen DINOv2 mean-pool is **insufficient** at CER 0.8601 (single-split).
  Mean-pool destroys spatial focus on the writing region.  Stage 3 fixes
  this by pooling the bell-weighted 3x3 patch window around the fingertip.

---

## 3. Hard-regime tail (per-signer floor)

Eight signers (PHW, PJH, SYB, KJM, KNY, LKS, KIM, YMG) sit at CER 0.72–0.90 and
are **near-invariant to model interventions** in Stage 1 v3 across the three
DANN variants (PHW: 0.898 / 0.853 / 0.851).  This pattern strongly suggests
the residual is a **feature / tracker-quality floor**, not a modelling deficit.

**Planned mitigations** (only if the Task A audit confirms tracker drift drives the tail):
1. Swap MediaPipe HandLandmarker for HRNet hand keypoints (offline pre-compute).
2. Keep the visibility-gate dim in Stage 3+ inputs so the Conformer can attenuate
   dropout frames rather than treat them as real content.
3. Inspect overlay GIFs for the worst signers; if trajectories look genuinely
   ambiguous (not tracker error), the residual is genuine handwriting hardness
   and is not addressable from the model side at this scale.

The audit driver lives in `scripts/audit_tracker_quality.py`.  Run with
`--mode cache` for a fast pass, or `--mode reextract` for the full audit with
overlay GIFs and native-resolution per-frame metrics.

---

## 4. Locked hyperparameters (Stage 1 v2 → present)

| Knob              | Value           |
|-------------------|-----------------|
| Conformer layers  | 4               |
| Conformer d_model | 256 (single-stream) / 2x128 (late-fusion) |
| Heads             | 4               |
| Conv kernel       | 15              |
| FFN mult          | 4               |
| Dropout           | 0.2             |
| Batch size        | 32              |
| Num epochs        | 80              |
| Optimiser         | AdamW (β=0.9, 0.999) |
| LR peak           | 5e-4            |
| Weight decay      | 5e-2            |
| Grad clip         | 1.0             |
| Scheduler         | OneCycleLR, cos anneal, 5% warmup |
| T_native          | 32              |
| Upsample          | 2 (T_out = 64)  |
| Seed              | 42              |

Augmentation (`LandmarkAugment` defaults):
- Temporal warp: p=0.80, max_warp=0.15
- Spatial jitter: p=0.80, σ=0.02 *(landmarks only; disabled for DINOv2 streams)*
- Temporal crop-resize: p=0.50, ratio ∈ [0.60, 1.00]

---

## 5. Pass/fail thresholds per stage

### Stage 3 (DINOv2 fingertip + temporal context + visibility gate)
- **Gating test**: every fold must reach `train NLL < 0.5` within 80 epochs.
  If any fold fails, the design is feature-insufficient — Stage 4 fusion
  will not save it.  Abort and audit MediaPipe + the patch pool.
- **Headline**: mean val CER ≤ **0.860** (beat Stage 2 mean).
- **Stretch**:  mean val CER ≤ **0.78** (Stage 4 fusion likely clears 0.55).

### Stage 4 (early- or late-fusion)
- **Headline**: mean val CER ≤ min(Stage 1 v3, Stage 3) − **0.03**.  With
  current measured baselines that's ≤ **0.61**.
- If neither fusion design clears the threshold, the two streams are
  mutually redundant; pick the cheaper.
- If late ≫ early, the streams are gradient-coupled badly under
  concatenation — interesting and worth reporting.

### Stages 5–8 (Swin-T, VideoMAE, CLIP/SigLIP, X-CLIP + landmarks)
- Each must beat **Stage 4 mean** to enter row-4-winner contention.
- Same per-signer scatter and Wilcoxon protocol; same 5 folds.

### Stage 9 (joint CTC + attention + KenLM)
- Run on the row-4 winner only.  Headline: mean val CER ≤ 0.48.

### Stage 10 (progressive backbone unfreeze)
- Final headroom test.  Headline: ≤ 0.44; stretch ≤ 0.36.

---

## 6. Reporting protocol (per Task D)

Every stage's report directory (`reports/stageN/`) must contain:
1. `stageN_report.md` — narrative + headline table.
2. `per_signer_cer.csv` (39 rows) + scatter PNG generated via
   `reports/template/per_signer_scatter.py`.
3. CV aggregate (mean ± std) printed via `reports/template/cv_summary.py`
   with a paired Wilcoxon against the previous-stage baseline.
4. The Stage-0 diagnostic suite snapshot
   (`length-bucketed CER`, `NLL gap`, `blank prob`, `KL(pred‖label)`,
   `edit decomposition`).
5. Train CTC loss curve overlaid against Stage 1 v2 and Stage 2 (same axis
   as the original Stage 1/2 report).

For amendments to historical reports (Stage 1 v1 → v3, Stage 2), use the
Section 8 generator: `python reports/build_section8_pdf.py`.  See
`reports/stage1_stage2_supplement.md`.

---

## 7. Stage ordering and dependencies

```
Task A (tracker audit)       — gates visibility-gate dim in Stage 3
       v
Stage 3 (DINOv2 fingertip)   — pass train NLL < 0.5 gates Stage 4
       v
Stage 4 (fusion)             — early & late
       v
Stages 5–8 (alternative backbones + landmarks)
       v
Stage 9 (CTC+attn+KenLM on row-4 winner)
       v
Stage 10 (progressive unfreeze of winner)
```

Tasks D (reporting) and E (this doc) run alongside whichever stage is
training.

---

## 8. Code locations cheat-sheet

| Concern                                    | File / module                                            |
|--------------------------------------------|----------------------------------------------------------|
| 5-fold CV manifest builder                 | `datasets/cv_splits.py`                                  |
| Subject-disjoint stream + IDs              | `datasets/subject_splits.py`                             |
| Landmark feature cache (Stage 1)           | `datasets/skeleton_cache.py`                             |
| DINOv2 mean-pool cache (Stage 2)           | `datasets/dinov2_cache.py`                               |
| DINOv2 2x2 fingertip cache (interim S3)    | `datasets/dinov2_fingertip_cache.py` (older variant)     |
| **DINOv2 3x3 bell-pool + temporal context** | `datasets/dinov2_feature_cache.py`                       |
| Stage 3 extractor primitives                | `models/encoders/dinov2_fingertip_extractor.py`          |
| Conformer + CTC                             | `models/conformer_ctc.py` (now with `input_layernorm`)    |
| DANN signer head (Stage 1 v3 only)          | `models/signer_adversary.py`, `models/grl.py`             |
| Late-fusion dual-stream                     | `models/dual_stream_head.py`                              |
| Stage 1 v3 trainer                          | `training/dann_train.py`                                  |
| Stage 3 trainer                             | `training/stage3_train.py`                                |
| Stage 4 fusion trainer                      | `training/stage4_train.py`                                |
| Tracker-quality audit (Task A)              | `scripts/audit_tracker_quality.py`                        |
| Per-signer scatter template                 | `reports/template/per_signer_scatter.py`                  |
| CV aggregate + Wilcoxon template            | `reports/template/cv_summary.py`                          |
| Section 8 PDF generator                     | `reports/build_section8_pdf.py`                           |

Notebooks (Colab + Kaggle):
- `notebooks/run_stage1v3_dann_cv.ipynb`        — Kaggle Stage 1 v3 sweep
- `notebooks/run_stage1v3_dann_cv_colab.ipynb`  — Colab Stage 1 v3 sweep
- `notebooks/run_stage3_cv.ipynb`               — Colab Stage 3 5-fold sweep
- `notebooks/run_stage4_cv.ipynb`               — Colab Stage 4 fusion (early + late)
