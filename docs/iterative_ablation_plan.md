# WiTA v2 — Iterative Ablation Plan (live document)

This is the canonical in-repo version of the iterative-ablation plan.  It
captures the *current* state of the project after each completed stage so
new collaborators (and future-me) can read one file instead of stitching
together prompts and notebooks.  Update it whenever a stage's verdict
lands.

Last updated: **after HRNet hand-keypoint swap experiment** (null verdict; PHW
and KIM accepted as dataset-side limits; dual-cohort reporting locked in).

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

| Stage | Description                                            | Status      | Full-cohort 5-fold | PHW/KIM-stripped | Notes |
|-------|--------------------------------------------------------|-------------|--------------------|------------------|-------|
| 1 v1  | Landmarks, no augment, 200 epochs                      | superseded  | —                  | —                | single-split 0.7060; collapsed train NLL |
| 1 v2  | + augmentation (temporal warp/jitter/crop) + dropout/wd | superseded  | —                  | —                | single-split 0.6870 |
| **1 v3** | Stage 1 v2 + DANN signer-adversarial (3 alpha × 5 folds) | **complete** | **0.6448 ± 0.052** (no_dann) | **0.6383 ± 0.0445** | DANN falsified |
| HRNet swap | PHW+KIM keypoint backend test (no retrain)        | **complete** | n/a                | n/a              | verdict `null`; PHW+KIM = dataset-side limit |
| 2     | DINOv2-S mean-pool over all 256 patches                | complete    | *(single-split)* 0.8601 | —          | mean-pool destroys spatial focus |
| 3     | DINOv2-S **fingertip 3x3 bell pool** + ±1 temporal context + visibility gate | **in progress** | target 0.70–0.78 | target 0.69–0.77 | see `configs/stage3.yaml` |
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

### From the tracker-quality audit (Task A)
- Verdict `tracker_partial` (Pearson r=-0.40 visibility / +0.34 dropout-length).
  Two signers — **PHW (vis 0.716, CER 0.898) and KIM (vis 0.738, CER 0.723)** —
  are the only ones below 90% MediaPipe visibility.  All six other hard-regime
  signers have ≥ 98% visibility, so their high CER is not tracker-driven.

### From the HRNet hand-keypoint swap experiment
- **Verdict `null`.**  Swapping MediaPipe for sensitive-MediaPipe on PHW+KIM
  clips (180 clips total) improved detection from 78.1% → 81.3% but cut
  CER by only +1.1 pp on each signer (PHW 0.8493 → 0.8387; KIM 0.7435 → 0.7323).
  Yield ratio ~0.33 CER pp per 1 pp of detection — well below the >1.0 you'd
  see if the tracker were genuinely the bottleneck.
- **Decision**: PHW and KIM are accepted as a **dataset-side limit** (motion blur,
  lighting, or hand off-frame on these two specific captures).  Do not swap
  the keypoint backend.  All future stage reports emit **dual-cohort numbers**:
  a full-cohort 5-fold mean AND a PHW/KIM-stripped 5-fold mean.  The
  stripped number isolates model-side progress from the dataset floor.
- RTMPose backend was prepared but not run (mmpose install failed on Kaggle).
  Re-running it would require a > 10 pp CER improvement on PHW+KIM to flip
  the verdict — an order of magnitude beyond what sensitive MediaPipe achieved.

---

## 3. Hard-regime tail (per-signer floor)

The Stage 1 v3 no_dann 5-fold no_dann sweep produced a bimodal per-signer
CER distribution with eight hard-regime signers above 0.72.  After the
HRNet-swap experiment they split into two distinct sub-groups:

| Sub-group | Signers | Behaviour |
|---|---|---|
| Dataset-side limit | PHW, KIM | Low MediaPipe visibility (<75%); tracker swap doesn't help (verdict null).  Floor likely sits around current CER. |
| Model-side hard regime | PJH, SYB, KJM, KNY, LKS, YMG | ≥ 98% visibility; high CER comes from genuine handwriting ambiguity or feature-deficit, not tracker.  Stage 3+ designs may still help. |

**Operating decision on the dataset-side group**:
- Report dual-cohort numbers from Stage 3 onwards (`reports/template/stripped_cohort.py`
  computes both means from any results JSON).
- Stage 1 v3 dual-cohort baseline (no_dann):
  - **Full cohort**: 0.6448 ± 0.052
  - **PHW/KIM-stripped**: **0.6383 ± 0.0445**  (Δ = +0.0065)
- The stripped baseline is what Stage 3+ must beat in addition to the full one.

The audit driver lives in `scripts/audit_tracker_quality.py`.  The HRNet-swap
experiment lives in `scripts/extract_phw_kim_keypoints.py` + `scripts/eval_phw_kim_hrnet.py`
+ `notebooks/run_hrnet_swap_kaggle.ipynb`.

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

## 6. Reporting protocol (per Task D, amended after HRNet swap)

Every stage's report directory (`reports/stageN/`) must contain:
1. `stageN_report.md` — narrative + headline table.
2. **Dual-cohort summary** (`reports/template/stripped_cohort.py`):
   - full-cohort 5-fold mean ± std
   - PHW/KIM-stripped 5-fold mean ± std
   - Δ between them
   Both numbers go in the headline table; the stripped one is the
   model-side progress indicator, the full one is the literature-comparable
   headline.
3. `per_signer_cer.csv` (39 rows) + scatter PNG generated via
   `reports/template/per_signer_scatter.py`.
4. CV aggregate (mean ± std) printed via `reports/template/cv_summary.py`
   with a paired Wilcoxon against the previous-stage baseline.
5. The Stage-0 diagnostic suite snapshot
   (`length-bucketed CER`, `NLL gap`, `blank prob`, `KL(pred‖label)`,
   `edit decomposition`).
6. Train CTC loss curve overlaid against Stage 1 v2 and Stage 2 (same axis
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
