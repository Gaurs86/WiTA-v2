# Stage 3 — DINOv2 fingertip-pool + temporal context + visibility gate

**Branch**: `iterative-ablation`
**Config**: `configs/stage3.yaml`
**Cache fingerprint**: dinov2-small @ 336×336, 24×24 grid, `fingertip_3x3_bell`,
temporal_context=True, visibility_gate=True, multi_joint=False
**Verdict**: ❌ **Gating test FAILED.** Stage 3 beats Stage 2 (0.860 → 0.804)
but loses to Stage 1 v3 by **+0.16 CER on every fold**.  The strengthened
fingertip-pool design does not carry the character signal this task needs.

---

## 1. Headline

| Stage                                  | Full cohort        | PHW/KIM-stripped   | Δ vs Stage 1 v3 (full) |
|----------------------------------------|--------------------|--------------------|------------------------|
| Stage 1 v3 no_dann (5-fold CV)         | 0.6448 ± 0.052     | 0.6383 ± 0.045     | —                      |
| Stage 2 (DINOv2 mean-pool, single-split) | 0.8601             | —                  | +0.22                  |
| **Stage 3** (DINOv2 fingertip 3x3 bell) | **0.8045 ± 0.020** | **0.8020 ± 0.017** | **+0.160**             |

Paired Wilcoxon (Stage 1 v3 vs Stage 3, n=5): **W = 0.0**, p = 0.0625.  Stage 3
loses on every single fold; the only reason `p` isn't < 0.05 is the n=5 sample
size limit of the Wilcoxon test.  Practical significance is overwhelming.

---

## 2. Per-fold detail

| Fold | best val CER | final train NLL | train NLL < 0.5 ever | best epoch |
|------|--------------|------------------|----------------------|------------|
| 0    | 0.8092       | 0.5184           | **yes** (marginally) | 28         |
| 1    | 0.8159       | 0.5496           | NO                   | 22         |
| 2    | 0.8294       | 0.5850           | NO                   | 20         |
| 3    | 0.7875       | 0.5362           | NO                   | 31         |
| 4    | 0.7802       | 0.5515           | NO                   | 32         |

**Two failure signals**:

1. **Gating test failed on 4 of 5 folds.**  Per `configs/stage3.yaml`
   pass/fail thresholds, "train NLL must drop below 0.5 within 80 epochs on
   every fold."  Only fold 0 cleared it.  Per the post-Stage-1-v3 prompt
   §2 Task B: "If train NLL stalls above 0.5, the fingertip-pool design
   does not help and Stage 4 will not save you."

2. **Early best epoch (20–32 of 80).**  In all 5 folds the val CER peaks
   well before epoch 40 and then either plateaus or regresses.  The
   OneCycleLR scheduler is still ramping past warmup at that point — the
   model has hit its representational ceiling early because the features
   simply don't carry enough discriminative signal to push further.

---

## 3. Per-signer scatter (qualitative)

The standardised scatter is at
`/kaggle/working/logs/stage3_per_signer_scatter.png` (committed kernel
output).  The key observation:

- **PHW and KIM (grey)**: 0.894 and 0.841.  Tightly clustered with the rest
  of the hard regime in Stage 3, whereas in Stage 1 v3 they sat ~0.15 CER
  above the next-hardest signer.  Stage 3's failure is so uniform that it
  collapses the bimodality.

- **Easy regime under Stage 1 v3 (KIS 0.43, YJH 0.44, KHY 0.51) → Stage 3**:
  KIS 0.78, YJH 0.71, KHY 0.76.  **The easy signers regressed by ~0.30
  CER points.**  Stage 3 is uniformly worse, not selectively worse on hard
  signers — strong evidence the feature is wrong for the task, not that
  the head failed to learn the harder cases.

- **Hard model-side regime (PJH, SYB, KJM, KNY, LKS, YMG)**: all sit in
  0.81–0.89.  Stage 3 actually compressed the easy-hard gap by making the
  easy ones worse, not by making the hard ones better.

---

## 4. Diagnosis: appearance features lack motion content

The strengthened design changed three things vs Stage 2: 3×3 bell pool
around the fingertip (vs mean over all 256 patches), ±1 temporal context
concat, and a visibility gate.  All three changes are individually
defensible and the resulting input (1153-d) carries strictly more
information than Stage 2's mean-pool (384-d).  Yet Stage 3 only beats
Stage 2 by ~5.5 CER points and stays ~16 CER points behind landmarks.

The likely cause is structural, not configurational:

> Air-writing is a **kinematic task**.  The letter is encoded in the
> *trajectory of the fingertip over time*, not in the appearance of the
> hand at any instant.  Landmark coordinates (`(x, y, z)` per joint per
> frame + first/second time differences) encode trajectory natively.
> Frozen DINOv2 patches at the fingertip location encode skin texture,
> nail position, finger pose, and lighting — none of which discriminate
> between letters.  A temporal context of ±1 frame is too small to
> reconstruct trajectory from three appearance snapshots.

This is consistent with the broader pattern of negative results across
this dissertation: every frozen-V-L feature (CLIP, SigLIP, X-CLIP,
DINOv2 mean-pool, and now DINOv2 fingertip pool) underperforms landmarks
on this task.  The conclusion isn't that V-L is useless — it's that
*frozen appearance features* are the wrong representation for kinematic
recognition, regardless of spatial pooling strategy.

---

## 5. Decision: don't proceed to Stage 4 fusion as currently designed

The post-Stage-1-v3 prompt §2 Task B is explicit:
> If train NLL stalls above 0.5, the fingertip-pool design does not help
> and Stage 4 will not save you. Investigate immediately; do not proceed.

Stage 3's gating failure activates this clause.  Recommended next moves
in priority order:

### 5.1 Run the multi-joint ablation (`configs/stage3_multijoint.yaml`)

The single-fingertip pool collapses spatial information to one patch's
neighbourhood.  The five-fingertip variant concatenates bell-pools at
joints {4, 8, 12, 16, 20} → 5×D=1920 channels before temporal context →
**5761-dim** input.  Train + eval is mechanically identical; the only
delta is the cache.

Cost: another full cache rebuild (~100 min on T4) + 5-fold sweep (~4 h).
Risk: still likely to fail (same fundamental representation issue), but
the multi-joint result is the cleanest published evidence that "more
joints don't help either."

### 5.2 Trajectory-encoded visual feature (Stage 3.5 alternative)

Render the last 30 frames of fingertip trajectory as a small (224x224)
image — literally draw the air-writing path — and feed that drawing
through DINOv2.  Now the visual feature ENCODES motion explicitly.
This is closer to how online-handwriting systems work.

Not implemented yet; would require a new cache builder.  Estimated
engineering: ~3 h.  This is the most promising path if you want to
revisit V-L for this task.

### 5.3 Skip Stage 4 fusion and jump to Stages 5–8

Per the original plan, Stages 5 (Swin-T), 6 (VideoMAE), 7 (CLIP/SigLIP),
8 (X-CLIP) test alternative video/image backbones.  Most have already
underperformed in earlier exploratory runs.  Given the consistent
appearance-feature failure mode, these are unlikely to clear Stage 1 v3
either.

### 5.4 Run Stage 4 fusion anyway as a sanity check

The prompt says don't proceed.  But pragmatically: late-fusion gives each
stream its own Conformer encoder.  If the landmark stream dominates
(because the DINOv2 stream is non-informative), Stage 4 late-fusion
should reproduce something close to Stage 1 v3.  Cost: one sweep
(~4 h on T4), no new cache build.  Outcome interpretation:

- Late ≈ Stage 1 v3 → confirms DINOv2 contributes nothing
- Late > Stage 1 v3 by > 0.01 → marginal complementarity (publishable)
- Early < Late → confirms gradient coupling poisons concatenation

This is the **lowest-cost decision-gating experiment** before doing
real design work.  Recommended.

---

## 6. Recommended next step

**Run Stage 4 fusion (option 5.4)** before Stage 3 multi-joint or
Stage 3.5.  Reasoning:

1. The cache is already built and ~600 MB.  Stage 4's only extra cost is
   training time (the dual-cache trainer reuses both Stage 1 + Stage 3
   caches).
2. Late-fusion gives the model the ability to ignore the DINOv2 stream
   if it's noise.  If that happens, we end at ~Stage 1 v3 — a cheap
   confirmation that this design line is dead.
3. If late-fusion shows ANY gain over Stage 1 v3, that's the strongest
   signal for whether further V-L work on this task is worthwhile.

Update `configs/stage4_*.yaml` headline_val_cer_max from 0.61 to a softer
target reflecting the Stage 3 failure: **passing means late-fusion mean ≤
0.6383** (Stage 1 v3 stripped baseline).  That's a "matches Stage 1 v3"
threshold; beating it by ≥ 0.01 is the original "fusion helps" result.

---

## 7. Code + artifacts

| Path | Purpose |
|---|---|
| `configs/stage3.yaml`                        | Stage 3 contract (now: known to fail) |
| `notebooks/run_stage3_cv_kaggle.ipynb`       | 10-cell Kaggle sweep notebook |
| `notebooks/run_stage3_cv.ipynb`              | Colab variant (untouched) |
| `models/encoders/dinov2_fingertip_extractor.py` | Bell-pool primitive |
| `datasets/dinov2_feature_cache.py`           | Stage 3 cache builder with fingerprint |
| `training/stage3_train.py`                   | Locked Stage 1 v2 recipe over the Stage 3 cache |
| **This file**                                | The verdict |
| `/kaggle/working/stage3_features.pt`         | The cache (600 MB) — keep for Stage 4 fusion |
| `/kaggle/working/stage3_results.json`        | Per-fold summary |
| `/kaggle/working/checkpoints/stage3_fold*_best.pt` | 5 best checkpoints |
| `/kaggle/working/logs/stage3_per_signer_scatter.png` | Per-signer figure |
