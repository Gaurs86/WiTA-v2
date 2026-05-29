# Supplement (Section 8) — Revised Stage 1 baseline under 5-fold subject CV

This supplement amends the Stage 1 / Stage 2 report (`stage1_stage2_report.pdf`)
after the Stage 1 v3 sweep completed.  Two findings invalidate the prior single-split numbers:

1. **The DANN hypothesis (H1) is falsified.**  Signer-adversarial training does not
   reduce val CER on this task; the variant-mean delta is +0.020 (DANN hurts by ~2 pts).
2. **The single-split fragility hypothesis (H2) is confirmed.**  The 5-fold std of the
   no-DANN baseline is **0.052**, well above the 0.04 threshold.  The single-split number
   from the original Stage 1 v2 report (0.687, one held-out 4-signer val fold) is therefore
   not statistically robust on its own and should not be cited without the CV mean.

## 8.1 Revised Stage 1 baseline (5-fold subject-disjoint CV)

The "Stage 1" entry in the headline comparison table is updated as follows.

| Stage                                  | Single-split (reported earlier) | **5-fold mean ± std (new)** | Best fold | Worst fold |
|----------------------------------------|---------------------------------|------------------------------|-----------|------------|
| Stage 1 v1 (no augment, 200 epochs)    | 0.7060                          | — *(not run under CV)*       | —         | —          |
| Stage 1 v2 (augment + reg, 80 epochs)  | 0.6870                          | **0.6448 ± 0.052**           | 0.578     | 0.699      |
| Stage 2 (DINOv2 mean-pool)             | 0.8601                          | *(not run under CV)*         | —         | —          |

**Per-fold best val CER for Stage 1 v3 no-DANN** (5 folds, identical recipe to Stage 1 v2):

| Fold | Best val CER | Val signers (8 / 7 each) |
|------|--------------|--------------------------|
| 0    | 0.631        | (see `manifests/subject_cv5.json`) |
| 1    | 0.695        | … |
| 2    | 0.699        | … |
| 3    | 0.621        | … |
| 4    | 0.578        | … |
| **Mean ± std** | **0.6448 ± 0.052** | — |

Going forward, every stage in this dissertation will report 5-fold subject-disjoint
CV mean ± std using `manifests/subject_cv5.json` (built deterministically from
`build_cv5_manifest(samples, n_folds=5, seed=42)`).  Single-split numbers are
exploratory only and will be flagged as such.

## 8.2 Per-signer val CER distribution (Stage 1 v3 no-DANN, n=39 signers)

Each of the 39 English-subset signers appears in exactly one CV fold's val set,
so the 39-entry per-signer CER vector is comparable across stages run on the
same manifest.

- **Range**: 0.431 → 0.898 (across-signer std: 0.096).
- **Easy regime** (CER ≤ 0.55, ~10 signers): KIS, YJH, KHY, HYW, KSH, JSA, YSY, KSJ.
- **Hard regime** (CER ≥ 0.75, ~8 signers): **PHW (0.898), PJH (0.842), SYB (0.809),
  KJM (0.746), KNY (0.737), LKS (0.730), KIM (0.723), YMG (0.728)**.

These hard-regime CERs are nearly variant-invariant
(PHW: 0.898 / 0.853 / 0.851 across no_dann / DANN α=1.0 / DANN α=0.3),
strongly suggesting a *feature-quality floor* (likely MediaPipe tracker dropouts
on a tail of harder signers) rather than a modelling deficit.  Task A (tracker
quality audit; see `scripts/audit_tracker_quality.py`) is the next diagnostic step.

## 8.3 Why DANN was dropped

The Stage 1 v3 prompt's decision tree treats DANN as confirmed only if the variant-mean
delta favours DANN by ≥ 0.05.  The measured deltas were:

| Variant       | Mean val CER | Δ vs no_dann |
|---------------|--------------|--------------|
| no_dann (ctrl) | 0.6448 ± 0.052 | —            |
| dann_a1        | 0.6660 ± 0.030 | **+0.021** (worse) |
| dann_a03       | 0.6688 ± 0.026 | **+0.024** (worse) |

Paired Wilcoxon (no_dann vs dann_a1, n=5): W₊=4, W₋=11 — trend favours the control.
Pattern is regression-to-mean — DANN hurts the easiest signers (KIS +0.146, YJH +0.131)
while marginally helping the hardest (PHW −0.045, SYB −0.049).  Net mean delta = +0.020.

DANN is removed from all downstream stages.

## 8.4 Implications for the reporting protocol

The original Stage 1/2 PDF figure showing single-split train/val NLL curves and per-epoch CER is
*still correct for the run it depicts* but its CER value should not be cited as
the canonical Stage 1 number.  Use 0.6448 ± 0.052 instead.  Future reports will:

1. Emit a 5-fold mean ± std and a paired Wilcoxon for every model change.
2. Emit the standardised per-signer scatter (via `reports/template/per_signer_scatter.py`)
   so cross-stage comparisons land on the same axis.
3. Cite single-split numbers only as exploratory or for one-off speed checks.
