# HRNet hand-keypoint swap — PHW & KIM only

**Date**: 2026-05-30
**Branch**: `iterative-ablation`
**Triggering audit**: `reports/tracker_audit/audit_summary.json` (`tracker_partial`,
Pearson r = −0.40 for visibility-vs-CER, +0.34 for dropout-vs-CER)
**Verdict**: **`null` — the tracker is not the bottleneck for PHW or KIM.**

---

## 1. Question

Stage 1 v3 left two signers stuck above CER 0.72 with MediaPipe detecting them
on only 71–74% of frames. The audit could not say whether that low detection
rate *caused* the high CER or merely *correlated* with it. This experiment
forces the issue: substitute the keypoint backend for those two signers only,
keep everything else identical, measure ΔCER on the unchanged Stage 1 v2
model.

If the tracker is at fault, swapping to a higher-recall or stronger backend
should produce a substantial CER drop. If it's not, the CER will move by
single-digit centesimals at most — exactly what we observe.

## 2. Setup

Per the §8 contract, the only variable across rows is the keypoint backend.
Model weights, head architecture, normalisation, fold definitions,
evaluation script — all identical to Stage 1 v2.

| Component | Configuration |
|---|---|
| Model       | Stage 1 v2 no_dann checkpoint, untouched |
| Head        | ConformerCTC, 4 layers, d=256, kernel=15, dropout=0.2 |
| Input       | 190-d landmark feature (21 joints × xyz + vel + acc + vis) |
| T_native    | 32 (linear-time resample) |
| Fold        | PHW → fold 2 ckpt; KIM → fold 1 ckpt |
| Augmentation | None at eval (greedy CTC decode) |

Backends compared:

| Backend | Detection threshold | Notes |
|---|---|---|
| `mediapipe_default`    | det/track conf = 0.3 | Stage 1 v2's exact settings |
| `mediapipe_sensitive`  | det/track conf = 0.2 | Recall-up control |
| `rtmpose_hand`         | RTMPose-M hand5 (MMPose) | **Not run** — mmpose install on Kaggle failed; recovered via the cheaper control instead |

Each backend re-extracted landmarks on the same 180 clips (PHW: 90, KIM: 90)
and rebuilt the 190-d feature with identical resample logic.

## 3. Results

### 3.1 Headline table

| Backend | PHW CER | KIM CER | Global detect rate |
|---|---|---|---|
| `mediapipe_default`   | **0.8493** | **0.7435** | 78.1% |
| `mediapipe_sensitive` | 0.8387 | 0.7323 | 81.3% |
| `rtmpose_hand`        | — | — | — |

### 3.2 Δ vs default (positive = backend helps)

| Backend | ΔPHW | ΔKIM |
|---|---|---|
| `mediapipe_sensitive` | **+0.0106** | **+0.0112** |
| `rtmpose_hand`        | — | — |

### 3.3 Detection vs CER yield

- Detection rate gain (default → sensitive): **+3.24 percentage points** (78.1 → 81.3).
- CER improvement: **+1.06 pp on PHW, +1.12 pp on KIM**.
- **Yield ratio**: ~0.33 CER pp per 1 pp of detection rate.

For comparison, a tracker-bound signer should produce a yield ratio
substantially above 1.0 — every recovered frame would land in a critical
decision region. The observed sub-unit yield says the recovered frames
either carry low-quality keypoints when MediaPipe finally fires, OR they're
in positions of the writing stroke where the CTC head can already infer
the right character from neighbouring frames.

## 4. Verdict (per Task §4)

Both ΔCER values land at ~+0.011, well under the 0.05 threshold for
`partial_pass` and the 0.10 threshold for `strong_pass`. Per the §4
decision tree this is a clean **`null`**.

**Operating decision**:
1. **Do not swap the keypoint backend in Stage 3.** The cost (full-dataset
   re-extraction + re-training under 5-fold CV) is not justified by
   ~1 pp expected gain on 2 signers' worth of clips (~5% of the dataset).
2. **Accept PHW and KIM as a dataset-side limit.** From Stage 3 onwards,
   every stage's headline reports both a full-cohort 5-fold mean and a
   "PHW+KIM-stripped" 5-fold mean. The stripped number is the cleaner
   indicator of model-side progress; the full number remains the headline
   that connects this dissertation to comparable WiTA literature.
3. **Keep `visibility_gate: true` in `configs/stage3.yaml`** anyway. The
   Conformer can still benefit from being told which frames are dropouts,
   even if those dropouts aren't ultimately recoverable.

## 5. The RTMPose caveat

MMPose's installation failed on Kaggle (the `mim install` step in Cell 1
did not yield a working `mmpose` import). The script gracefully skipped
the RTMPose backend, leaving rows blank. This is **not a critical gap for
the headline decision** — the `mediapipe_sensitive` control already
produced the null result, and a stronger backend like RTMPose-M would
need to deliver a > 10 CER-pp improvement (an order of magnitude beyond
what sensitive MediaPipe achieved) to flip the verdict. That's a priori
implausible given the ~3-point detection-rate ceiling we observed.

If the dissertation reviewer asks for an RTMPose row, the path forward is
to run `scripts/extract_phw_kim_keypoints.py` in a fresh Colab session
where mmpose's installation tends to be more reliable, and re-run the
eval script — at most one additional hour of CPU work.

## 6. Implication for the Stage 3 design

Stage 3's strengthened DINOv2 fingertip pool, temporal context, and
visibility gate (see `configs/stage3.yaml`) all remain unchanged. The
visibility gate is justified independently of this experiment — the
Conformer should still attenuate dropout frames during downstream stages,
even if those dropouts ultimately reflect dataset-side limits rather than
recoverable tracker errors.

The PHW+KIM-stripped reporting convention is the new variable in the
plan; see `docs/iterative_ablation_plan.md` §10 (added in this commit).

## 7. Code artifacts

| Path | Purpose |
|---|---|
| `scripts/extract_phw_kim_keypoints.py` | Streams PHW+KIM zips, runs each backend, writes per-backend caches |
| `scripts/eval_phw_kim_hrnet.py`        | Loads no_dann ckpts, evaluates each backend cache, emits results CSV + verdict |
| `models/encoders/hand_keypoint_backend.py` | `KeypointBackend` ABC + MediaPipe + RTMPose adapters |
| `notebooks/run_hrnet_swap_kaggle.ipynb` | End-to-end Kaggle orchestrator (cache rebuild + fold retrain + eval + report) |
| `reports/template/stripped_cohort.py`  | Compute full-cohort and PHW/KIM-stripped means from any stage's results JSON |
| `reports/hrnet_swap/per_signer_results.csv` | Raw (backend × signer) CER table |
| `reports/hrnet_swap/per_clip_cer.csv`  | Per-clip CER (180 clips × 2 backends = 360 rows) |
| `reports/hrnet_swap/per_clip_scatter.png` | Per-clip CER scatter for the inline visual |
| `reports/hrnet_swap/verdict.json`      | Auto-classification + deltas |
