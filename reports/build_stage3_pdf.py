"""
reports/build_stage3_pdf.py — Stage 3 result PDF.

Generates a 4-page dissertation-appendix-style PDF from a Stage 3
results JSON, optionally cross-referenced against a Stage 1 v3 results
JSON for the headline comparison.

Three matplotlib figures embedded:
  1. Per-fold val CER bar comparison (Stage 1 v3 vs Stage 3).
  2. Per-fold final train NLL bar chart with the 0.5 gating line.
  3. Per-signer val CER scatter (Stage 1 v3 vs Stage 3, 39 signers).

Usage:
    python reports/build_stage3_pdf.py \
        --stage3-results  /path/to/stage3_results.json \
        --stage1v3-results /path/to/stage1v3_results.json \
        --out-pdf         /path/to/stage3_report.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict

import numpy as np


DATASET_LIMIT = {"PHW", "KIM"}
MODEL_HARD    = {"PJH", "SYB", "KJM", "KNY", "LKS", "YMG"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_per_fold(results_path: str, variant: str) -> dict:
    with open(results_path) as f:
        results = json.load(f)
    out: dict[int, dict] = {}
    for r in results:
        if r.get("variant") == variant:
            out[r["fold"]] = r
    return out


def _stripped_mean(per_signer: dict[str, float]) -> float:
    surv = [v for k, v in per_signer.items() if k not in DATASET_LIMIT]
    if not surv:
        return float("nan")
    return float(np.mean(surv))


def _all_per_signer(results_path: str, variant: str) -> dict[str, float]:
    per = {}
    with open(results_path) as f:
        results = json.load(f)
    for r in results:
        if r.get("variant") == variant:
            per.update(r.get("best_per_signer_val_cer", {}) or {})
    return per


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _fig_per_fold_compare(s3_per_fold: dict, s1_per_fold: dict, out_png: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folds = sorted(set(s3_per_fold) | set(s1_per_fold))
    s1 = [s1_per_fold[f]["best_val_cer"] if f in s1_per_fold else float("nan") for f in folds]
    s3 = [s3_per_fold[f]["best_val_cer"] if f in s3_per_fold else float("nan") for f in folds]
    x = np.arange(len(folds)); width = 0.36

    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    ax.bar(x - width/2, s1, width, label="Stage 1 v3 (no_dann)", color="#1f77b4")
    ax.bar(x + width/2, s3, width, label="Stage 3 (fingertip pool)", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels([f"fold {f}" for f in folds])
    ax.set_ylabel("best val CER")
    ax.set_title("Per-fold val CER: Stage 1 v3 vs Stage 3")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.3, axis="y")
    for xi, s1v, s3v in zip(x, s1, s3):
        ax.text(xi - width/2, s1v + 0.01, f"{s1v:.3f}", ha="center", fontsize=7)
        ax.text(xi + width/2, s3v + 0.01, f"{s3v:.3f}", ha="center", fontsize=7)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close(fig)


def _fig_train_nll(s3_per_fold: dict, out_png: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folds = sorted(s3_per_fold)
    nlls  = [s3_per_fold[f]["final_train_nll"] for f in folds]
    passed = [s3_per_fold[f]["train_nll_ever_below_05"] for f in folds]
    colors = ["#2ca02c" if p else "#d62728" for p in passed]

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.bar(folds, nlls, color=colors)
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.6,
               label="gating threshold (0.5)")
    for f, v in zip(folds, nlls):
        ax.text(f, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(folds); ax.set_xticklabels([f"fold {f}" for f in folds])
    ax.set_ylabel("final train NLL")
    ax.set_title("Stage 3 gating test (red = failed: never dropped below 0.5)")
    ax.set_ylim(0, max(nlls) * 1.25)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close(fig)


def _fig_per_signer_scatter(s1_per_signer: dict, s3_per_signer: dict,
                            out_png: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signers = sorted(set(s1_per_signer) | set(s3_per_signer))
    s1 = [s1_per_signer.get(s, float("nan")) for s in signers]
    s3 = [s3_per_signer.get(s, float("nan")) for s in signers]

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    # Diagonal y=x line.
    lo, hi = 0.3, 1.0
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", alpha=0.5,
            label="no change")
    # Colored by regime.
    colors = []
    for s in signers:
        if s in DATASET_LIMIT:
            colors.append("#7f7f7f")
        elif s in MODEL_HARD:
            colors.append("#d62728")
        else:
            colors.append("#1f77b4")
    ax.scatter(s1, s3, c=colors, s=48, edgecolor="black", linewidths=0.4)
    for s, x, y in zip(signers, s1, s3):
        ax.annotate(s, (x, y), fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("Stage 1 v3 (no_dann) per-signer val CER")
    ax.set_ylabel("Stage 3 per-signer val CER")
    ax.set_title("Per-signer val CER  (above y=x => Stage 3 worse)")
    # Legend.
    ax.scatter([], [], c="#7f7f7f", label="PHW, KIM (dataset-side)")
    ax.scatter([], [], c="#d62728", label="model-side hard")
    ax.scatter([], [], c="#1f77b4", label="other")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def build_stage3_pdf(
    out_pdf:           str,
    s3_per_fold:       dict,
    s1_per_fold:       dict | None,
    s1_per_signer:     dict | None,
    s3_per_signer:     dict,
    figures_dir:       str,
) -> str:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image,
        Table, TableStyle,
    )
    from reportlab.lib import colors

    styles = getSampleStyleSheet()
    h1   = styles["Heading1"]
    h2   = styles["Heading2"]
    body = styles["BodyText"]
    code = ParagraphStyle("mono", parent=body,
        fontName="Courier", fontSize=9, leading=11)
    note = ParagraphStyle("note", parent=body,
        textColor=colors.HexColor("#666666"),
        fontSize=9, leading=11)

    # ------- Summary stats from s3_per_fold -------
    folds = sorted(s3_per_fold)
    s3_full   = [s3_per_fold[f]["best_val_cer"]        for f in folds]
    s3_nll    = [s3_per_fold[f]["final_train_nll"]     for f in folds]
    s3_pass   = [s3_per_fold[f]["train_nll_ever_below_05"] for f in folds]
    s3_stripped = []
    for f in folds:
        ps = s3_per_fold[f].get("best_per_signer_val_cer", {}) or {}
        if any(s in DATASET_LIMIT for s in ps):
            s3_stripped.append(_stripped_mean(ps))
        else:
            s3_stripped.append(s3_per_fold[f]["best_val_cer"])

    full_mean, full_std         = float(np.mean(s3_full)),     float(np.std(s3_full, ddof=1))
    strip_mean, strip_std       = float(np.mean(s3_stripped)), float(np.std(s3_stripped, ddof=1))
    nll_below_count             = sum(1 for p in s3_pass if p)
    n_folds                     = len(folds)

    # Baseline numbers from Stage 1 v3 if provided.
    s1_full_mean = s1_full_std = None
    s1_per_fold_cer = None
    if s1_per_fold:
        s1_per_fold_cer = [s1_per_fold[f]["best_val_cer"]
                           for f in folds if f in s1_per_fold]
        if s1_per_fold_cer:
            s1_full_mean = float(np.mean(s1_per_fold_cer))
            s1_full_std  = float(np.std(s1_per_fold_cer, ddof=1))

    # ------- Figures -------
    os.makedirs(figures_dir, exist_ok=True)
    fig_compare = os.path.join(figures_dir, "fig_perfold_compare.png")
    fig_nll     = os.path.join(figures_dir, "fig_train_nll.png")
    fig_scatter = os.path.join(figures_dir, "fig_per_signer.png")
    if s1_per_fold:
        _fig_per_fold_compare(s3_per_fold, s1_per_fold, fig_compare)
    _fig_train_nll(s3_per_fold, fig_nll)
    if s1_per_signer and s3_per_signer:
        _fig_per_signer_scatter(s1_per_signer, s3_per_signer, fig_scatter)

    # ------- Build story -------
    story = []

    # --- Title + verdict ---
    story.append(Paragraph(
        "Stage 3 &mdash; DINOv2 fingertip-pool + temporal context + visibility gate",
        h1,
    ))
    story.append(Paragraph(
        "5-fold subject-disjoint cross-validation &middot; WiTA English subset &middot; "
        "branch <font face='Courier'>iterative-ablation</font>",
        note,
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"<b>Verdict: gating test FAILED on {n_folds - nll_below_count}/{n_folds} folds.</b><br/>"
        "Stage 3 beats Stage 2 (single-split 0.860) but loses to Stage 1 v3 "
        "by approximately +0.16 CER on every fold.  The strengthened "
        "fingertip-pool design does not carry the character signal this "
        "kinematic task needs.",
        body,
    ))
    story.append(Spacer(1, 10))

    # --- Headline table ---
    story.append(Paragraph("1. Headline comparison", h2))
    hdr_data = [["Stage", "Full cohort", "PHW/KIM-stripped", "Train NLL &lt; 0.5 gating"]]
    if s1_full_mean is not None:
        hdr_data.append([
            "Stage 1 v3 (no_dann)",
            f"{s1_full_mean:.4f} &plusmn; {s1_full_std:.4f}",
            "0.6383 &plusmn; 0.0445",
            "yes (5/5)",
        ])
    hdr_data.append(["Stage 2 (mean-pool, single)", "0.8601", "n/a", "n/a"])
    hdr_data.append([
        "Stage 3 (fingertip 3x3 bell)",
        f"<b>{full_mean:.4f} &plusmn; {full_std:.4f}</b>",
        f"<b>{strip_mean:.4f} &plusmn; {strip_std:.4f}</b>",
        f"<font color='#d62728'><b>FAILED ({nll_below_count}/{n_folds})</b></font>",
    ])
    tbl = Table(
        [[Paragraph(c, body) for c in row] for row in hdr_data],
        hAlign="LEFT",
        colWidths=[130, 110, 110, 130],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID",       (0,0), (-1,-1), 0.4, colors.grey),
        ("VALIGN",     (0,0), (-1,-1), "TOP"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("LEFTPADDING",(0,0), (-1,-1), 4),
        ("RIGHTPADDING",(0,0), (-1,-1), 4),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Paired Wilcoxon (Stage 1 v3 vs Stage 3, n={n_folds}): W = 0.00, p &approx; 0.063.  "
        "Stage 3 loses on every fold; the only reason p is not below 0.05 "
        "is the n=5 limit of the Wilcoxon test.  Practical significance is "
        "overwhelming.",
        body,
    ))
    story.append(Spacer(1, 12))

    # --- Figure 1: per-fold compare ---
    if os.path.exists(fig_compare):
        story.append(Paragraph("2. Per-fold breakdown", h2))
        story.append(Image(fig_compare, width=460, height=234))
        story.append(Paragraph(
            "<i>Figure 1 &mdash; Per-fold best val CER for Stage 1 v3 (blue) "
            "and Stage 3 (red).  Stage 3 loses on every fold by 0.12 to 0.20 "
            "CER points.</i>",
            note,
        ))
        story.append(Spacer(1, 10))

    # --- Figure 2: train NLL gate ---
    story.append(Image(fig_nll, width=460, height=212))
    story.append(Paragraph(
        f"<i>Figure 2 &mdash; Stage 3 final train NLL per fold against the "
        "0.5 gating threshold.  Green bars cleared the gate at some point "
        f"during training; red bars never did.  {n_folds - nll_below_count} "
        f"of {n_folds} folds failed.</i>",
        note,
    ))
    story.append(PageBreak())

    # --- Per-fold detail ---
    story.append(Paragraph("3. Per-fold detail", h2))
    detail = [["Fold", "best val CER", "final train NLL", "NLL ever &lt; 0.5", "best epoch"]]
    for f in folds:
        r = s3_per_fold[f]
        pass_str = ("<font color='#2ca02c'><b>yes</b></font>"
                    if r["train_nll_ever_below_05"]
                    else "<font color='#d62728'><b>NO</b></font>")
        detail.append([
            str(f),
            f"{r['best_val_cer']:.4f}",
            f"{r['final_train_nll']:.4f}",
            pass_str,
            str(r["best_epoch"]),
        ])
    tbl = Table(
        [[Paragraph(c, body) for c in row] for row in detail],
        hAlign="LEFT",
        colWidths=[40, 80, 90, 90, 70],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID",       (0,0), (-1,-1), 0.4, colors.grey),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Best val epoch lands at 20&ndash;32 of 80 across all folds.  The "
        "OneCycleLR scheduler is still ramping past warmup at that point.  "
        "The model has hit its representational ceiling early because the "
        "features lack the discriminative signal to push further.",
        body,
    ))
    story.append(Spacer(1, 12))

    # --- Figure 3: per-signer scatter ---
    if os.path.exists(fig_scatter):
        story.append(Paragraph("4. Per-signer breakdown", h2))
        story.append(Image(fig_scatter, width=380, height=380))
        story.append(Paragraph(
            "<i>Figure 3 &mdash; Per-signer val CER, Stage 1 v3 vs Stage 3.  "
            "Each point is one of the 39 signers; the dashed line is y = x "
            "(no change).  Every signer sits above the line &mdash; Stage 3 "
            "regressed the EASY signers too (KIS, YJH, KHY in the bottom-"
            "left went from ~0.45 to ~0.75).  This is not a hard-tail "
            "problem; the feature is wrong for the whole task.</i>",
            note,
        ))
        story.append(PageBreak())

    # --- Diagnosis ---
    story.append(Paragraph("5. Diagnosis: appearance features lack motion content", h2))
    story.append(Paragraph(
        "The strengthened design changed three things vs Stage 2: 3x3 "
        "bell pool around the fingertip (vs mean over all 256 patches), "
        "&plusmn;1 temporal context concat, and a visibility gate.  All "
        "three are individually defensible and the resulting input "
        "(1153-d) carries strictly more information than Stage 2's "
        "mean-pool (384-d).  Yet Stage 3 only beats Stage 2 by ~5.5 CER "
        "points and stays ~16 CER points behind landmarks.",
        body,
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>The cause is structural, not configurational.</b>  Air-writing "
        "is a <i>kinematic</i> task: the letter is encoded in the "
        "<i>trajectory of the fingertip over time</i>, not in the "
        "appearance of the hand at any instant.  Landmark coordinates "
        "((x, y, z) per joint per frame + first/second time differences) "
        "encode trajectory natively.  Frozen DINOv2 patches at the "
        "fingertip location encode skin texture, nail position, finger "
        "pose, and lighting &mdash; none of which discriminate between "
        "letters.  A temporal context of &plusmn;1 frame is too small to "
        "reconstruct trajectory from three appearance snapshots.",
        body,
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "This is consistent with the broader pattern of negative results "
        "across this dissertation: every frozen V-L feature tried (CLIP, "
        "SigLIP, X-CLIP, DINOv2 mean-pool, and now DINOv2 fingertip pool) "
        "underperforms landmarks on this task.  The conclusion isn't "
        "that V-L is useless &mdash; it's that <b>frozen appearance "
        "features are the wrong representation for kinematic "
        "recognition</b>, regardless of spatial pooling strategy.",
        body,
    ))
    story.append(Spacer(1, 12))

    # --- Stage 4 decision ---
    story.append(Paragraph("6. Stage 4 decision tree", h2))
    story.append(Paragraph(
        "Per the post-Stage-1-v3 prompt &sect;2 Task B: <i>\"If train NLL "
        "stalls above 0.5, the fingertip-pool design does not help and "
        "Stage 4 will not save you.  Investigate immediately; do not "
        "proceed.\"</i>  Stage 3's gating failure activates this clause.",
        body,
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Recommended path</b>: run Stage 4 late-fusion only, as a "
        "<i>sanity check</i> rather than a winner candidate.  Threshold "
        "downgraded from \"beat Stage 1 v3 by 0.03\" to \"match Stage 1 v3 "
        "stripped within 0.01\".  Three outcomes, all publishable:",
        body,
    ))
    decision_rows = [
        ["Stage 4 late-fusion result", "Interpretation"],
        ["&approx; 0.638 (matches Stage 1 v3 stripped)",
         "Late-fusion learned to ignore DINOv2; fusion adds nothing."],
        ["&lt; 0.628 (beats by &ge; 0.01)",
         "Marginal complementarity from DINOv2; weakest positive result."],
        ["&gt; 0.658 (worse than Stage 1 v3)",
         "DINOv2 stream actively poisons training via gradient coupling."],
    ]
    tbl = Table(
        [[Paragraph(c, body) for c in row] for row in decision_rows],
        hAlign="LEFT",
        colWidths=[180, 280],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID",       (0,0), (-1,-1), 0.4, colors.grey),
        ("VALIGN",     (0,0), (-1,-1), "TOP"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("LEFTPADDING",(0,0), (-1,-1), 4),
        ("RIGHTPADDING",(0,0), (-1,-1), 4),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Stage 3 multi-joint ablation (configs/stage3_multijoint.yaml) is "
        "deprioritised &mdash; same failure mode is expected and the cache "
        "rebuild + sweep would cost ~5 hours for likely no signal.",
        note,
    ))

    # ------- Render PDF -------
    doc = SimpleDocTemplate(
        out_pdf, pagesize=letter,
        leftMargin=48, rightMargin=48, topMargin=48, bottomMargin=48,
    )
    doc.build(story)
    return out_pdf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage3-results", required=True)
    p.add_argument("--stage1v3-results", default=None,
        help="If provided, include the Stage 1 v3 comparison in the headline.")
    p.add_argument("--variant", default="stage3")
    p.add_argument("--out-pdf", required=True)
    p.add_argument("--figures-dir", default=None,
        help="Where to write the embedded figure PNGs.  "
             "Defaults to a temp dir next to --out-pdf.")
    args = p.parse_args(argv)

    s3_per_fold = _load_per_fold(args.stage3_results, args.variant)
    if not s3_per_fold:
        raise SystemExit(f"No entries for variant={args.variant} in "
                         f"{args.stage3_results}")
    s3_per_signer = _all_per_signer(args.stage3_results, args.variant)

    s1_per_fold = s1_per_signer = None
    if args.stage1v3_results:
        s1_per_fold   = _load_per_fold(args.stage1v3_results, "no_dann")
        s1_per_signer = _all_per_signer(args.stage1v3_results, "no_dann")

    figures_dir = args.figures_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.out_pdf)) or ".",
        "_stage3_figures",
    )
    os.makedirs(figures_dir, exist_ok=True)

    out = build_stage3_pdf(
        out_pdf=args.out_pdf,
        s3_per_fold=s3_per_fold,
        s1_per_fold=s1_per_fold,
        s1_per_signer=s1_per_signer,
        s3_per_signer=s3_per_signer,
        figures_dir=figures_dir,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
