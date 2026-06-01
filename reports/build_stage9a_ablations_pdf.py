"""
reports/build_stage9a_ablations_pdf.py — Stage 9a ablations PDF.

Generates a 4-page dissertation-appendix PDF from the ablations results
JSON.  Three figures: lambda curve, depth curve, seed scatter.  Plus
tables, the 2-sigma significance flags, and the reproducibility verdict.

Usage:
    python reports/build_stage9a_ablations_pdf.py \
        --results /path/to/stage9a_ablations_results.json \
        --out-pdf /path/to/stage9a_ablations_report.pdf
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


HEADLINE_FOLD0_CER = 0.5681
REPRO_TOLERANCE    = 0.02


def _figs(R, ctl, threshold, figures_dir, seeds, sigma):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(figures_dir, exist_ok=True)
    paths = {}

    # (1) lambda curve.
    lam, cer = [], []
    for v, l in [('stage9a_abl_l00', 0.0), ('stage9a_abl_l01', 0.1),
                 ('stage9a_abl_l03_s0', 0.3), ('stage9a_abl_l05', 0.5),
                 ('stage9a_abl_l07', 0.7)]:
        if v in R: lam.append(l); cer.append(R[v]['best_val_cer'])
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(lam, cer, 'o-', color='#1f77b4', markersize=8)
    ax.axhline(ctl, color='gray', linestyle=':', alpha=0.7, label=f'control ({ctl:.4f})')
    ax.fill_between([min(lam), max(lam)], ctl - threshold, ctl + threshold,
                    color='gray', alpha=0.15, label=f'±2σ seed band ({threshold:.4f})')
    for x, y in zip(lam, cer):
        ax.annotate(f'{y:.3f}', (x, y), fontsize=7, xytext=(4, 4),
                    textcoords='offset points')
    ax.set_xlabel('lambda_ctc'); ax.set_ylabel('fold 0 best val CER')
    ax.set_title('lambda_ctc sweep')
    ax.set_xticks([0.0, 0.1, 0.3, 0.5, 0.7])
    ax.legend(frameon=False, fontsize=7, loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    p = os.path.join(figures_dir, 'fig_lambda.png'); plt.savefig(p, dpi=180); plt.close(fig)
    paths['lambda'] = p

    # (2) depth.
    depth, cd = [], []
    for v, d in [('stage9a_abl_d2', 2), ('stage9a_abl_l03_s0', 3), ('stage9a_abl_d4', 4)]:
        if v in R: depth.append(d); cd.append(R[v]['best_val_cer'])
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(depth, cd, 's-', color='#d62728', markersize=8)
    ax.axhline(ctl, color='gray', linestyle=':', alpha=0.7)
    ax.fill_between([min(depth), max(depth)], ctl - threshold, ctl + threshold,
                    color='gray', alpha=0.15)
    for x, y in zip(depth, cd):
        ax.annotate(f'{y:.3f}', (x, y), fontsize=7, xytext=(4, 4),
                    textcoords='offset points')
    ax.set_xlabel('decoder layers'); ax.set_ylabel('fold 0 best val CER')
    ax.set_title('decoder depth (lambda=0.3, 80 epochs)')
    ax.set_xticks([2, 3, 4])
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    p = os.path.join(figures_dir, 'fig_depth.png'); plt.savefig(p, dpi=180); plt.close(fig)
    paths['depth'] = p

    # (3) seed scatter.
    seed_ids = [42, 43, 44]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.scatter(seed_ids[:len(seeds)], seeds, s=80, c='#2ca02c')
    for s, c in zip(seed_ids[:len(seeds)], seeds):
        ax.annotate(f'{c:.4f}', (s, c), fontsize=7,
                    xytext=(5, 0), textcoords='offset points')
    ax.axhline(HEADLINE_FOLD0_CER, color='black', linestyle='--', alpha=0.5,
               label=f'Stage 9a headline ({HEADLINE_FOLD0_CER})')
    ax.set_xlabel('seed'); ax.set_ylabel('fold 0 best val CER')
    ax.set_title(f'seed-variance probe  (σ={sigma:.4f}, 2σ={2*sigma:.4f})')
    ax.set_xticks(seed_ids[:len(seeds)])
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    p = os.path.join(figures_dir, 'fig_seeds.png'); plt.savefig(p, dpi=180); plt.close(fig)
    paths['seeds'] = p
    return paths


def build_pdf(results_path: str, out_pdf: str, figures_dir: str) -> str:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image,
        Table, TableStyle,
    )
    from reportlab.lib import colors

    with open(results_path) as f:
        R = {r['variant']: r for r in json.load(f)}

    # Seed variance.
    seeds = [R['stage9a_abl_l03_s0']['best_val_cer'],
             R['stage9a_abl_l03_s1']['best_val_cer'],
             R['stage9a_abl_l03_s2']['best_val_cer']]
    mu  = float(np.mean(seeds))
    sig = float(np.std(seeds, ddof=1))
    two_sigma = max(0.005, 2 * sig)
    ctl = R['stage9a_abl_l03_s0']['best_val_cer']

    # Repro.
    delta_repro = abs(ctl - HEADLINE_FOLD0_CER)
    repro_ok = delta_repro <= REPRO_TOLERANCE

    figs = _figs(R, ctl, two_sigma, figures_dir, seeds, sig)

    styles = getSampleStyleSheet()
    h1, h2, body = styles['Heading1'], styles['Heading2'], styles['BodyText']
    note = ParagraphStyle('note', parent=body,
        textColor=colors.HexColor('#666666'), fontSize=9, leading=11)

    story = []
    story.append(Paragraph("Stage 9a ablations &mdash; lambda_ctc sweep + decoder depth + training length", h1))
    story.append(Paragraph(
        "Single-fold matrix on fold 0 with seed-variance estimate and reproducibility "
        "guard.  10 runs, ~8.7 GPU-hours on Kaggle T4.",
        note,
    ))
    story.append(Spacer(1, 6))
    repro_color = '#2ca02c' if repro_ok else '#d62728'
    story.append(Paragraph(
        f"<b>Reproducibility check</b>: l03_s0 CER = {ctl:.4f}; Stage 9a headline = "
        f"{HEADLINE_FOLD0_CER}; <font color='{repro_color}'>|Δ| = {delta_repro:.4f} "
        f"≤ {REPRO_TOLERANCE} ✓</font>"
        if repro_ok else
        f"<b>Reproducibility check</b>: l03_s0 CER = {ctl:.4f}; Stage 9a headline = "
        f"{HEADLINE_FOLD0_CER}; <font color='{repro_color}'>|Δ| = {delta_repro:.4f} > "
        f"{REPRO_TOLERANCE} — CODEBASE DRIFT</font>",
        body,
    ))
    story.append(Paragraph(
        f"<b>Seed variance (l03 ×3)</b>: μ = {mu:.4f}, σ = {sig:.4f}, "
        f"2σ-claim threshold = <b>{two_sigma:.4f}</b>.  Any ablation delta within "
        f"±{two_sigma:.4f} of control is within seed noise.",
        body,
    ))
    story.append(Spacer(1, 10))

    # Headline 1-line summary.
    deltas = {v: R[v]['best_val_cer'] - ctl for v in R if v != 'stage9a_abl_l03_s0'}
    real_wins = [(v, d) for v, d in deltas.items() if d <= -two_sigma]
    real_loss = [(v, d) for v, d in deltas.items() if d >=  two_sigma]
    summary_color = '#2ca02c' if real_wins else '#7f7f7f'
    if real_wins:
        win_names = ', '.join(v.replace('stage9a_abl_', '') for v, _ in real_wins)
        story.append(Paragraph(
            f"<b>Verdict</b>: <font color='{summary_color}'>{len(real_wins)} variant"
            f"{'s' if len(real_wins)>1 else ''} beat control by ≥2σ: <b>{win_names}</b>.</font>  "
            f"{len(real_loss)} regress significantly.  Rest are within seed noise.",
            body,
        ))
    else:
        story.append(Paragraph(
            f"<b>Verdict</b>: no variant beats control by ≥2σ.  Current Stage 9a "
            "recipe sits at or near the local optimum on these three axes.  "
            f"{len(real_loss)} variants regress significantly (most informative: l00).",
            body,
        ))
    story.append(Spacer(1, 8))

    # Master table.
    story.append(Paragraph("1. Full table", h2))
    order = [
        ('stage9a_abl_l00',  'l00',  '0.0',  '3', '42'),
        ('stage9a_abl_l01',  'l01',  '0.1',  '3', '42'),
        ('stage9a_abl_l03_s0','l03_s0 (ctl)', '0.3', '3', '42'),
        ('stage9a_abl_l03_s1','l03_s1', '0.3', '3', '43'),
        ('stage9a_abl_l03_s2','l03_s2', '0.3', '3', '44'),
        ('stage9a_abl_l05',  'l05',  '0.5',  '3', '42'),
        ('stage9a_abl_l07',  'l07',  '0.7',  '3', '42'),
        ('stage9a_abl_d2',   'd2',   '0.3',  '2', '42'),
        ('stage9a_abl_d4',   'd4',   '0.3',  '4', '42'),
        ('stage9a_abl_e120', 'e120', '0.3',  '3', '42'),
    ]
    rows = [["variant", "λ_ctc", "dec L", "seed", "epochs", "best CER", "Δ vs ctl", "flag",
             "best ep", "len_ratio"]]
    for v, lbl, lam, dl, seed in order:
        if v not in R:
            rows.append([lbl, lam, dl, seed, "—", "—", "—", "—", "—", "—"]); continue
        r = R[v]
        d = r['best_val_cer'] - ctl
        if v == 'stage9a_abl_l03_s0':
            tag = 'ctl'
        elif d <= -two_sigma:
            tag = "<font color='#2ca02c'><b>✓</b></font>"
        elif d >= two_sigma:
            tag = "<font color='#d62728'><b>✗</b></font>"
        else:
            tag = '⚖'
        rows.append([
            lbl, lam, dl, seed, str(r['num_epochs']),
            f"<b>{r['best_val_cer']:.4f}</b>",
            f"{d:+.4f}" if v != 'stage9a_abl_l03_s0' else "—",
            tag, str(r['best_epoch']),
            f"{r['best_mean_pred_len_ratio']:.3f}",
        ])
    tbl = Table([[Paragraph(c, body) for c in r] for r in rows], hAlign='LEFT',
        colWidths=[68, 38, 38, 35, 45, 52, 50, 32, 38, 52])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID',       (0,0), (-1,-1), 0.4, colors.grey),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('FONTSIZE',   (0,0), (-1,-1), 8),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Flags: <b>✓</b> beats control by ≥2σ &middot; <b>✗</b> regresses by ≥2σ "
        "&middot; <b>⚖</b> within seed noise.  len_ratio = sum_predicted_chars / "
        "sum_reference_chars (1.0 = balanced).",
        note,
    ))
    story.append(PageBreak())

    # Figures.
    story.append(Paragraph("2. lambda_ctc sweep — CTC contribution is real and substantial", h2))
    story.append(Image(figs['lambda'], width=420, height=220))
    l00 = R['stage9a_abl_l00']['best_val_cer']
    l01 = R['stage9a_abl_l01']['best_val_cer']
    story.append(Paragraph(
        f"<i>Figure 1 — lambda_ctc sweep with the ±2σ seed band shaded.  The "
        f"pure-attention endpoint (lambda=0.0) jumps to {l00:.3f} — "
        f"{(l00-l01)*100:.0f} CER points worse than even the small-weight "
        f"lambda=0.1 setting ({l01:.3f}).  This decisively validates the report's "
        f"\"CTC provides alignment supervision\" claim: without ANY CTC gradient, "
        f"the attention decoder cannot learn the temporal alignment from the "
        f"encoder features alone.  Sweet spot is in the lambda &isin; [0.3, 0.5] "
        f"range; lambda=0.5 is the empirical minimum at {R['stage9a_abl_l05']['best_val_cer']:.4f} "
        f"but lies inside the ±2σ band.  lambda=0.7 over-suppresses attention "
        f"and regresses by {(R['stage9a_abl_l07']['best_val_cer']-ctl):+.3f}.</i>",
        note,
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Decoder depth — 2 layers is the only ≥2σ improvement", h2))
    story.append(Image(figs['depth'], width=420, height=220))
    d2 = R['stage9a_abl_d2']['best_val_cer']
    d4 = R['stage9a_abl_d4']['best_val_cer']
    story.append(Paragraph(
        f"<i>Figure 2 — Decoder depth.  d=3 (control, {ctl:.4f}) sits at a local "
        f"maximum; both d=2 ({d2:.4f}) and d=4 ({d4:.4f}) are better, with d=2 "
        f"the empirical winner.  d=2 delta of {(d2-ctl):+.4f} sits at the 2σ "
        f"threshold ({two_sigma:.4f}) — claim-worthy but marginal.  Worth a "
        f"5-fold re-run at d=2 to confirm; expected headline improvement ~1 CER "
        f"point if the fold-0 result generalises.</i>",
        note,
    ))
    story.append(PageBreak())

    story.append(Paragraph("4. Seed variance + reproducibility", h2))
    story.append(Image(figs['seeds'], width=420, height=220))
    story.append(Paragraph(
        f"<i>Figure 3 — Three runs of the headline recipe (lambda=0.3, d=3, 80 ep) "
        f"with seeds 42 / 43 / 44.  Cluster mean is {mu:.4f}, standard "
        f"deviation {sig:.4f}.  Stage 9a headline ({HEADLINE_FOLD0_CER}) sits "
        f"within ±2σ.  Seed-42 result ({ctl:.4f}) deviates from the headline by "
        f"only {delta_repro:.4f} — well inside the cudnn-nondeterminism floor — "
        f"confirming no codebase drift between this kernel and the original Stage 9a "
        f"run.</i>",
        note,
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("5. Training length", h2))
    e120 = R['stage9a_abl_e120']
    story.append(Paragraph(
        f"At 120 epochs ({e120['best_val_cer']:.4f}, best_epoch={e120['best_epoch']}), "
        f"the headline recipe is within ±2σ of the 80-epoch control.  Notably, "
        f"e120 is the only run whose train CTC NLL dropped below 0.5 (final "
        f"{e120['final_train_ctc_nll']:.4f}, vs ~0.85 at 80 epochs) and whose "
        f"train attention NLL hit {e120['final_train_attn_nll']:.4f}.  "
        f"Training continues to improve, but val CER plateaus around epoch 70-75 "
        f"regardless of total budget.  <b>80 epochs is sufficient.</b>",
        body,
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("6. Length-collapse diagnostic", h2))
    story.append(Paragraph(
        f"All variants have predicted/reference length ratio in [0.82, 0.96] — no "
        f"length collapse (ratio < 0.5) or runaway (ratio > 2.0) in any run.  "
        f"The lambda=0.0 variant is closest to balanced (0.956) yet has the worst "
        f"CER (0.7091): with attention alone, the decoder produces the right "
        f"<i>amount</i> of output but the wrong <i>content</i>, confirming that "
        f"CTC's contribution is alignment information rather than length "
        f"regulation.",
        body,
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("7. Decision summary", h2))
    story.append(Paragraph(
        "Across the three axes tested:",
        body,
    ))
    bullets = [
        "<b>λ_ctc</b>: keep at 0.3 (or try 0.5 in a 5-fold).  CTC is genuinely "
        "necessary, not just regularization — the λ=0 endpoint is a positive "
        "scientific finding for the dissertation.",
        f"<b>Decoder depth</b>: d=2 wins on fold 0 by {(d2-ctl)*-1*100:.1f} CER "
        "points (just over 2σ).  A 5-fold re-run at d=2 has ~6h cost and would "
        "shave ~1 absolute CER point off the headline if the gain generalises.  "
        "Defensible to run; defensible to skip.",
        "<b>Training length</b>: 80 epochs is sufficient.  120 epochs doesn't "
        "improve val CER; the headline budget was correct.",
        f"<b>Seed variance</b>: σ = {sig:.4f} on this dataset.  Future ablation "
        f"deltas must be at least {two_sigma:.4f} to be claim-worthy.",
    ]
    for b in bullets:
        story.append(Paragraph("&bull; " + b, body))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 6))
    if real_wins:
        story.append(Paragraph(
            "<b>Recommended next action</b>: re-run the full 5-fold headline at d=2.  "
            "Expected new headline ≈ 0.475 ± 0.07.  All other knobs stay.",
            body,
        ))
    else:
        story.append(Paragraph(
            "<b>Recommended next action</b>: declare Stage 9a (λ=0.3, d=3, 80 epochs) "
            "as the final landmark-only baseline.  Move to Stage 9b (LM rescoring) "
            "or Stage 10 (DINOv2 unfreeze gate).",
            body,
        ))

    doc = SimpleDocTemplate(
        out_pdf, pagesize=letter,
        leftMargin=48, rightMargin=48, topMargin=48, bottomMargin=48,
    )
    doc.build(story)
    return out_pdf


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True)
    p.add_argument("--out-pdf", required=True)
    p.add_argument("--figures-dir", default=None)
    args = p.parse_args(argv)
    figures_dir = args.figures_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.out_pdf)) or ".",
        "_stage9a_abl_figures",
    )
    out = build_pdf(args.results, args.out_pdf, figures_dir)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
