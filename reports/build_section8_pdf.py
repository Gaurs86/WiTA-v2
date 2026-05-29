"""
reports/build_section8_pdf.py — Generate Section 8 of the Stage 1 / 2 report
and merge it into the existing PDF.

Run after Stage 1 v3 finishes:

    python reports/build_section8_pdf.py \
        --existing-pdf  /path/to/stage1_stage2_report.pdf \
        --scatter-png   /path/to/stage1v3_per_signer_scatter.png \
        --out-pdf       /path/to/stage1_stage2_report_v2.pdf

If --scatter-png is missing or absent, the PDF is built with a textual
placeholder where the figure would go.  If --existing-pdf is missing, only
the Section 8 standalone PDF is written (--out-pdf) and merging is skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

# Stage 1 v3 measured numbers (from the post-Stage-1-v3 prompt §1.1)
TABLE_5FOLD = [
    ('no_dann',  0.6448, 0.0520, '+0.000', 'new baseline'),
    ('dann_a1',  0.6660, 0.0296, '+0.0212', 'hurts ~2 pts'),
    ('dann_a03', 0.6688, 0.0255, '+0.0240', 'hurts ~2 pts'),
]
PER_FOLD_NO_DANN = [
    (0, 0.631), (1, 0.695), (2, 0.699), (3, 0.621), (4, 0.578),
]


def build_section8_pdf(out_pdf: str, scatter_png: str | None) -> str:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image,
        Table, TableStyle,
    )
    from reportlab.lib import colors

    styles = getSampleStyleSheet()
    h1 = styles['Heading1']
    h2 = styles['Heading2']
    body = styles['BodyText']
    code = ParagraphStyle('mono', parent=body,
        fontName='Courier', fontSize=9, leading=11)

    story = []
    story.append(Paragraph(
        "Section 8 — Revised Stage 1 baseline under 5-fold subject CV",
        h1,
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "This section amends the Stage 1 / Stage 2 report after the Stage 1 v3 sweep "
        "(3 variants &times; 5 subject-disjoint CV folds = 15 runs, 80 epochs each) "
        "completed.  Two findings invalidate the prior single-split numbers: "
        "(1) the DANN hypothesis is <b>falsified</b>; (2) single-split CER is "
        "<b>statistically fragile</b> with std = 0.052 across folds, well above the "
        "0.04 threshold.  Going forward every stage reports 5-fold CV mean &plusmn; std "
        "on the locked <i>manifests/subject_cv5.json</i> split.",
        body,
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("8.1 Revised Stage 1 baseline (5-fold)", h2))
    tbl_data = [
        ['Variant', 'Mean CER', 'Std', '&Delta; vs no_dann', 'Verdict'],
    ]
    for name, m, s, delta, verdict in TABLE_5FOLD:
        tbl_data.append([name, f'{m:.4f}', f'{s:.4f}', delta, verdict])
    t = Table(tbl_data, hAlign='LEFT', colWidths=[80, 60, 50, 80, 110])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Per-fold best val CER for variant <b>no_dann</b> (locked Stage 1 v2 recipe):",
        body,
    ))
    pf_data = [['Fold', 'Best val CER']]
    for f, c in PER_FOLD_NO_DANN:
        pf_data.append([str(f), f'{c:.4f}'])
    pf_data.append(['Mean &plusmn; std', '0.6448 &plusmn; 0.052'])
    t2 = Table(pf_data, hAlign='LEFT', colWidths=[80, 100])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    story.append(t2)
    story.append(Spacer(1, 16))

    story.append(Paragraph("8.2 Per-signer val CER distribution", h2))
    story.append(Paragraph(
        "Each of the 39 English-subset signers appears in exactly one CV fold's "
        "val set, so the 39-entry per-signer CER vector is comparable across "
        "stages run on the same manifest.  Range: 0.431 &rarr; 0.898; "
        "across-signer std 0.096; clearly bimodal.",
        body,
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Easy regime</b> (CER &le; 0.55, ~10 signers): KIS, YJH, KHY, HYW, KSH, "
        "JSA, YSY, KSJ.<br/>"
        "<b>Hard regime</b> (CER &ge; 0.75, ~8 signers): PHW (0.898), PJH (0.842), "
        "SYB (0.809), KJM (0.746), KNY (0.737), LKS (0.730), KIM (0.723), YMG (0.728).",
        body,
    ))
    story.append(Spacer(1, 8))
    if scatter_png and os.path.exists(scatter_png):
        story.append(Image(scatter_png, width=460, height=190))
        story.append(Paragraph(
            "<i>Figure 8.1 &mdash; Per-signer best val CER across the 5 CV folds for "
            "the Stage 1 v3 no_dann variant.  Each signer appears in exactly one "
            "fold's val set.  Red dashed line marks the hard-regime threshold (0.75), "
            "green dashed line marks the easy-regime threshold (0.55).</i>",
            body,
        ))
    else:
        story.append(Paragraph(
            "<i>[Placeholder for Figure 8.1: per-signer val CER scatter. "
            "Generate via the Stage 1 v3 notebook Cell 9 and re-run this script "
            "with --scatter-png /path/to/stage1v3_per_signer_scatter.png.]</i>",
            code,
        ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("8.3 Why DANN was dropped", h2))
    story.append(Paragraph(
        "Stage 1 v3's decision tree confirms DANN only if &Delta;(B - A) &le; -0.05.  "
        "Measured deltas were +0.021 (&alpha;=1.0) and +0.024 (&alpha;=0.3) &mdash; "
        "DANN <i>hurts</i> by ~2 CER points.  Paired Wilcoxon (n=5) gives "
        "W&#8203;&#8314;=4, W&#8203;&#8331;=11; trend favours the control.  "
        "Per-signer breakdown is regression-to-mean: DANN penalises easy signers "
        "(KIS +0.146, YJH +0.131) while marginally helping the hardest "
        "(PHW &minus;0.045, SYB &minus;0.049).  Net mean &Delta; = +0.020.  "
        "DANN is removed from all downstream stages.",
        body,
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("8.4 Implications for the reporting protocol", h2))
    story.append(Paragraph(
        "The original Stage 1/2 PDF's single-split CER figure (0.687) is correct for "
        "the run it depicts but is not the canonical Stage 1 number.  All future "
        "stages will:",
        body,
    ))
    bullets = [
        "Emit 5-fold mean &plusmn; std + a paired Wilcoxon for every model change.",
        "Emit the standardised per-signer scatter (reports/template/per_signer_scatter.py) "
        "so cross-stage comparisons land on the same axes.",
        "Cite single-split numbers only as exploratory or one-off speed checks.",
        "Use the locked manifest manifests/subject_cv5.json (seed=42).",
    ]
    for b in bullets:
        story.append(Paragraph("&bull; " + b, body))

    doc = SimpleDocTemplate(
        out_pdf, pagesize=letter,
        leftMargin=48, rightMargin=48, topMargin=48, bottomMargin=48,
    )
    doc.build(story)
    return out_pdf


def merge_pdfs(existing: str, section: str, out: str) -> str:
    from pypdf import PdfReader, PdfWriter
    w = PdfWriter()
    for src in (existing, section):
        for page in PdfReader(src).pages:
            w.add_page(page)
    with open(out, 'wb') as f:
        w.write(f)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--existing-pdf', default=None,
        help='Existing Stage 1 / 2 PDF to prepend (optional).')
    p.add_argument('--scatter-png',  default=None,
        help='Per-signer scatter PNG generated by Stage 1 v3 Cell 9.')
    p.add_argument('--out-pdf',      required=True,
        help='Output PDF path (final merged document if --existing-pdf set, '
             'else just the Section 8 standalone).')
    args = p.parse_args(argv)

    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        section_pdf = tmp.name
    build_section8_pdf(section_pdf, args.scatter_png)
    print(f'built  {section_pdf}')

    if args.existing_pdf and os.path.exists(args.existing_pdf):
        merge_pdfs(args.existing_pdf, section_pdf, args.out_pdf)
        print(f'merged {args.existing_pdf} + section -> {args.out_pdf}')
    else:
        os.replace(section_pdf, args.out_pdf)
        print(f'section-only PDF -> {args.out_pdf}')


if __name__ == '__main__':
    main()
