import os
import shutil
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = r"C:\Users\terry\Downloads\Projects\ISEF"
ACSEF = os.path.join(ROOT, "ACSEF_Final_Submission")
FIGURES = os.path.join(ACSEF, "figures")
IMAGES = os.path.join(ACSEF, "images")
DOCS = os.path.join(ACSEF, "acsef_documents", "publication_package")

os.makedirs(FIGURES, exist_ok=True)
os.makedirs(IMAGES, exist_ok=True)
os.makedirs(DOCS, exist_ok=True)


POSTER_TEXT = {
    "title": "Graph-Guided Domain-Adversarial Learning for Neonatal Sepsis Biomarker Discovery",
    "authors": "Student Researcher | ACSEF 2026",
    "objective": (
        "Can a biology-constrained multiplex hypergraph model improve external diagnosis "
        "of neonatal sepsis while preserving biomarker interpretability?"
    ),
    "methods": (
        "Data: GSE25504 + GSE69686 for development, GSE26440 for external validation. "
        "Pipeline: harmonization, batch correction, multiplex hypergraph relations "
        "(KEGG/STRING/Co-expression), relation attention, gene-scoring mask, MLP classifier, "
        "domain-adversarial head."
    ),
    "results": (
        "5-fold sepsis CV: AUROC 0.9796, Accuracy 0.9779, F1 0.9733. "
        "External GSE26440: AUROC 0.9856, Accuracy 0.9519, F1 0.9697. "
        "Top biomarkers include TNFAIP6, S100A12, RETN, and CD52."
    ),
    "conclusion": (
        "Multiplex-Hypergraph-DANN-MLP outperformed HGCN/GCN/GAT baselines and remained "
        "stable externally. Transfer experiments in osteogenesis imperfecta support broader "
        "rare-disease applicability."
    ),
}


def _block(ax, x, y, w, h, title, body, title_size=18, body_size=11):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=2.2, edgecolor="#1f3f5b"))
    ax.text(
        x + 0.012,
        y + h - 0.03,
        title,
        fontsize=title_size,
        fontweight="bold",
        color="#0d2d4a",
        va="top",
    )
    wrapped = "\n".join(textwrap.wrap(body, width=85))
    ax.text(
        x + 0.012,
        y + h - 0.07,
        wrapped,
        fontsize=body_size,
        color="#1d1d1d",
        va="top",
        linespacing=1.35,
    )


def _copy_raw_images():
    copied = []
    for name in sorted(os.listdir(FIGURES)):
        if not name.lower().endswith(".png"):
            continue
        if name == "fig_acsef_poster_layout_draft.png":
            continue
        src = os.path.join(FIGURES, name)
        dst = os.path.join(IMAGES, name)
        shutil.copy2(src, dst)
        copied.append(name)
    return copied


def _draw_image_panel(fig, bg_ax, x, y, w, h, title, image_path):
    bg_ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=1.8, edgecolor="#2f6690"))
    bg_ax.text(x + 0.01, y + h - 0.01, title, fontsize=11, fontweight="bold", color="#133a5e", va="top")

    img_ax = fig.add_axes([x + 0.008, y + 0.008, w - 0.016, h - 0.04])
    img_ax.axis("off")

    if os.path.exists(image_path):
        img = plt.imread(image_path)
        img_ax.imshow(img)
        img_ax.set_aspect("auto")
    else:
        img_ax.text(0.5, 0.5, f"Missing image\n{os.path.basename(image_path)}", ha="center", va="center", fontsize=12)


def build_poster():
    copied_images = _copy_raw_images()

    # Draft print layout: 48x36in landscape at 150 DPI.
    fig = plt.figure(figsize=(48, 36), dpi=150)
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)
    bg.axis("off")

    bg.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f7fbff", edgecolor="none"))
    bg.add_patch(Rectangle((0, 0.90), 1, 0.10, facecolor="#173f5f", edgecolor="none"))

    bg.text(
        0.02,
        0.955,
        POSTER_TEXT["title"],
        fontsize=36,
        fontweight="bold",
        color="white",
        va="center",
    )
    bg.text(0.02, 0.915, POSTER_TEXT["authors"], fontsize=22, color="#e8f1fa", va="center")

    _block(bg, 0.02, 0.69, 0.30, 0.18, "Objective", POSTER_TEXT["objective"], body_size=14)
    _block(bg, 0.34, 0.69, 0.32, 0.18, "Methods", POSTER_TEXT["methods"], body_size=12)
    _block(bg, 0.68, 0.69, 0.30, 0.18, "Primary Results", POSTER_TEXT["results"], body_size=12)
    _block(bg, 0.02, 0.61, 0.96, 0.06, "Conclusions", POSTER_TEXT["conclusion"], title_size=16, body_size=11)

    panels = [
        (0.02, 0.33, 0.30, 0.25, "A. ROC Comparisons", "fig_roc_comparisons.png"),
        (0.34, 0.33, 0.32, 0.25, "B. Architecture Flowchart", "fig_architecture_flowchart.png"),
        (0.68, 0.33, 0.30, 0.25, "C. Biomarker Attributions", "fig_biomarker_attributions.png"),
        (0.02, 0.05, 0.30, 0.25, "D. External Validation", "fig_external_validation_gse26440.png"),
        (0.34, 0.05, 0.32, 0.25, "E. OI Scaling Summary", "fig_osteogenesis_scaling_summary.png"),
        (0.68, 0.05, 0.30, 0.25, "F. Model Metric Radar", "fig_model_metric_radar.png"),
    ]

    for x, y, w, h, title, filename in panels:
        _draw_image_panel(fig, bg, x, y, w, h, title, os.path.join(IMAGES, filename))

    bg.text(
        0.02,
        0.015,
        "Poster panels now render directly from ACSEF_Final_Submission/images/*.png",
        fontsize=12,
        color="#4a4a4a",
    )
    bg.text(
        0.98,
        0.015,
        f"Images synced: {len(copied_images)}",
        fontsize=12,
        color="#4a4a4a",
        ha="right",
    )

    png_out = os.path.join(FIGURES, "fig_acsef_poster_layout_draft.png")
    png_out_images = os.path.join(IMAGES, "acsef_poster_layout_draft.png")
    pdf_out = os.path.join(DOCS, "acsef_poster_layout_draft.pdf")
    svg_out = os.path.join(DOCS, "acsef_poster_layout_draft.svg")

    fig.savefig(png_out, dpi=150, bbox_inches="tight")
    fig.savefig(png_out_images, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_out, dpi=300, bbox_inches="tight")
    fig.savefig(svg_out, bbox_inches="tight")
    plt.close(fig)

    print(f"Generated: {png_out}")
    print(f"Generated: {png_out_images}")
    print(f"Generated: {pdf_out}")
    print(f"Generated: {svg_out}")


if __name__ == "__main__":
    build_poster()
