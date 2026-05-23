import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Directory setup
out_dir = r"c:\Users\csath\Downloads\ppi_gnn_combined_dataset\edits"
os.makedirs(out_dir, exist_ok=True)

def draw_box(ax, center, width, height, text, bg_color='#f0f4f8', edge_color='#b0c4de', text_size=10, font_weight='normal'):
    x = center[0] - width / 2
    y = center[1] - height / 2
    rect = patches.Rectangle((x, y), width, height, linewidth=1.5, edgecolor=edge_color, facecolor=bg_color, zorder=2)
    ax.add_patch(rect)
    ax.text(center[0], center[1], text, ha='center', va='center', fontsize=text_size, fontweight=font_weight, zorder=3, wrap=True)

def draw_arrow(ax, start, end, color='#4a5d73'):
    ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=color, lw=1.5), zorder=1)

def draw_flowchart(with_architectures=False):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    title = "Neonatal Sepsis Cohort Policy & Architecture Flow" if with_architectures else "Neonatal Sepsis Cohort Policy & Split Flow"
    ax.text(50, 95, title, fontsize=16, fontweight='bold', ha='center')

    # Data Sources
    draw_box(ax, (20, 80), 25, 8, "GSE25504 (Illumina/Affy)\nN = 170\nNeonatal Sepsis/Control", '#eef1f6', '#9ca8b8', 9)
    draw_box(ax, (20, 65), 25, 8, "GSE69686 (Affy)\nN = 149\nNeonatal Sepsis/Control", '#eef1f6', '#9ca8b8', 9)
    draw_box(ax, (20, 30), 25, 8, "GSE26440 (Affy)\nN = 104\nPediatric Sepsis/Control", '#fff5e6', '#eebd82', 9)
    
    # Preprocessing
    draw_box(ax, (50, 72.5), 22, 12, "Data Harmonization\n(ComBat EB)\nPreserving Biological Signal", '#e6f4ea', '#82c092', 10, 'bold')
    
    draw_arrow(ax, (32.5, 80), (40, 75))
    draw_arrow(ax, (32.5, 65), (40, 70))
    
    # Internal Validation Model / Flow
    if with_architectures:
        y_model = 72.5
        draw_box(ax, (80, y_model), 25, 16, "Internal 5-Fold CV\n\nMultiplex-HGCN-DANN-MLP\nGen: AUROC 0.980\n\nBaseline GCN: AUROC 0.685\nLogistic Reg: AUROC 0.913", '#e8eaf6', '#7986cb', 9)
        draw_arrow(ax, (61, 72.5), (67.5, 72.5))
        
        # External holdout
        draw_box(ax, (80, 30), 25, 12, "Locked External Holdout\n\nNeonatal HGCN-MLP Applied\nAUROC 0.986", '#fbe9e7', '#ff8a65', 9)
        draw_arrow(ax, (32.5, 30), (67.5, 30))
        draw_arrow(ax, (80, 64.5), (80, 36), color="#7986cb") # Model transfer
        ax.text(82, 50, "Frozen Weights", fontsize=8, fontstyle='italic', rotation=270, color="#7986cb")
        
    else:
        draw_box(ax, (80, 72.5), 20, 10, "Internal Training &\n5-Fold CV Module\n(Neonatal)", '#e8eaf6', '#7986cb', 10, 'bold')
        draw_arrow(ax, (61, 72.5), (70, 72.5))
        
        draw_box(ax, (80, 30), 20, 10, "Locked External Holdout\nValidation\n(Pediatric Out-of-Distribution)", '#fbe9e7', '#ff8a65', 10, 'bold')
        draw_arrow(ax, (32.5, 30), (70, 30))
        draw_arrow(ax, (80, 67.5), (80, 35), color="#7986cb") 
        ax.text(82, 50, "Apply Trained Model", fontsize=8, fontstyle='italic', rotation=270, color="#7986cb")

    # Legend / Annotation
    if with_architectures:
        draw_box(ax, (20, 10), 30, 8, "V12 Pure-GNN Ablation (No MLP)\nAUROC ~0.781 (Collapsed)", '#ffebee', '#ef9a9a', 9)
        
    plt.tight_layout()
    file_suffix = "with_architectures" if with_architectures else "no_architectures"
    filepath = os.path.join(out_dir, f"neonatal_sepsis_flow_{file_suffix}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

# Generate both
draw_flowchart(with_architectures=False)
draw_flowchart(with_architectures=True)

print(f"Generated flowcharts in {out_dir}")
