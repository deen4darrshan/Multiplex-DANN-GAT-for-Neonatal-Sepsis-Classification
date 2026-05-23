import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Directory setup
out_dir = r"c:\Users\csath\Downloads\ppi_gnn_combined_dataset\edits"
os.makedirs(out_dir, exist_ok=True)

def draw_box(ax, x, y, width, height, title, subtitle, details, bg_color, border_color):
    """Draws a styled box with a title, subtitle, and bulleted details."""
    rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor=border_color, 
                             facecolor=bg_color, zorder=2, transform=ax.transData)
    # Add a subtle drop shadow
    shadow = patches.Rectangle((x + 0.5, y - 0.5), width, height, linewidth=0, 
                               facecolor='#000000', alpha=0.1, zorder=1, transform=ax.transData)
    ax.add_patch(shadow)
    ax.add_patch(rect)
    
    # Title
    ax.text(x + width/2, y + height - 1.5, title, ha='center', va='top', 
            fontsize=12, fontweight='bold', color='#1a202c', zorder=3)
    
    # Subtitle
    if subtitle:
        ax.text(x + width/2, y + height - 3.5, subtitle, ha='center', va='top', 
                fontsize=10, fontstyle='italic', color='#4a5568', zorder=3)
        
    # Details (Bullets)
    if details:
        text_y = y + height - 6.5
        for line in details:
            ax.text(x + 1.5, text_y, f"• {line}", ha='left', va='center', 
                    fontsize=9, color='#2d3748', zorder=3)
            text_y -= 1.8

def draw_arrow(ax, start_x, start_y, end_x, end_y, text=""):
    """Draws a thick, styled arrow between components, optionally with text."""
    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle="-|>", color='#4a5568', lw=2.5, 
                                mutation_scale=20, shrinkA=0, shrinkB=0), 
                zorder=1)
    if text:
        mid_x = (start_x + end_x) / 2
        mid_y = ((start_y + end_y) / 2) + 2.5  # Moved text up higher away from arrow
        ax.text(mid_x, mid_y, text, ha='center', va='bottom', fontsize=9, 
                fontweight='bold', color='#2b6cb0', zorder=3)

# Setup Figure
fig, ax = plt.subplots(figsize=(24, 8))  # Widened figure even more
ax.set_xlim(0, 145) # Increased xlim
ax.set_ylim(0, 50)
ax.axis('off')

# Title
ax.text(72.5, 45, "Hybrid Multiplex-Hypergraph Domain-Adversarial Pipeline", 
        fontsize=22, fontweight='bold', ha='center', color='#1a202c')

# Box dimensions
bx, by, bw, bh = 4, 15, 17, 14
spacing = 23  # Increased spacing between boxes to give arrows >6 units of width

# 1. Input Layer
draw_box(ax, bx, by, bw, bh, 
         "1. Transcriptomic Input", "ComBat Harmonized", 
         ["N = 319 Neonates", "GSE25504 & GSE69686", "Top 2,000 MAD Genes", "Removed batch effects"], 
         '#ebf8fa', '#319795')

# Arrow 1
draw_arrow(ax, bx + bw, by + bh/2, bx + spacing, by + bh/2, "Gene\nExpression")

# 2. Graph Construction
draw_box(ax, bx + spacing, by, bw, bh, 
         "2. Multiplex Prior", "3-Relation Hypergraph", 
         ["KEGG Pathway links", "STRING PPI (score > 700)", "Fold-specific Co-expression", "Protects against leakage"], 
         '#ebf4ff', '#3182ce')

# Arrow 2
draw_arrow(ax, bx + spacing + bw, by + bh/2, bx + spacing*2, by + bh/2, "Graph\nStructure")

# 3. GNN Layer
draw_box(ax, bx + spacing*2, by, bw, bh, 
         "3. Hypergraph Conv", "HGCN Module", 
         ["Propagates gene signals", "Along biological pathways", "Learns relation attention", "Extracts structural features"], 
         '#e9d8fd', '#805ad5')

# Arrow 3
draw_arrow(ax, bx + spacing*2 + bw, by + bh/2, bx + spacing*3, by + bh/2, "Gene\nEmbeddings")

# 4. Fusion & Classification
draw_box(ax, bx + spacing*3, by, bw, bh, 
         "4. MLP Classifier", "Non-linear Integration", 
         ["Gene-scoring attention mask", "Filters unimportant genes", "Captures complex interactions", "Critical for performance"], 
         '#fefcbf', '#d69e2e')

# Arrow 4
draw_arrow(ax, bx + spacing*3 + bw, by + bh/2, bx + spacing*4, by + bh/2, "Diagnostic\nScore")

# 5. Output & DANN
draw_box(ax, bx + spacing*4, by, bw, bh, 
         "5. Multi-Task Output", "Diagnosis & DANN", 
         ["Primary: Sepsis vs. Control", "Secondary: Batch Prediction", "Gradient Reversal Layer", "Suppresses cohort shortcuts"], 
         '#fff5f5', '#e53e3e')

# Arrow 5
draw_arrow(ax, bx + spacing*4 + bw, by + bh/2, bx + spacing*5, by + bh/2, "Model\nTransfer")

# 6. External Validation
draw_box(ax, bx + spacing*5, by, bw, bh, 
         "6. External Validation", "Generalizability & Scaling", 
         ["Pediatric Holdout (AUROC 0.986)", "Alzheimer's (AUROC 0.944)", "Osteogenesis (AUROC 0.774)", "No performance drop-off"], 
         '#f0fff4', '#38a169')

# Explanatory Footer Bar
rect = patches.Rectangle((4, 2), 137, 6, linewidth=1, edgecolor='#cbd5e0', 
                         facecolor='#f7fafc', zorder=1)
ax.add_patch(rect)
ax.text(72.5, 5, "Why it works: The Multiplex Graph embeds known biology (making it explainable), while the MLP captures complex signal patterns\n(achieving 0.980 CV AUROC), and the DANN ensures the model learns real disease signals instead of laboratory hardware noise.", 
        fontsize=12, ha='center', va='center', color='#2d3748', wrap=True)

plt.tight_layout()
filepath = os.path.join(out_dir, "architecture_pipeline_aesthetic.png")
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()

print(f"Generated aesthetic architecture flowchart at {filepath}")
