# -*- coding: utf-8 -*-
"""
Regenerate transcriptome figures (fig12, fig13) with English labels
for Human Brain Mapping submission.
"""
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Use English-compatible font
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.dpi'] = 300

base_dir = r"D:\epilepsy_MIND"
result_dir = os.path.join(base_dir, "results")
fig_dir = os.path.join(result_dir, "figures")

# Load data
region_df = pd.read_csv(os.path.join(result_dir, "E1_region_vulnerability.csv"))
gene_df = pd.read_csv(os.path.join(result_dir, "E2_gene_vulnerability_correlations.csv"))

# Sort
region_df = region_df.sort_values('vulnerability_score', ascending=False).reset_index(drop=True)
gene_df = gene_df.sort_values('p').reset_index(drop=True)

# ========== fig12: Top 30 vulnerability regions ==========
print("Generating fig12: regional vulnerability bar plot...")
top_reg = region_df.head(30)
colors = ['#c0392b' if 'lh_' in r else '#2980b9' for r in top_reg['region']]

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(range(len(top_reg)), top_reg['vulnerability_score'], color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
ax.set_xticks(range(len(top_reg)))
ax.set_xticklabels([r.replace('lh_','L-').replace('rh_','R-') for r in top_reg['region']],
                    rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Epilepsy Vulnerability Score', fontsize=12, fontweight='bold')
ax.set_xlabel('Brain Region', fontsize=12, fontweight='bold')
ax.set_title('Top 30 Epilepsy-Vulnerable Brain Regions\n(Red = Left Hemisphere, Blue = Right Hemisphere)',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3)
# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#c0392b', label='Left hemisphere'),
                   Patch(facecolor='#2980b9', label='Right hemisphere')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig12_region_vulnerability.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  fig12 saved.")

# ========== fig13: Gene volcano plot ==========
print("Generating fig13: gene volcano plot...")
fig, ax = plt.subplots(figsize=(10, 7))

log_p = -np.log10(gene_df['p'].clip(lower=1e-300))

# Non-significant (gray)
non_sig = gene_df['p'] >= 0.001
ax.scatter(gene_df.loc[non_sig, 'r'], log_p[non_sig], s=8, alpha=0.25, color='#bdc3c7', label='NS (P ≥ 0.001)')

# Significant (red)
sig = gene_df['p'] < 0.001
ax.scatter(gene_df.loc[sig, 'r'], log_p[sig], s=15, alpha=0.7, color='#c0392b', label='Significant (P < 0.001)')

# Label top genes
top_genes = gene_df.head(20)
for _, row in top_genes.iterrows():
    ax.annotate(str(row['gene']),
                (row['r'], -np.log10(max(row['p'], 1e-300))),
                fontsize=8, alpha=0.85,
                xytext=(5, 3), textcoords='offset points')

ax.set_xlabel('Pearson Correlation Coefficient (r)', fontsize=12, fontweight='bold')
ax.set_ylabel('-log$_{10}$(P-value)', fontsize=12, fontweight='bold')
ax.set_title('Spatial Association Between Gene Expression and Epilepsy Vulnerability\n(Volcano Plot, n = 20,736 genes)',
             fontsize=13, fontweight='bold')
ax.axhline(y=-np.log10(0.001), color='#e74c3c', linestyle='--', alpha=0.6, linewidth=1, label='P = 0.001')
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig13_gene_volcano.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  fig13 saved.")

print("\nDone! Both figures regenerated with English labels at 300 DPI.")
