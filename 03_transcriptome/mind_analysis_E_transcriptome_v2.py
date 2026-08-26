# -*- coding: utf-8 -*-
"""
AHBA Transcriptomic spatial association analysis v2
Uses precomputed AllenHBA_DK_ExpressionMatrix.tsv (from French 2015 Frontiers)
Spatially correlates epilepsy-associated regional abnormality map with AHBA gene expression
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ========== Configuration ==========
base_dir = r"D:\epilepsy_MIND"
ahba_dir = os.path.join(base_dir, "ahba_data")
proc_dir = os.path.join(base_dir, "processed")
result_dir = os.path.join(base_dir, "results")
fig_dir = os.path.join(result_dir, "figures")

# ========== Step 1: Load AHBA expression matrix ==========
print("=" * 60)
print("Step 1: Loading AHBA Desikan-Killiany gene expression matrix")
print("=" * 60)

expr_df = pd.read_csv(os.path.join(ahba_dir, "AllenHBA_DK_ExpressionMatrix.tsv"), sep='\t')
print(f"Raw matrix: {expr_df.shape}")

# Gene names in first column
gene_names = expr_df.iloc[:, 0].values
print(f"Number of genes: {len(gene_names)}")

# Extract 68 region columns (excluding first gene name column and "Average donor correlation to median")
region_cols = [c for c in expr_df.columns if c.startswith('ctx-')]
print(f"Number of region columns: {len(region_cols)}")

# Expression matrix: genes x regions
expression = expr_df[region_cols].values
print(f"Expression matrix: {expression.shape}")

# Region name mapping: ctx-lh-xxx -> lh_xxx
def map_region_name(name):
    if name.startswith('ctx-lh-'):
        return 'lh_' + name[7:]
    elif name.startswith('ctx-rh-'):
        return 'rh_' + name[7:]
    return name

ahba_regions = [map_region_name(c) for c in region_cols]
print(f"AHBA regions example: {ahba_regions[:5]}")

# ========== Step 2: Compute region-level epilepsy vulnerability score ==========
print("\n" + "=" * 60)
print("Step 2: Computing region-level epilepsy vulnerability score")
print("=" * 60)

# Load data
thick = pd.read_csv(os.path.join(proc_dir, "thickness_aligned.csv"), index_col=0)
vol = pd.read_csv(os.path.join(proc_dir, "volume_aligned.csv"), index_col=0)
area = pd.read_csv(os.path.join(proc_dir, "area_aligned.csv"), index_col=0)
clin = pd.read_csv(os.path.join(proc_dir, "clinical_aligned.csv"), index_col=0)
resect = pd.read_csv(os.path.join(proc_dir, "resection_aligned.csv"), index_col=0)
with open(os.path.join(proc_dir, "region_names.txt")) as f:
    region_names = [line.strip() for line in f]

subject_ids = thick.index.tolist()
clin = clin[~clin.index.duplicated(keep='first')].loc[subject_ids]

# Region mapping
def map_resect_region(name):
    if name.startswith('ctx-lh-'): return 'lh_' + name[7:]
    if name.startswith('ctx-rh-'): return 'rh_' + name[7:]
    return None
resect_map = {c: map_resect_region(c) for c in resect.columns if map_resect_region(c) and map_resect_region(c) in region_names}
resect_cortex = resect[list(resect_map.keys())].rename(columns=resect_map)[region_names]

# Load MIND networks
mind_nets = np.load(os.path.join(proc_dir, "mind_networks.npy"))

# Resection frequency
resection_freq = (resect_cortex > 0).mean().values

# Compute total connection strength effect size per region
total_strength_all = np.abs(mind_nets).sum(axis=2)
from scipy import stats
effect_sizes = []
for r in range(len(region_names)):
    resected_mask = resect_cortex.iloc[:, r].values > 0
    if resected_mask.sum() < 5 or (~resected_mask).sum() < 5:
        effect_sizes.append(0); continue
    g1 = total_strength_all[resected_mask, r]
    g2 = total_strength_all[~resected_mask, r]
    n1, n2 = len(g1), len(g2)
    s1, s2 = g1.std(), g2.std()
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    effect_sizes.append((g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0)
effect_sizes = np.array(effect_sizes)

# Epilepsy vulnerability score = resection frequency x |effect size|
vulnerability_score = resection_freq * np.abs(effect_sizes)
vulnerability_score = (vulnerability_score - vulnerability_score.min()) / (vulnerability_score.max() - vulnerability_score.min())

# Save region-level table
region_df = pd.DataFrame({
    'region': region_names,
    'resection_freq': resection_freq,
    'effect_size_d': effect_sizes,
    'vulnerability_score': vulnerability_score
}).sort_values('vulnerability_score', ascending=False)
region_df.to_csv(os.path.join(result_dir, "E1_region_vulnerability.csv"), index=False)

print(f"Top 10 vulnerable regions:")
print(region_df.head(10).to_string(index=False))

# ========== Step 3: Align AHBA regions with MIND regions ==========
print("\n" + "=" * 60)
print("Step 3: Aligning AHBA and MIND regions")
print("=" * 60)

# Find common regions
common_regions = [r for r in ahba_regions if r in region_names]
print(f"Number of common regions: {len(common_regions)}")

# AHBA expression matrix indices
ahba_idx = [ahba_regions.index(r) for r in common_regions]
# MIND vulnerability score indices
mind_idx = [region_names.index(r) for r in common_regions]

# Aligned expression matrix (genes x common regions)
expr_aligned = expression[:, ahba_idx]
# Aligned vulnerability score
vuln_aligned = vulnerability_score[mind_idx]

print(f"Aligned expression matrix: {expr_aligned.shape}")
print(f"Aligned vulnerability score: {vuln_aligned.shape}")

# ========== Step 4: Gene-vulnerability spatial association analysis ==========
print("\n" + "=" * 60)
print("Step 4: Spatial association between gene expression and epilepsy vulnerability")
print("=" * 60)

from scipy.stats import pearsonr, spearmanr

gene_results = []
n_genes = len(gene_names)
for g in range(n_genes):
    if g % 3000 == 0:
        print(f"  Processing gene {g}/{n_genes}...")
    
    expr_vals = expr_aligned[g, :]
    # Remove NaN
    valid = ~np.isnan(expr_vals)
    if valid.sum() < 30:
        gene_results.append({'gene': gene_names[g], 'r': 0, 'p': 1, 'n_valid': valid.sum()})
        continue
    
    r, p = pearsonr(expr_vals[valid], vuln_aligned[valid])
    gene_results.append({'gene': gene_names[g], 'r': r, 'p': p, 'n_valid': valid.sum()})

gene_df = pd.DataFrame(gene_results)

# FDR correction
from statsmodels.stats.multitest import multipletests
valid_p = gene_df['p'].values
_, fdr_p, _, _ = multipletests(valid_p, method='fdr_bh')
gene_df['fdr_p'] = fdr_p
gene_df = gene_df.sort_values('p')

print(f"\nGene association analysis complete!")
print(f"Total genes: {len(gene_df)}")
print(f"P<0.05: {(gene_df['p']<0.05).sum()}")
print(f"P<0.01: {(gene_df['p']<0.01).sum()}")
print(f"P<0.001: {(gene_df['p']<0.001).sum()}")
print(f"FDR<0.05: {(gene_df['fdr_p']<0.05).sum()}")
print(f"FDR<0.1: {(gene_df['fdr_p']<0.1).sum()}")

gene_df.to_csv(os.path.join(result_dir, "E2_gene_vulnerability_correlations.csv"), index=False)

# Top genes
print(f"\nTop 20 positively correlated genes (high expression -> high vulnerability):")
top_pos = gene_df[gene_df['r']>0].sort_values('p').head(20)
print(top_pos[['gene','r','p','fdr_p']].to_string(index=False))

print(f"\nTop 20 negatively correlated genes (high expression -> low vulnerability):")
top_neg = gene_df[gene_df['r']<0].sort_values('p').head(20)
print(top_neg[['gene','r','p','fdr_p']].to_string(index=False))

# ========== Step 5: GO/KEGG enrichment analysis ==========
print("\n" + "=" * 60)
print("Step 5: GO/KEGG enrichment analysis")
print("=" * 60)

try:
    import gseapy as gp
    
    # Take genes with P<0.01
    sig_genes = gene_df[gene_df['p']<0.01]['gene'].tolist()
    print(f"Number of candidate genes (P<0.01): {len(sig_genes)}")
    
    if len(sig_genes) > 20:
        # GO BP
        print("\nGO BP enrichment analysis...")
        go_enr = gp.enrichr(gene_list=sig_genes, gene_sets='GO_Biological_Process_2023',
                             organism='human', outdir=None, cutoff=0.1)
        go_res = go_enr.results.sort_values('Adjusted P-value')
        go_sig = go_res[go_res['Adjusted P-value']<0.1]
        print(f"GO BP significant terms (FDR<0.1): {len(go_sig)}")
        if len(go_sig) > 0:
            print(go_sig[['Term','Overlap','Adjusted P-value']].head(10).to_string(index=False))
            go_sig.to_csv(os.path.join(result_dir, "E3_GO_BP_enrichment.csv"), index=False)
        
        # KEGG
        print("\nKEGG pathway enrichment analysis...")
        kegg_enr = gp.enrichr(gene_list=sig_genes, gene_sets='KEGG_2021_Human',
                               organism='human', outdir=None, cutoff=0.1)
        kegg_res = kegg_enr.results.sort_values('Adjusted P-value')
        kegg_sig = kegg_res[kegg_res['Adjusted P-value']<0.1]
        print(f"KEGG significant terms (FDR<0.1): {len(kegg_sig)}")
        if len(kegg_sig) > 0:
            print(kegg_sig[['Term','Overlap','Adjusted P-value']].head(10).to_string(index=False))
            kegg_sig.to_csv(os.path.join(result_dir, "E4_KEGG_enrichment.csv"), index=False)
    else:
        print("Too few candidate genes, skipping enrichment analysis")
except ImportError:
    print("gseapy not installed, skipping enrichment analysis")
except Exception as e:
    print(f"Enrichment analysis exception: {e}")

# ========== Step 6: Visualization ==========
print("\n" + "=" * 60)
print("Step 6: Visualization")
print("=" * 60)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Figure 1: Region vulnerability bar plot
fig, ax = plt.subplots(figsize=(14, 6))
top_reg = region_df.head(30)
colors = ['#c0392b' if 'lh_' in r else '#2980b9' for r in top_reg['region']]
ax.bar(range(len(top_reg)), top_reg['vulnerability_score'], color=colors, alpha=0.8)
ax.set_xticks(range(len(top_reg)))
ax.set_xticklabels([r.replace('lh_','L-').replace('rh_','R-') for r in top_reg['region']], 
                    rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Epilepsy vulnerability score', fontsize=12)
ax.set_title('Top 30 epilepsy-vulnerable regions (red=left hemisphere, blue=right hemisphere)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig12_region_vulnerability.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 12: Region vulnerability bar plot")

# Figure 2: Volcano plot
fig, ax = plt.subplots(figsize=(10, 7))
log_p = -np.log10(gene_df['p'].clip(lower=1e-300))
ax.scatter(gene_df['r'], log_p, s=5, alpha=0.3, color='gray')
sig = gene_df['p'] < 0.001
ax.scatter(gene_df.loc[sig,'r'], log_p[sig], s=15, alpha=0.7, color='#c0392b')
top_genes = gene_df.head(15)
for _, row in top_genes.iterrows():
    ax.annotate(str(row['gene']), (row['r'], -np.log10(max(row['p'],1e-300))), fontsize=8, alpha=0.8)
ax.set_xlabel('Correlation coefficient r', fontsize=12)
ax.set_ylabel('-log10(P-value)', fontsize=12)
ax.set_title('Spatial association between gene expression and epilepsy vulnerability (volcano plot)', fontsize=14)
ax.axhline(y=-np.log10(0.001), color='red', linestyle='--', alpha=0.5, label='P=0.001')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig13_gene_volcano.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 13: Gene volcano plot")

# Figure 3: Example gene scatter plot
top_pos_gene = gene_df[gene_df['r']>0].iloc[0]['gene']
top_neg_gene = gene_df[gene_df['r']<0].iloc[0]['gene']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, gene, title in zip(axes, [top_pos_gene, top_neg_gene],
                             [f'{top_pos_gene} (positive correlation)', f'{top_neg_gene} (negative correlation)']):
    g_idx = list(gene_names).index(gene)
    expr_vals = expr_aligned[g_idx, :]
    valid = ~np.isnan(expr_vals)
    ax.scatter(expr_vals[valid], vuln_aligned[valid], s=30, alpha=0.6, color='#2980b9')
    z = np.polyfit(expr_vals[valid], vuln_aligned[valid], 1)
    p = np.poly1d(z)
    x_line = np.linspace(expr_vals[valid].min(), expr_vals[valid].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.7)
    r_val = gene_df[gene_df['gene']==gene]['r'].values[0]
    ax.set_xlabel(f'{gene} expression level', fontsize=11)
    ax.set_ylabel('Epilepsy vulnerability score', fontsize=11)
    ax.set_title(f'{title}\nr={r_val:.3f}', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig14_example_gene_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 14: Example gene scatter plot")

# ========== Summary ==========
print("\n" + "=" * 60)
print("AHBA transcriptomic analysis complete!")
print("=" * 60)
print(f"\nKey findings:")
print(f"  Analyzed {len(gene_df)} genes for spatial association with epilepsy vulnerability")
print(f"  P<0.05: {(gene_df['p']<0.05).sum()} genes")
print(f"  P<0.01: {(gene_df['p']<0.01).sum()} genes")
print(f"  P<0.001: {(gene_df['p']<0.001).sum()} genes")
if (gene_df['fdr_p']<0.05).sum() > 0:
    print(f"  FDR<0.05: {(gene_df['fdr_p']<0.05).sum()} genes")
print(f"\nResult files:")
print(f"  E1_region_vulnerability.csv")
print(f"  E2_gene_vulnerability_correlations.csv")
print(f"  E3_GO_BP_enrichment.csv (if successful)")
print(f"  E4_KEGG_enrichment.csv (if successful)")
