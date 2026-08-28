# -*- coding: utf-8 -*-
"""
HS subgroup transcriptomic spatial association analysis
Compares with full-cohort results to assess the impact of surgical access contamination
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
result_dir = os.path.join(base_dir, "results_HS")
os.makedirs(result_dir, exist_ok=True)

# ========== Step 1: Load AHBA expression matrix ==========
print("=" * 60)
print("Step 1: Loading AHBA Desikan gene expression matrix")
print("=" * 60)

expr_df = pd.read_csv(os.path.join(ahba_dir, "AllenHBA_DK_ExpressionMatrix.tsv"), sep='\t')
gene_names = expr_df.iloc[:, 0].values
region_cols = [c for c in expr_df.columns if c.startswith('ctx-')]
expression = expr_df[region_cols].values

def map_region_name(name):
    if name.startswith('ctx-lh-'):
        return 'lh_' + name[7:]
    elif name.startswith('ctx-rh-'):
        return 'rh_' + name[7:]
    return name

ahba_regions = [map_region_name(c) for c in region_cols]

# ========== Step 2: Load data and filter HS subgroup ==========
print("\n" + "=" * 60)
print("Step 2: Loading data, filtering HS subgroup")
print("=" * 60)

thick = pd.read_csv(os.path.join(proc_dir, "thickness_aligned.csv"), index_col=0)
vol = pd.read_csv(os.path.join(proc_dir, "volume_aligned.csv"), index_col=0)
area = pd.read_csv(os.path.join(proc_dir, "area_aligned.csv"), index_col=0)
clin = pd.read_csv(os.path.join(proc_dir, "clinical_aligned.csv"), index_col=0)
resect = pd.read_csv(os.path.join(proc_dir, "resection_aligned.csv"), index_col=0)
with open(os.path.join(proc_dir, "region_names.txt")) as f:
    region_names = [line.strip() for line in f]

subject_ids = thick.index.tolist()
clin = clin[~clin.index.duplicated(keep='first')].loc[subject_ids]

# Filter HS subgroup
hs_mask = clin['Pathology'].values == 'HS'
hs_subjects = [s for s, m in zip(subject_ids, hs_mask) if m]
print(f"Full cohort: {len(subject_ids)} subjects")
print(f"HS subgroup: {len(hs_subjects)} subjects")

# Region name mapping
def map_resect_region(name):
    if name.startswith('ctx-lh-'): return 'lh_' + name[7:]
    if name.startswith('ctx-rh-'): return 'rh_' + name[7:]
    return None
resect_map = {c: map_resect_region(c) for c in resect.columns if map_resect_region(c) and map_resect_region(c) in region_names}
resect_cortex = resect[list(resect_map.keys())].rename(columns=resect_map)[region_names]

# Load MIND networks
mind_nets = np.load(os.path.join(proc_dir, "mind_networks.npy"))

# Keep only HS subgroup
hs_idx = [subject_ids.index(s) for s in hs_subjects]
mind_nets_hs = mind_nets[hs_idx]
resect_cortex_hs = resect_cortex.loc[hs_subjects]

# ========== Step 3: Compute HS subgroup regional epilepsy vulnerability score ==========
print("\n" + "=" * 60)
print("Step 3: Computing HS subgroup regional epilepsy vulnerability score")
print("=" * 60)

# Resection frequency (HS subgroup)
resection_freq_hs = (resect_cortex_hs > 0).mean().values

# Compute total connection strength effect size for each region (HS subgroup)
total_strength_hs = np.abs(mind_nets_hs).sum(axis=2)
from scipy import stats
effect_sizes_hs = []
for r in range(len(region_names)):
    resected_mask = resect_cortex_hs.iloc[:, r].values > 0
    if resected_mask.sum() < 5 or (~resected_mask).sum() < 5:
        effect_sizes_hs.append(0); continue
    g1 = total_strength_hs[resected_mask, r]
    g2 = total_strength_hs[~resected_mask, r]
    n1, n2 = len(g1), len(g2)
    s1, s2 = g1.std(ddof=1), g2.std(ddof=1)
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    effect_sizes_hs.append((g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0)
effect_sizes_hs = np.array(effect_sizes_hs)

# Epilepsy vulnerability score = resection frequency x |effect size|
vulnerability_score_hs = resection_freq_hs * np.abs(effect_sizes_hs)
vulnerability_score_hs = (vulnerability_score_hs - vulnerability_score_hs.min()) / (vulnerability_score_hs.max() - vulnerability_score_hs.min())

# Save regional table
region_df_hs = pd.DataFrame({
    'region': region_names,
    'resection_freq_hs': resection_freq_hs,
    'effect_size_d_hs': effect_sizes_hs,
    'vulnerability_score_hs': vulnerability_score_hs
}).sort_values('vulnerability_score_hs', ascending=False)
region_df_hs.to_csv(os.path.join(result_dir, "E1_region_vulnerability_HS.csv"), index=False)

print(f"HS subgroup Top 10 vulnerable regions:")
print(region_df_hs.head(10).to_string(index=False))

# Compare with full cohort
region_df_full = pd.read_csv(os.path.join(base_dir, "results", "E1_region_vulnerability.csv"))
merged = region_df_full.merge(region_df_hs, on='region', suffixes=('_full', '_hs'))
# Full cohort columns are vulnerability_score, resection_freq, effect_size_d
corr_vuln = np.corrcoef(merged['vulnerability_score'], merged['vulnerability_score_hs'])[0,1]
corr_freq = np.corrcoef(merged['resection_freq'], merged['resection_freq_hs'])[0,1]
print(f"\nFull cohort vs HS subgroup vulnerability score correlation: r={corr_vuln:.3f}")
print(f"Full cohort vs HS subgroup resection frequency correlation: r={corr_freq:.3f}")

# ========== Step 4: Align AHBA regions with MIND regions ==========
print("\n" + "=" * 60)
print("Step 4: Aligning AHBA and MIND regions")
print("=" * 60)

common_regions = [r for r in ahba_regions if r in region_names]
ahba_idx = [ahba_regions.index(r) for r in common_regions]
mind_idx = [region_names.index(r) for r in common_regions]

expr_aligned = expression[:, ahba_idx]
vuln_aligned_hs = vulnerability_score_hs[mind_idx]

# ========== Step 5: Gene-vulnerability spatial association analysis (HS subgroup) ==========
print("\n" + "=" * 60)
print("Step 5: HS subgroup gene expression vs epilepsy vulnerability spatial association")
print("=" * 60)

from scipy.stats import pearsonr

gene_results_hs = []
n_genes = len(gene_names)
for g in range(n_genes):
    if g % 3000 == 0:
        print(f"  Processing gene {g}/{n_genes}...")
    expr_vals = expr_aligned[g, :]
    valid = ~np.isnan(expr_vals)
    if valid.sum() < 30:
        gene_results_hs.append({'gene': gene_names[g], 'r': 0, 'p': 1, 'n_valid': valid.sum()})
        continue
    r, p = pearsonr(expr_vals[valid], vuln_aligned_hs[valid])
    gene_results_hs.append({'gene': gene_names[g], 'r': r, 'p': p, 'n_valid': valid.sum()})

gene_df_hs = pd.DataFrame(gene_results_hs)

from statsmodels.stats.multitest import multipletests
valid_p = gene_df_hs['p'].values
_, fdr_p, _, _ = multipletests(valid_p, method='fdr_bh')
gene_df_hs['fdr_p'] = fdr_p
gene_df_hs = gene_df_hs.sort_values('p')

print(f"\nHS subgroup gene association analysis complete!")
print(f"Total genes: {len(gene_df_hs)}")
print(f"P<0.05: {(gene_df_hs['p']<0.05).sum()}")
print(f"P<0.01: {(gene_df_hs['p']<0.01).sum()}")
print(f"FDR<0.05: {(gene_df_hs['fdr_p']<0.05).sum()}")
print(f"FDR<0.1: {(gene_df_hs['fdr_p']<0.1).sum()}")

gene_df_hs.to_csv(os.path.join(result_dir, "E2_gene_vulnerability_correlations_HS.csv"), index=False)

# Compare with full cohort gene results
gene_df_full = pd.read_csv(os.path.join(base_dir, "results", "E2_gene_vulnerability_correlations.csv"))
merged_genes = gene_df_full.merge(gene_df_hs, on='gene', suffixes=('_full', '_hs'))
corr_r = np.corrcoef(merged_genes['r_full'], merged_genes['r_hs'])[0,1]
print(f"\nFull cohort vs HS subgroup gene correlation coefficient r correlation: {corr_r:.3f}")

# Top gene overlap
top_full_pos = set(gene_df_full[gene_df_full['r']>0].sort_values('p').head(50)['gene'])
top_hs_pos = set(gene_df_hs[gene_df_hs['r']>0].sort_values('p').head(50)['gene'])
overlap_pos = top_full_pos & top_hs_pos
print(f"Top50 positively correlated genes overlap: {len(overlap_pos)}/50")

top_full_neg = set(gene_df_full[gene_df_full['r']<0].sort_values('p').head(50)['gene'])
top_hs_neg = set(gene_df_hs[gene_df_hs['r']<0].sort_values('p').head(50)['gene'])
overlap_neg = top_full_neg & top_hs_neg
print(f"Top50 negatively correlated genes overlap: {len(overlap_neg)}/50")

# ========== Step 6: GO/KEGG enrichment analysis (HS subgroup) ==========
print("\n" + "=" * 60)
print("Step 6: HS subgroup GO/KEGG enrichment analysis")
print("=" * 60)

try:
    import gseapy as gp
    
    sig_genes_hs = gene_df_hs[gene_df_hs['p']<0.01]['gene'].tolist()
    print(f"HS subgroup candidate genes (P<0.01): {len(sig_genes_hs)}")
    
    if len(sig_genes_hs) > 20:
        # GO BP
        print("\nGO BP enrichment analysis...")
        go_enr = gp.enrichr(gene_list=sig_genes_hs, gene_sets='GO_Biological_Process_2023',
                             organism='human', outdir=None, cutoff=0.1)
        go_res = go_enr.results.sort_values('Adjusted P-value')
        go_sig = go_res[go_res['Adjusted P-value']<0.1]
        print(f"GO BP significant terms (FDR<0.1): {len(go_sig)}")
        if len(go_sig) > 0:
            print(go_sig[['Term','Overlap','Adjusted P-value']].head(10).to_string(index=False))
            go_sig.to_csv(os.path.join(result_dir, "E3_GO_BP_enrichment_HS.csv"), index=False)
        
        # KEGG
        print("\nKEGG pathway enrichment analysis...")
        kegg_enr = gp.enrichr(gene_list=sig_genes_hs, gene_sets='KEGG_2021_Human',
                               organism='human', outdir=None, cutoff=0.1)
        kegg_res = kegg_enr.results.sort_values('Adjusted P-value')
        kegg_sig = kegg_res[kegg_res['Adjusted P-value']<0.1]
        print(f"KEGG significant terms (FDR<0.1): {len(kegg_sig)}")
        if len(kegg_sig) > 0:
            print(kegg_sig[['Term','Overlap','Adjusted P-value']].head(10).to_string(index=False))
            kegg_sig.to_csv(os.path.join(result_dir, "E4_KEGG_enrichment_HS.csv"), index=False)
except Exception as e:
    print(f"Enrichment analysis error: {e}")

print("\n" + "=" * 60)
print("HS subgroup transcriptomic analysis complete!")
print("=" * 60)
