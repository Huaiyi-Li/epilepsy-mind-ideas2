"""
Within-subject paired analysis for nodal features.
For each patient: mean(feature in resected) - mean(feature in non-resected)
Then one-sample t-test against zero across patients.
"""
import numpy as np
import pandas as pd
from scipy import stats
import os

BASE = r'D:\癫痫MIND'
node_df = pd.read_csv(os.path.join(BASE, 'results_final', 'node_features.csv'))

feat_cols = ['pos_degree','neg_degree','pos_strength','neg_strength','total_strength',
             'betweenness','closeness','eigenvector','clustering','participation',
             'within_module_strength','left_hemi_strength','right_hemi_strength','inter_hemi_strength']

print(f"Total nodes: {len(node_df)}")
print(f"Total subjects: {node_df['subject_id'].nunique()}")
print(f"Resected nodes: {(node_df['resected']==1).sum()}")
print(f"Non-resected nodes: {(node_df['resected']==0).sum()}")

# Within-subject paired analysis
results = []
for f in feat_cols:
    # For each subject, compute mean in resected vs non-resected
    sub_means = node_df.groupby(['subject_id','resected'])[f].mean().unstack()
    sub_means.columns = ['non_resected', 'resected']  # 0, 1
    sub_means = sub_means.dropna()
    # Only keep subjects with both resected and non-resected
    diff = sub_means['resected'] - sub_means['non_resected']
    if len(diff) > 1 and diff.std(ddof=1) > 0:
        t, p = stats.ttest_1samp(diff, 0)
        d = diff.mean() / diff.std(ddof=1)  # Cohen's d for one-sample
        results.append({
            'feature': f,
            'n_subjects': len(diff),
            'mean_resected': sub_means['resected'].mean(),
            'mean_nonresected': sub_means['non_resected'].mean(),
            'mean_diff': diff.mean(),
            'cohens_d': d,
            't_stat': t,
            'p_value': p
        })

res_df = pd.DataFrame(results).sort_values('p_value')
print("\n=== Within-subject paired analysis (patient-level) ===")
print(res_df[['feature','n_subjects','mean_diff','cohens_d','t_stat','p_value']].to_string(index=False))

# FDR correction
from statsmodels.stats.multitest import multipletests
res_df['p_adj_fdr'] = multipletests(res_df['p_value'].values, method='fdr_bh')[1]
print("\n=== With FDR correction ===")
print(res_df[['feature','cohens_d','p_value','p_adj_fdr']].to_string(index=False))

res_df.to_csv(os.path.join(BASE, 'results_final', 'paired_nodal_comparison.csv'), index=False)
print("\nSaved to paired_nodal_comparison.csv")
