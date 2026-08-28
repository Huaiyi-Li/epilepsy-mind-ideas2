# -*- coding: utf-8 -*-
"""
DTI subsample baseline comparison: 87 DTI-included vs 355 excluded subjects
"""
import pandas as pd
import numpy as np
import os
from scipy import stats

base_dir = r"D:\epilepsy_MIND"
proc_dir = os.path.join(base_dir, "processed")
result_dir = os.path.join(base_dir, "results_10pct")

# Load clinical data
clin = pd.read_csv(os.path.join(base_dir, "processed", "clinical_aligned.csv"), index_col=0)
clin = clin[~clin.index.duplicated(keep='first')]

# Load DTI nodal features to get DTI-included subject IDs
dti_node = pd.read_csv(os.path.join(base_dir, "results_10pct", "dti_node_features.csv"))
dti_subjects = dti_node['subject_id'].unique()
print(f"DTI included: {len(dti_subjects)} subjects")

# All subjects
all_subjects = clin.index.tolist()
excluded_subjects = [s for s in all_subjects if s not in dti_subjects]
print(f"DTI excluded: {len(excluded_subjects)} subjects")

# Baseline comparison
print("\n" + "=" * 60)
print("DTI included vs excluded baseline comparison")
print("=" * 60)

# Categorical variables: chi-square test
def chi2_test(col):
    table = pd.crosstab(clin[col], clin.index.isin(dti_subjects))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return p

# Continuous variables: t-test
def t_test(col):
    g1 = clin.loc[clin.index.isin(dti_subjects), col].dropna()
    g2 = clin.loc[~clin.index.isin(dti_subjects), col].dropna()
    if len(g1) < 2 or len(g2) < 2:
        return np.nan, np.nan, np.nan
    t, p = stats.ttest_ind(g1, g2)
    return g1.mean(), g2.mean(), p

results = []

# Sex
p_sex = chi2_test('Sex')
results.append({'Variable': 'Sex (female %)', 
                'DTI_included': f"{(clin.loc[clin.index.isin(dti_subjects), 'Sex']=='F').mean()*100:.1f}%",
                'Excluded': f"{(clin.loc[~clin.index.isin(dti_subjects), 'Sex']=='F').mean()*100:.1f}%",
                'p_value': p_sex})

# Operation side
p_side = chi2_test('Op_Side')
results.append({'Variable': 'Operation side (left %)',
                'DTI_included': f"{(clin.loc[clin.index.isin(dti_subjects), 'Op_Side']=='L').mean()*100:.1f}%",
                'Excluded': f"{(clin.loc[~clin.index.isin(dti_subjects), 'Op_Side']=='L').mean()*100:.1f}%",
                'p_value': p_side})

# Pathology type
p_path = chi2_test('Pathology')
results.append({'Variable': 'Pathology (HS %)',
                'DTI_included': f"{(clin.loc[clin.index.isin(dti_subjects), 'Pathology']=='HS').mean()*100:.1f}%",
                'Excluded': f"{(clin.loc[~clin.index.isin(dti_subjects), 'Pathology']=='HS').mean()*100:.1f}%",
                'p_value': p_path})

# Pathology distribution details
print("\nPathology distribution:")
print("DTI included:")
print(clin.loc[clin.index.isin(dti_subjects), 'Pathology'].value_counts())
print("\nExcluded:")
print(clin.loc[~clin.index.isin(dti_subjects), 'Pathology'].value_counts())

# Outcome ILAE Year1
if 'ILAE_Year1' in clin.columns:
    p_ilae1 = chi2_test('ILAE_Year1')
    included_seizure_free = (clin.loc[clin.index.isin(dti_subjects), 'ILAE_Year1']==1).mean()*100
    excluded_seizure_free = (clin.loc[~clin.index.isin(dti_subjects), 'ILAE_Year1']==1).mean()*100
    results.append({'Variable': 'Seizure-free at 1y (ILAE=1 %)',
                    'DTI_included': f"{included_seizure_free:.1f}%",
                    'Excluded': f"{excluded_seizure_free:.1f}%",
                    'p_value': p_ilae1})

# Onset age
if 'Binned_Onset_Age' in clin.columns:
    p_onset = chi2_test('Binned_Onset_Age')
    results.append({'Variable': 'Onset age distribution',
                    'DTI_included': '-', 'Excluded': '-', 'p_value': p_onset})

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))
results_df.to_csv(os.path.join(base_dir, "results_10pct", "dti_baseline_comparison.csv"), index=False)
print("\nResults saved.")
