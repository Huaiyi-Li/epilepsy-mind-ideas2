# -*- coding: utf-8 -*-
"""
Direction A in-depth analysis: Epileptogenic Zone Localization

Pipeline:
1. Compute 14 nodal topological features per subject per region
2. Statistical comparison: resected vs non-resected (node-level + within-subject paired)
3. Multi-classifier comparison with GroupKFold (patient-level grouping)
4. Stratified analysis by pathology type
5. Feature importance analysis
6. Region-level aggregation

Key methodological notes:
- Betweenness/closeness centrality use DISTANCE matrix (1 - |similarity|),
  NOT raw similarity weights, because NetworkX interprets weight as cost.
- Eigenvector centrality correctly uses similarity weights (higher = more connected).
- Cohen's d uses sample-size-weighted pooled SD (not equal-variance formula).
- Local efficiency is excluded: nx.local_efficiency() returns a graph-level scalar,
  not node-level dict; its feature importance was 0 in all analyses.
"""
import pandas as pd
import numpy as np
import os
import networkx as nx
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ========== Configuration ==========
# Set base_dir to your data root (contains processed/ subdirectory)
base_dir = r"D:\癫痫MIND"
proc_dir = os.path.join(base_dir, "processed")
result_dir = os.path.join(base_dir, "results")
os.makedirs(result_dir, exist_ok=True)

RANDOM_STATE = 42

# ========== Load data ==========
print("=" * 60)
print("Loading data")
mind_nets = np.load(os.path.join(proc_dir, "mind_networks.npy"))
thick_aligned = pd.read_csv(os.path.join(proc_dir, "thickness_aligned.csv"), index_col=0)
clin = pd.read_csv(os.path.join(proc_dir, "clinical_aligned.csv"), index_col=0)
resect = pd.read_csv(os.path.join(proc_dir, "resection_aligned.csv"), index_col=0)
with open(os.path.join(proc_dir, "region_names.txt")) as f:
    region_names = [line.strip() for line in f]

subject_ids = thick_aligned.index.tolist()
# Deduplicate clinical data by ID, keep first
clin = clin[~clin.index.duplicated(keep='first')].loc[subject_ids]
resect = resect.loc[subject_ids]
print(f"  subjects: {len(subject_ids)}, regions: {len(region_names)}")

# Data consistency assertions
assert mind_nets.shape[1] == len(region_names), "MIND matrix region count mismatch"
assert mind_nets.shape[0] == len(subject_ids), "MIND matrix subject count mismatch"
assert len(region_names) == 68, f"Expected 68 regions, got {len(region_names)}"
half = len(region_names) // 2
assert all(r.startswith('lh_') for r in region_names[:half]), "First 34 regions must be left hemisphere"
assert all(r.startswith('rh_') for r in region_names[half:]), "Last 34 regions must be right hemisphere"

# Map resection columns (ctx-lh-xxx) to region names (lh_xxx)
def map_region(name):
    if name.startswith('ctx-lh-'):
        return 'lh_' + name[7:]
    if name.startswith('ctx-rh-'):
        return 'rh_' + name[7:]
    return None

resect_map = {c: map_region(c) for c in resect.columns
              if map_region(c) and map_region(c) in region_names}
resect_cortex = resect[list(resect_map.keys())].rename(columns=resect_map)[region_names]

# ========== Compute nodal features ==========
print("\n" + "=" * 60)
print("Computing nodal topological features")

def compute_node_features(network):
    """
    Compute 14 nodal features for one subject's MIND network.

    Parameters
    ----------
    network : np.ndarray (68x68)
        Symmetric similarity matrix (Pearson correlation of morphological features).

    Returns
    -------
    pd.DataFrame with 14 feature columns.
    """
    n = network.shape[0]
    half = n // 2
    feat = {}

    # --- Degree and strength (signed) ---
    pos_net = (network > 0).astype(float)
    neg_net = (network < 0).astype(float)
    feat['pos_degree'] = pos_net.sum(axis=1)
    feat['neg_degree'] = neg_net.sum(axis=1)
    feat['pos_strength'] = np.where(network > 0, network, 0).sum(axis=1)
    feat['neg_strength'] = np.where(network < 0, -network, 0).sum(axis=1)
    feat['total_strength'] = feat['pos_strength'] + feat['neg_strength']

    # --- Centrality measures ---
    # CRITICAL: betweenness and closeness use DISTANCE matrix (1 - |similarity|)
    # because NetworkX interprets edge weight as traversal cost (lower = shorter path).
    # Using raw similarity would invert the direction of these measures.
    abs_sim = np.abs(network)
    distance = 1.0 - abs_sim
    np.fill_diagonal(distance, 0.0)
    G_dist = nx.from_numpy_array(distance)

    try:
        bc = nx.betweenness_centrality(G_dist, weight='weight')
        feat['betweenness'] = [bc[i] for i in range(n)]
    except Exception as e:
        print(f"    Warning: betweenness failed: {e}")
        feat['betweenness'] = [0.0] * n

    try:
        cc = nx.closeness_centrality(G_dist, distance='weight')
        feat['closeness'] = [cc[i] for i in range(n)]
    except Exception as e:
        print(f"    Warning: closeness failed: {e}")
        feat['closeness'] = [0.0] * n

    # Eigenvector centrality correctly uses similarity weights (higher similarity = stronger connection)
    G_sim = nx.from_numpy_array(abs_sim)
    try:
        ec = nx.eigenvector_centrality_numpy(G_sim, weight='weight')
        feat['eigenvector'] = [ec[i] for i in range(n)]
    except Exception as e:
        print(f"    Warning: eigenvector failed: {e}")
        feat['eigenvector'] = [0.0] * n

    # --- Clustering coefficient ---
    try:
        clust = nx.clustering(G_sim, weight='weight')
        feat['clustering'] = [clust[i] for i in range(n)]
    except Exception as e:
        print(f"    Warning: clustering failed: {e}")
        feat['clustering'] = [0.0] * n

    # --- Module detection + participation coefficient + within-module strength ---
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G_sim, weight='weight'))
        comm_map = {}
        for ci, comm in enumerate(comms):
            for node in comm:
                comm_map[node] = ci

        participation = []
        within_strength_list = []
        for i in range(n):
            ci = comm_map[i]
            total_deg = feat['total_strength'][i]
            if total_deg > 0:
                within_str = sum(abs_sim[i, j] for j in range(n)
                                 if comm_map.get(j) == ci and i != j)
                within_strength_list.append(within_str)
                mod_strengths = {}
                for j in range(n):
                    if i != j:
                        cj = comm_map.get(j, -1)
                        mod_strengths[cj] = mod_strengths.get(cj, 0) + abs_sim[i, j]
                p = 1 - sum((s / total_deg) ** 2 for s in mod_strengths.values())
                participation.append(p)
            else:
                within_strength_list.append(0.0)
                participation.append(0.0)
        feat['participation'] = participation
        feat['within_module_strength'] = within_strength_list
    except Exception as e:
        print(f"    Warning: module detection failed: {e}")
        feat['participation'] = [0.0] * n
        feat['within_module_strength'] = [0.0] * n

    # --- Intra/inter-hemispheric connection strength ---
    left_str = []
    right_str = []
    inter_str = []
    for i in range(n):
        left_conn = abs_sim[i, :half]
        right_conn = abs_sim[i, half:]
        left_str.append(np.mean(left_conn[left_conn != 0]) if np.any(left_conn != 0) else 0.0)
        right_str.append(np.mean(right_conn[right_conn != 0]) if np.any(right_conn != 0) else 0.0)
        if i < half:
            inter_str.append(right_str[-1])
        else:
            inter_str.append(left_str[-1])
    feat['left_hemi_strength'] = left_str
    feat['right_hemi_strength'] = right_str
    feat['inter_hemi_strength'] = inter_str

    return pd.DataFrame(feat)

print("  Computing nodal features for all subjects...")
all_feats = []
for i in range(mind_nets.shape[0]):
    nf = compute_node_features(mind_nets[i])
    nf['subject_id'] = subject_ids[i]
    nf['region'] = region_names
    subj_resect = resect_cortex.iloc[i].values
    nf['resected'] = (subj_resect > 0).astype(int)
    nf['resect_pct'] = subj_resect
    nf['pathology'] = clin.iloc[i]['Pathology']
    nf['op_side'] = clin.iloc[i]['Op_Side']
    all_feats.append(nf)

node_df = pd.concat(all_feats, ignore_index=True)
feat_cols = [c for c in node_df.columns if c not in
             ['subject_id', 'region', 'resected', 'resect_pct', 'pathology', 'op_side']]
print(f"  Nodal feature matrix: {node_df.shape}")
print(f"  Number of features: {len(feat_cols)}")
print(f"  Feature list: {feat_cols}")

# ========== Statistical comparison ==========
print("\n" + "=" * 60)
print("Node-level comparison (resected vs non-resected)")
print("  Cohen's d uses sample-size-weighted pooled SD")

def cohens_d_weighted(group1, group2):
    """Compute Cohen's d with sample-size-weighted pooled SD."""
    n1, n2 = len(group1), len(group2)
    s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_sd

comp = []
for f in feat_cols:
    r = node_df[node_df['resected'] == 1][f].dropna()
    nr = node_df[node_df['resected'] == 0][f].dropna()
    if len(r) > 1 and len(nr) > 1 and r.std() > 0 and nr.std() > 0:
        t, p = stats.ttest_ind(r, nr)
        d = cohens_d_weighted(r, nr)
        comp.append({'feature': f, 'resected_mean': r.mean(),
                     'non_resected_mean': nr.mean(), 't': t,
                     'p_value': p, 'cohens_d': d})

comp_df = pd.DataFrame(comp).sort_values('p_value')
# Bonferroni correction
comp_df['p_bonferroni'] = np.minimum(comp_df['p_value'] * len(comp_df), 1.0)
print(comp_df.to_string(index=False))
comp_df.to_csv(os.path.join(result_dir, "A2_rich_feature_comparison.csv"), index=False)

# ========== Within-subject paired test (addresses non-independence) ==========
print("\n" + "=" * 60)
print("Within-subject paired test (patient-level N, addresses non-independence)")
paired_results = []
for f in feat_cols:
    diffs = []
    for subj, subj_df in node_df.groupby('subject_id'):
        r = subj_df[subj_df['resected'] == 1][f].dropna()
        nr = subj_df[subj_df['resected'] == 0][f].dropna()
        if len(r) > 0 and len(nr) > 0:
            diffs.append(r.mean() - nr.mean())
    diffs = np.array(diffs)
    if len(diffs) > 10 and np.std(diffs) > 0:
        t, p = stats.ttest_1samp(diffs, 0)
        d = np.mean(diffs) / np.std(diffs, ddof=1)
        paired_results.append({'feature': f, 'n_patients': len(diffs),
                               'mean_diff': np.mean(diffs), 'cohens_d': d,
                               't_stat': t, 'p_value': p})

paired_df = pd.DataFrame(paired_results).sort_values('p_value')
print(paired_df.to_string(index=False))
paired_df.to_csv(os.path.join(result_dir, "A2_paired_test.csv"), index=False)

# ========== Multi-classifier comparison ==========
print("\n" + "=" * 60)
print("Multi-classifier comparison (GroupKFold, patient-level grouping)")
X = node_df[feat_cols].values
y = node_df['resected'].values
groups = node_df['subject_id'].values
gkf = GroupKFold(n_splits=5)

classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced',
                                               random_state=RANDOM_STATE),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, class_weight='balanced',
                     random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10,
                                             class_weight='balanced',
                                             random_state=RANDOM_STATE, n_jobs=-1),
}

results = []
for name, clf in classifiers.items():
    y_prob = np.zeros(len(y))
    y_pred = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf.fit(Xtr, y[tr])
        y_prob[te] = clf.predict_proba(Xte)[:, 1]
        y_pred[te] = clf.predict(Xte)
    auc = roc_auc_score(y, y_prob)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    sens = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    print(f"  [{name}] AUC={auc:.3f} Acc={acc:.3f} Sens={sens:.3f} Spec={spec:.3f}")
    results.append({'classifier': name, 'auc': auc, 'accuracy': acc,
                    'sensitivity': sens, 'specificity': spec})

pd.DataFrame(results).to_csv(os.path.join(result_dir, "A2_classifier_comparison.csv"), index=False)

# ========== Patient-level classification evaluation ==========
print("\n" + "=" * 60)
print("Patient-level evaluation (RF with GroupKFold)")
y_prob_all = np.zeros(len(y))
for tr, te in gkf.split(X, y, groups):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[tr])
    Xte = sc.transform(X[te])
    clf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                  class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(Xtr, y[tr])
    y_prob_all[te] = clf.predict_proba(Xte)[:, 1]

node_df['pred_prob'] = y_prob_all
patient_level = []
for subj, subj_df in node_df.groupby('subject_id'):
    r_prob = subj_df[subj_df['resected'] == 1]['pred_prob'].mean()
    nr_prob = subj_df[subj_df['resected'] == 0]['pred_prob'].mean()
    n_resected = (subj_df['resected'] == 1).sum()
    if n_resected > 0 and not np.isnan(r_prob) and not np.isnan(nr_prob):
        patient_level.append({'subject_id': subj, 'resected_prob': r_prob,
                              'non_resected_prob': nr_prob, 'diff': r_prob - nr_prob})

pat_df = pd.DataFrame(patient_level)
patient_acc = (pat_df['diff'] > 0).mean()
t_paired, p_paired = stats.ttest_rel(pat_df['resected_prob'], pat_df['non_resected_prob'])
print(f"  Patients with both regions: {len(pat_df)}")
print(f"  Patient-level accuracy (resected prob > non-resected): {patient_acc:.3f}")
print(f"  Paired t-test: t={t_paired:.2f}, p={p_paired:.2e}")
pat_df.to_csv(os.path.join(result_dir, "A2_patient_level.csv"), index=False)

# ========== Stratified by pathology ==========
print("\n" + "=" * 60)
print("Stratified analysis by pathology type")
for pathology_group, label in [('HS', 'Hippocampal sclerosis'), ('non_HS', 'non-HS (other)')]:
    if pathology_group == 'HS':
        sub_df = node_df[node_df['pathology'] == 'HS']
    else:
        sub_df = node_df[node_df['pathology'] != 'HS']

    if len(sub_df) == 0 or sub_df['resected'].nunique() < 2:
        print(f"  [{label}] insufficient data, skipping")
        continue

    X_sub = sub_df[feat_cols].values
    y_sub = sub_df['resected'].values
    g_sub = sub_df['subject_id'].values
    n_groups = len(np.unique(g_sub))
    n_splits = min(5, n_groups)

    y_prob = np.zeros(len(y_sub))
    for tr, te in GroupKFold(n_splits=n_splits).split(X_sub, y_sub, g_sub):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_sub[tr])
        Xte = sc.transform(X_sub[te])
        clf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                      class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(Xtr, y_sub[tr])
        y_prob[te] = clf.predict_proba(Xte)[:, 1]

    auc = roc_auc_score(y_sub, y_prob)
    n_resected = (y_sub == 1).sum()
    print(f"  [{label}] n={len(sub_df)} nodes, {n_groups} patients, "
          f"Resected={n_resected}, AUC={auc:.3f}")

# ========== Feature importance ==========
print("\n" + "=" * 60)
print("Feature importance (Random Forest, full sample, descriptive)")
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
rf = RandomForestClassifier(n_estimators=500, max_depth=10, class_weight='balanced',
                             random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_scaled, y)
imp = pd.DataFrame({'feature': feat_cols, 'importance': rf.feature_importances_})
imp = imp.sort_values('importance', ascending=False)
print(imp.to_string(index=False))
imp.to_csv(os.path.join(result_dir, "A2_feature_importance.csv"), index=False)

# ========== Region-level analysis ==========
print("\n" + "=" * 60)
print("Region-level aggregation")
resect_freq = node_df.groupby('region')['resected'].mean().sort_values(ascending=False)
print("\n  Most frequently resected regions (top 15):")
for reg, freq in resect_freq.head(15).items():
    print(f"    {reg}: {freq:.1%}")
resect_freq.to_csv(os.path.join(result_dir, "A2_resection_frequency_by_region.csv"))

print("\n  Effect size of total_strength per region (top 10 negative):")
region_effect = []
for reg in region_names:
    sub = node_df[node_df['region'] == reg]
    r = sub[sub['resected'] == 1]['total_strength'].dropna()
    nr = sub[sub['resected'] == 0]['total_strength'].dropna()
    if len(r) > 5 and len(nr) > 5 and r.std() > 0 and nr.std() > 0:
        d = cohens_d_weighted(r, nr)
        region_effect.append({'region': reg, 'cohens_d': d, 'n_resected': len(r)})
reg_eff_df = pd.DataFrame(region_effect).sort_values('cohens_d')
print(reg_eff_df.head(10).to_string(index=False))
reg_eff_df.to_csv(os.path.join(result_dir, "A2_region_level_effect_size.csv"), index=False)

# ========== Summary ==========
print("\n" + "=" * 60)
print("Direction A analysis complete. Results directory:", result_dir)
print("=" * 60)
print("\nGenerated files:")
for f in sorted(os.listdir(result_dir)):
    if f.startswith('A2'):
        print(f"  {f}")
