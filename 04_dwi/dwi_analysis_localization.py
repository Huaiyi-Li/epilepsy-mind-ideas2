"""
DTI white matter connectome analysis for epileptogenic zone localization.
- Weighted networks (streamline counts, not binarized)
- Distance-transformed betweenness/closeness
- Node-level local efficiency (manual computation, not nx.local_efficiency scalar)
- GroupKFold by subject (prevents data leakage)
- Correct Cohen's d with sample weighting
- 10% resection threshold
- Within-subject paired test (primary) + node-level t-test (descriptive)
- Bootstrap 95% CI
"""
import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

DWI_BASE = r'D:\癫痫MIND\fully_processed_dwi_connectomes\networks\deterministic\deterministic_tractography\Lausanne-36'
CLINICAL_FILE = r'D:\癫痫MIND\processed\clinical_aligned.csv'
RESECTION_FILE = r'D:\癫痫MIND\resection_percentage_table\table_resected.csv'
OUTPUT_DIR = r'D:\癫痫MIND\results_final'
os.makedirs(OUTPUT_DIR, exist_ok=True)

THRESHOLD = 0.10  # Regions with >10% resection overlap are defined as resected

print("=" * 60)
print("DTI ANALYSIS - 10% RESECTION THRESHOLD")
print("=" * 60)

# Load clinical
clinical = pd.read_csv(CLINICAL_FILE)
clinical['ID'] = clinical['ID'].astype(int)

# Load resection
resection = pd.read_csv(RESECTION_FILE, index_col=0)
resection.columns = [c.strip().strip("'") for c in resection.columns]
for col in resection.columns:
    resection[col] = pd.to_numeric(resection[col], errors='coerce')

# Load DWI matrices
print("\nLoading DWI connectomes...")
dwi_matrices = {}
dwi_subjects = []
for sub_dir in sorted(os.listdir(DWI_BASE)):
    if not sub_dir.startswith('sub-'):
        continue
    sub_id = int(sub_dir.replace('sub-', ''))
    ses_dir = os.path.join(DWI_BASE, sub_dir, 'ses-1', 'dwi')
    if not os.path.exists(ses_dir):
        continue
    count_file = None
    for f in os.listdir(ses_dir):
        if 'Count.csv' in f and 'CountScaled' not in f:
            count_file = os.path.join(ses_dir, f)
            break
    if count_file is None:
        continue
    try:
        df = pd.read_csv(count_file, index_col=0)
        mat = df.values.astype(float)
        n = min(mat.shape)
        mat = mat[:n, :n]
        mat = (mat + mat.T) / 2  # Symmetrize
        np.fill_diagonal(mat, 0)
        dwi_matrices[sub_id] = mat
        dwi_subjects.append(sub_id)
    except Exception as e:
        print(f"  Failed sub-{sub_id}: {e}")

print(f"Loaded {len(dwi_matrices)} DWI subjects")
n_reg = list(dwi_matrices.values())[0].shape[0]
print(f"Regions per matrix: {n_reg}")

# Align subjects
common_subjects = [
    s for s in dwi_subjects
    if s in clinical['ID'].values and str(s) in resection.columns
]
print(f"Common subjects (DWI+clinical+resection): {len(common_subjects)}")


def compute_dwi_features(mat):
    """Compute 7 nodal features from weighted DTI connectome."""
    n = mat.shape[0]
    feat = {}

    # Degree and strength (weighted)
    feat['degree'] = (mat > 0).sum(axis=1)
    feat['strength'] = mat.sum(axis=1)

    # Distance-transformed for betweenness/closeness
    # For DTI: higher streamline count = stronger connection = shorter distance
    max_val = mat.max() if mat.max() > 0 else 1
    dist = max_val - mat
    np.fill_diagonal(dist, 0)
    G_dist = nx.from_numpy_array(dist)

    bc = nx.betweenness_centrality(G_dist, weight='weight', normalized=True)
    feat['betweenness'] = [bc[i] for i in range(n)]

    cc = nx.closeness_centrality(G_dist, distance='weight')
    feat['closeness'] = [cc[i] for i in range(n)]

    # Similarity-weighted for eigenvector and clustering
    G_sim = nx.from_numpy_array(mat)
    try:
        ec = nx.eigenvector_centrality_numpy(G_sim, weight='weight')
        feat['eigenvector'] = [ec[i] for i in range(n)]
    except Exception:
        feat['eigenvector'] = [0] * n

    clust = nx.clustering(G_sim, weight='weight')
    feat['clustering'] = [clust[i] for i in range(n)]

    # Node-level local efficiency (manual: global efficiency of neighbor subgraph)
    # Note: nx.local_efficiency(G) returns a scalar (graph average), not node-level
    local_eff = np.zeros(n)
    for v in range(n):
        neighbors = list(G_sim.neighbors(v))
        if len(neighbors) > 1:
            subg = G_sim.subgraph(neighbors)
            try:
                local_eff[v] = nx.global_efficiency(subg)
            except Exception:
                local_eff[v] = 0
    feat['local_efficiency'] = local_eff

    return pd.DataFrame(feat)


feat_names = [
    'degree', 'strength', 'betweenness', 'closeness',
    'eigenvector', 'clustering', 'local_efficiency'
]

print("\nComputing nodal features...")
all_feats = []
for sub_id in common_subjects:
    mat = dwi_matrices[sub_id]
    n = mat.shape[0]
    nf = compute_dwi_features(mat)
    nf['subject_id'] = sub_id
    nf['region_idx'] = range(n)

    # Resection label (10% threshold)
    resection_col = resection[str(sub_id)].values
    n_min = min(n, len(resection_col))
    is_resected = np.zeros(n)
    is_resected[:n_min] = (resection_col[:n_min] > THRESHOLD).astype(int)
    nf['resected'] = is_resected
    nf['resect_pct'] = np.concatenate([resection_col[:n_min], np.zeros(n - n_min)])

    all_feats.append(nf)

node_df = pd.concat(all_feats, ignore_index=True)
n_resected = (node_df['resected'] == 1).sum()
n_total = len(node_df)
print(f"Total nodes: {n_total}")
print(f"Resected (>10%): {n_resected} ({n_resected / n_total * 100:.1f}%)")
print(f"Non-resected: {n_total - n_resected}")

node_df.to_csv(os.path.join(OUTPUT_DIR, 'dti_node_features.csv'), index=False)

# ========== STATISTICS ==========
print("\n" + "=" * 60)
print("STATISTICS")
print("=" * 60)

# Within-subject paired test (primary)
print("\n--- Within-subject paired test ---")
paired_results = []
for f in feat_names:
    diffs = []
    for subj in node_df['subject_id'].unique():
        subj_data = node_df[node_df['subject_id'] == subj]
        r = subj_data[subj_data['resected'] == 1][f]
        nr = subj_data[subj_data['resected'] == 0][f]
        if len(r) > 0 and len(nr) > 0:
            diffs.append(r.mean() - nr.mean())
    diffs = np.array(diffs)
    if len(diffs) > 1:
        t, p = stats.ttest_1samp(diffs, 0)
        d = diffs.mean() / diffs.std() if diffs.std() > 0 else 0
        paired_results.append({
            'feature': f, 'mean_diff': diffs.mean(),
            'cohens_d': d, 't': t, 'p_value': p
        })

paired_df = pd.DataFrame(paired_results).sort_values('p_value')
_, paired_df['p_adj'], _, _ = multipletests(paired_df['p_value'], method='fdr_bh')
print(paired_df[['feature', 'cohens_d', 'p_value', 'p_adj']].to_string(index=False))
paired_df.to_csv(os.path.join(OUTPUT_DIR, 'dti_paired_stats.csv'), index=False)

# Node-level t-test (descriptive)
print("\n--- Node-level t-test ---")
comp = []
for f in feat_names:
    r = node_df[node_df['resected'] == 1][f].dropna()
    nr = node_df[node_df['resected'] == 0][f].dropna()
    if len(r) > 1 and len(nr) > 1:
        t, p = stats.ttest_ind(r, nr)
        n1, n2 = len(r), len(nr)
        s1, s2 = r.std(ddof=1), nr.std(ddof=1)
        pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
        d = (r.mean() - nr.mean()) / pooled if pooled > 0 else 0
        comp.append({
            'feature': f, 'resected_mean': r.mean(),
            'non_resected_mean': nr.mean(), 't': t, 'p_value': p, 'cohens_d': d
        })
comp_df = pd.DataFrame(comp).sort_values('p_value')
_, comp_df['p_adj'], _, _ = multipletests(comp_df['p_value'], method='fdr_bh')
print(comp_df[['feature', 'cohens_d', 'p_value', 'p_adj']].to_string(index=False))
comp_df.to_csv(os.path.join(OUTPUT_DIR, 'dti_node_stats.csv'), index=False)

# ========== CLASSIFICATION ==========
print("\n" + "=" * 60)
print("CLASSIFICATION (GroupKFold by subject)")
print("=" * 60)

X = node_df[feat_names].values
y = node_df['resected'].values
groups = node_df['subject_id'].values

gkf = GroupKFold(n_splits=5)
y_prob = np.zeros(len(y))
for tr, te in gkf.split(X, y, groups):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[tr])
    Xte = sc.transform(X[te])
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=2,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    clf.fit(Xtr, y[tr])
    y_prob[te] = clf.predict_proba(Xte)[:, 1]

y_pred = (y_prob > 0.5).astype(int)
auc = roc_auc_score(y, y_prob)
acc = accuracy_score(y, y_pred)
cm = confusion_matrix(y, y_pred)
sens = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
spec = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0

# Bootstrap CI
np.random.seed(42)
auc_boots = []
for _ in range(1000):
    idx = np.random.choice(len(y), len(y), replace=True)
    try:
        auc_boots.append(roc_auc_score(y[idx], y_prob[idx]))
    except Exception:
        pass
ci_low, ci_high = np.percentile(auc_boots, [2.5, 97.5])

print(f"AUC = {auc:.3f} (95% CI: {ci_low:.3f}-{ci_high:.3f})")
print(f"Accuracy = {acc:.3f}, Sensitivity = {sens:.3f}, Specificity = {spec:.3f}")

pd.DataFrame({
    'metric': ['AUC', 'AUC_CI_low', 'AUC_CI_high', 'Accuracy', 'Sensitivity', 'Specificity'],
    'value': [auc, ci_low, ci_high, acc, sens, spec]
}).to_csv(os.path.join(OUTPUT_DIR, 'dti_classifier_results.csv'), index=False)

# Feature importance
sc = StandardScaler()
Xs = sc.fit_transform(X)
clf_full = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_split=2,
    class_weight='balanced', random_state=42, n_jobs=-1
)
clf_full.fit(Xs, y)
imp = pd.DataFrame({'feature': feat_names, 'importance': clf_full.feature_importances_})
imp = imp.sort_values('importance', ascending=False)
print("\nFeature importance:")
print(imp.to_string(index=False))
imp.to_csv(os.path.join(OUTPUT_DIR, 'dti_feature_importance.csv'), index=False)

# Patient-level
print("\n--- Patient-level ---")
patient_correct = 0
patient_total = 0
for subj in node_df['subject_id'].unique():
    subj_data = node_df[node_df['subject_id'] == subj]
    if (subj_data['resected'] == 1).sum() > 0:
        patient_total += 1
        r_prob = y_prob[subj_data.index[subj_data['resected'] == 1]].mean()
        nr_prob = y_prob[subj_data.index[subj_data['resected'] == 0]].mean()
        if r_prob > nr_prob:
            patient_correct += 1
print(f"Patient-level: {patient_correct}/{patient_total} = {patient_correct / patient_total * 100:.1f}%")

print("\n" + "=" * 60)
print("DTI ANALYSIS COMPLETE")
print("=" * 60)
