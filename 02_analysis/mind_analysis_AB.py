"""
MIND analysis with 10% resection threshold and all bug fixes.
- Betweenness/closeness on distance-transformed networks
- Correct Cohen's d with sample weighting
- Within-subject paired test (primary) + node-level t-test (descriptive)
- GroupKFold by subject for classification
- 10% resection threshold (following epilepsy surgery imaging conventions)
- Pathology and surgical side stratification
- Bootstrap confidence intervals
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

BASE = r'D:\癫痫MIND'
PROC = os.path.join(BASE, 'processed')
RESULT = os.path.join(BASE, 'results_final')
os.makedirs(RESULT, exist_ok=True)

THRESHOLD = 0.10  # Regions with >10% resection overlap are defined as resected

print("=" * 60)
print("MIND ANALYSIS - 10% RESECTION THRESHOLD")
print("=" * 60)

# Load data
mind_nets = np.load(os.path.join(PROC, 'mind_networks.npy'))
clin = pd.read_csv(os.path.join(PROC, 'clinical_aligned.csv'), index_col=0)
resect = pd.read_csv(os.path.join(PROC, 'resection_aligned.csv'), index_col=0)
with open(os.path.join(PROC, 'region_names.txt')) as f:
    region_names = [l.strip() for l in f if l.strip()]

print(f"MIND networks: {mind_nets.shape}")
print(f"Clinical: {clin.shape}")
print(f"Regions: {len(region_names)}")

# Align resection to 68 cortical regions
def map_region(name):
    if name.startswith('ctx-lh-'):
        return 'lh_' + name[7:]
    if name.startswith('ctx-rh-'):
        return 'rh_' + name[7:]
    return None

resect_map = {}
for col in resect.columns:
    m = map_region(col)
    if m and m in region_names:
        resect_map[col] = m
resect_cortex = resect[list(resect_map.keys())].rename(columns=resect_map)
resect_cortex = resect_cortex[region_names]
print(f"Resection aligned: {resect_cortex.shape}")

n_subj, n_reg = mind_nets.shape[0], mind_nets.shape[1]
half = n_reg // 2


def compute_node_features(network):
    """Compute 14 nodal features with corrected graph-theoretic methods."""
    n = network.shape[0]
    feat = {}

    # Degree and strength
    pos_net = (network > 0).astype(float)
    neg_net = (network < 0).astype(float)
    feat['pos_degree'] = pos_net.sum(axis=1)
    feat['neg_degree'] = neg_net.sum(axis=1)
    feat['pos_strength'] = np.where(network > 0, network, 0).sum(axis=1)
    feat['neg_strength'] = np.where(network < 0, -network, 0).sum(axis=1)
    feat['total_strength'] = feat['pos_strength'] + feat['neg_strength']

    # Distance-transformed network for betweenness/closeness
    # Similarity -> distance: d = 1 - |sim|
    dist = 1.0 - np.abs(network)
    np.fill_diagonal(dist, 0)
    G_dist = nx.from_numpy_array(dist)

    bc = nx.betweenness_centrality(G_dist, weight='weight', normalized=True)
    feat['betweenness'] = [bc[i] for i in range(n)]

    cc = nx.closeness_centrality(G_dist, distance='weight')
    feat['closeness'] = [cc[i] for i in range(n)]

    # Similarity-weighted network for eigenvector and clustering
    G_sim = nx.from_numpy_array(np.abs(network))
    ec = nx.eigenvector_centrality_numpy(G_sim, weight='weight')
    feat['eigenvector'] = [ec[i] for i in range(n)]

    clust = nx.clustering(G_sim, weight='weight')
    feat['clustering'] = [clust[i] for i in range(n)]

    # Module-based features (participation coefficient, within-module strength)
    from networkx.algorithms.community import greedy_modularity_communities
    comms = list(greedy_modularity_communities(G_sim, weight='weight'))
    comm_map = {}
    for ci, comm in enumerate(comms):
        for node in comm:
            comm_map[node] = ci

    deg = np.array([d for _, d in G_sim.degree(weight='weight')])
    pc = np.zeros(n)
    wms = np.zeros(n)
    for i in range(n):
        ci = comm_map[i]
        within = 0.0
        for j in range(n):
            if j != i and comm_map[j] == ci:
                within += np.abs(network[i, j])
        total = deg[i]
        pc[i] = 1 - (within / total) ** 2 if total > 0 else 0
        wms[i] = within
    feat['participation'] = pc
    feat['within_module_strength'] = wms

    # Hemispheric features
    left_str = np.zeros(n)
    right_str = np.zeros(n)
    inter_str = np.zeros(n)
    for i in range(n):
        left_conn = np.abs(network[i, :half])
        right_conn = np.abs(network[i, half:])
        left_str[i] = left_conn[left_conn > 0].mean() if (left_conn > 0).any() else 0
        right_str[i] = right_conn[right_conn > 0].mean() if (right_conn > 0).any() else 0
        if i < half:
            inter_str[i] = right_str[i]
        else:
            inter_str[i] = left_str[i]
    feat['left_hemi_strength'] = left_str
    feat['right_hemi_strength'] = right_str
    feat['inter_hemi_strength'] = inter_str

    return pd.DataFrame(feat)


print("\nComputing nodal features for all subjects...")
all_feats = []
for i in range(n_subj):
    nf = compute_node_features(mind_nets[i])
    nf['subject_id'] = clin.index[i]
    nf['region'] = region_names
    subj_resect = resect_cortex.iloc[i].values
    nf['resected'] = (subj_resect > THRESHOLD).astype(int)
    nf['resect_pct'] = subj_resect
    all_feats.append(nf)

node_df = pd.concat(all_feats, ignore_index=True)
feat_cols = [
    'pos_degree', 'neg_degree', 'pos_strength', 'neg_strength', 'total_strength',
    'betweenness', 'closeness', 'eigenvector', 'clustering', 'participation',
    'within_module_strength', 'left_hemi_strength', 'right_hemi_strength',
    'inter_hemi_strength'
]

n_resected = (node_df['resected'] == 1).sum()
n_non = (node_df['resected'] == 0).sum()
print(f"Resected regions (>10%): {n_resected} ({n_resected / (n_resected + n_non) * 100:.1f}%)")
print(f"Non-resected: {n_non}")

node_df.to_csv(os.path.join(RESULT, 'node_features.csv'), index=False)

# ========== STATISTICS ==========
print("\n" + "=" * 60)
print("STATISTICAL ANALYSIS")
print("=" * 60)

# Within-subject paired test (primary analysis, accounts for non-independence)
print("\n--- Within-subject paired test (primary) ---")
paired_results = []
for f in feat_cols:
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
            'cohens_d': d, 't': t, 'p_value': p, 'n_patients': len(diffs)
        })

paired_df = pd.DataFrame(paired_results).sort_values('p_value')
_, paired_df['p_adj'], _, _ = multipletests(paired_df['p_value'], method='fdr_bh')
print(paired_df[['feature', 'cohens_d', 'p_value', 'p_adj']].to_string(index=False))
paired_df.to_csv(os.path.join(RESULT, 'paired_stats.csv'), index=False)

# Node-level independent t-test (descriptive, with correct Cohen's d)
print("\n--- Node-level independent t-test (descriptive) ---")
comp = []
for f in feat_cols:
    r = node_df[node_df['resected'] == 1][f].dropna()
    nr = node_df[node_df['resected'] == 0][f].dropna()
    if len(r) > 1 and len(nr) > 1:
        t, p = stats.ttest_ind(r, nr)
        n1, n2 = len(r), len(nr)
        s1, s2 = r.std(ddof=1), nr.std(ddof=1)
        pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
        d = (r.mean() - nr.mean()) / pooled if pooled > 0 else 0
        comp.append({
            'feature': f, 'resected_mean': r.mean(), 'non_resected_mean': nr.mean(),
            't': t, 'p_value': p, 'cohens_d': d
        })
comp_df = pd.DataFrame(comp).sort_values('p_value')
_, comp_df['p_adj'], _, _ = multipletests(comp_df['p_value'], method='fdr_bh')
print(comp_df[['feature', 'cohens_d', 'p_value', 'p_adj']].to_string(index=False))
comp_df.to_csv(os.path.join(RESULT, 'node_feature_comparison.csv'), index=False)

# ========== CLASSIFICATION ==========
print("\n" + "=" * 60)
print("RANDOM FOREST CLASSIFICATION (GroupKFold by subject)")
print("=" * 60)

X = node_df[feat_cols].values
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

# Bootstrap 95% CI
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
print(f"Confusion matrix: TN={cm[0, 0]} FP={cm[0, 1]} FN={cm[1, 0]} TP={cm[1, 1]}")

pd.DataFrame({
    'metric': ['AUC', 'AUC_CI_low', 'AUC_CI_high', 'Accuracy', 'Sensitivity', 'Specificity'],
    'value': [auc, ci_low, ci_high, acc, sens, spec]
}).to_csv(os.path.join(RESULT, 'classifier_results.csv'), index=False)

# Patient-level evaluation
print("\n--- Patient-level evaluation ---")
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

# Feature importance (full model, descriptive)
sc = StandardScaler()
Xs = sc.fit_transform(X)
clf_full = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_split=2,
    class_weight='balanced', random_state=42, n_jobs=-1
)
clf_full.fit(Xs, y)
imp = pd.DataFrame({'feature': feat_cols, 'importance': clf_full.feature_importances_})
imp = imp.sort_values('importance', ascending=False)
print("\nFeature importance:")
print(imp.to_string(index=False))
imp.to_csv(os.path.join(RESULT, 'feature_importance.csv'), index=False)

# ========== STRATIFIED ANALYSIS ==========
print("\n" + "=" * 60)
print("STRATIFIED ANALYSIS BY PATHOLOGY")
print("=" * 60)

pathologies = clin['Pathology'].value_counts().index.tolist()
strat_results = []
for path in pathologies:
    path_subjs = clin[clin['Pathology'] == path].index
    path_node = node_df[node_df['subject_id'].isin(path_subjs)]
    if len(path_subjs) < 10 or (path_node['resected'] == 1).sum() < 10:
        continue
    Xp = path_node[feat_cols].values
    yp = path_node['resected'].values
    gp = path_node['subject_id'].values
    n_splits = min(5, len(np.unique(gp)))
    if n_splits < 2:
        continue
    gkf_p = GroupKFold(n_splits=n_splits)
    yp_prob = np.zeros(len(yp))
    for tr, te in gkf_p.split(Xp, yp, gp):
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xp[tr])
        Xte = sc.transform(Xp[te])
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=2,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        clf.fit(Xtr, yp[tr])
        yp_prob[te] = clf.predict_proba(Xte)[:, 1]
    try:
        auc_p = roc_auc_score(yp, yp_prob)
    except Exception:
        auc_p = np.nan
    strat_results.append({
        'pathology': path, 'n_patients': len(path_subjs),
        'n_resected_nodes': (path_node['resected'] == 1).sum(), 'AUC': auc_p
    })
    print(f"  {path}: n={len(path_subjs)}, AUC={auc_p:.3f}")

pd.DataFrame(strat_results).to_csv(os.path.join(RESULT, 'pathology_stratified.csv'), index=False)

# Surgical side stratification
print("\n--- By surgical side ---")
for side in ['L', 'R']:
    side_subjs = clin[clin['Op_Side'] == side].index
    side_node = node_df[node_df['subject_id'].isin(side_subjs)]
    if len(side_subjs) < 10:
        continue
    Xs = side_node[feat_cols].values
    ys = side_node['resected'].values
    gs = side_node['subject_id'].values
    n_splits = min(5, len(np.unique(gs)))
    gkf_s = GroupKFold(n_splits=n_splits)
    ys_prob = np.zeros(len(ys))
    for tr, te in gkf_s.split(Xs, ys, gs):
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xs[tr])
        Xte = sc.transform(Xs[te])
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=2,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        clf.fit(Xtr, ys[tr])
        ys_prob[te] = clf.predict_proba(Xte)[:, 1]
    try:
        auc_s = roc_auc_score(ys, ys_prob)
    except Exception:
        auc_s = np.nan
    print(f"  Side {side}: n={len(side_subjs)}, AUC={auc_s:.3f}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print(f"Results saved to: {RESULT}")
print("=" * 60)
