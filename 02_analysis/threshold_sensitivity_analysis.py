# -*- coding: utf-8 -*-
"""
Resection threshold sensitivity analysis
Tests localization performance and effect sizes at 5%, 10%, 15%, 20% thresholds
"""
import pandas as pd
import numpy as np
import os
import networkx as nx
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

base_dir = r"D:\epilepsy_MIND"
proc_dir = os.path.join(base_dir, "processed")
result_dir = os.path.join(base_dir, "results_threshold_sensitivity")
os.makedirs(result_dir, exist_ok=True)

# Load data
print("Loading data...")
mind_nets = np.load(os.path.join(proc_dir, "mind_networks.npy"))
thick = pd.read_csv(os.path.join(proc_dir, "thickness_aligned.csv"), index_col=0)
clin = pd.read_csv(os.path.join(proc_dir, "clinical_aligned.csv"), index_col=0)
resect = pd.read_csv(os.path.join(proc_dir, "resection_aligned.csv"), index_col=0)
with open(os.path.join(proc_dir, "region_names.txt")) as f:
    region_names = [line.strip() for line in f]

subject_ids = thick.index.tolist()
clin = clin[~clin.index.duplicated(keep='first')].loc[subject_ids]
n_subjects = len(subject_ids)
n_regions = len(region_names)
half = n_regions // 2

# Region name mapping
def map_resect_region(name):
    if name.startswith('ctx-lh-'): return 'lh_' + name[7:]
    if name.startswith('ctx-rh-'): return 'rh_' + name[7:]
    return None
resect_map = {c: map_resect_region(c) for c in resect.columns if map_resect_region(c) and map_resect_region(c) in region_names}
resect_cortex = resect[list(resect_map.keys())].rename(columns=resect_map)[region_names]

print(f"Subjects: {n_subjects}, Regions: {n_regions}")

# Nodal feature extraction function (consistent with main analysis)
def extract_nodal_features(network):
    n = network.shape[0]
    half = n // 2
    feat = {}
    abs_net = np.abs(network)
    # Degree and strength
    pos_mask = network > 0
    neg_mask = network < 0
    feat['pos_degree'] = pos_mask.sum(axis=1)
    feat['neg_degree'] = neg_mask.sum(axis=1)
    feat['pos_strength'] = np.where(pos_mask, network, 0).sum(axis=1)
    feat['neg_strength'] = np.where(neg_mask, np.abs(network), 0).sum(axis=1)
    feat['total_strength'] = abs_net.sum(axis=1)
    # Distance matrix for betweenness/closeness
    distance = 1.0 - abs_net
    np.fill_diagonal(distance, 0)
    G_dist = nx.from_numpy_array(distance)
    feat['betweenness'] = np.array(list(nx.betweenness_centrality(G_dist, weight='weight').values()))
    feat['closeness'] = np.array(list(nx.closeness_centrality(G_dist, distance='weight').values()))
    # Similarity network for eigenvector/clustering
    G_sim = nx.from_numpy_array(abs_net)
    try:
        feat['eigenvector'] = np.array(list(nx.eigenvector_centrality_numpy(G_sim, weight='weight').values()))
    except:
        feat['eigenvector'] = np.zeros(n)
    feat['clustering'] = np.array(list(nx.clustering(G_sim, weight='weight').values()))
    # Module and participation coefficient
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G_sim, weight='weight'))
        comm_dict = {}
        for ci, comm in enumerate(comms):
            for node in comm:
                comm_dict[node] = ci
        comm_labels = np.array([comm_dict[i] for i in range(n)])
        within_strength = np.zeros(n)
        for i in range(n):
            same_comm = comm_labels == comm_labels[i]
            within_strength[i] = abs_net[i, same_comm].sum()
        feat['within_module_strength'] = within_strength
        total_str = abs_net.sum(axis=1)
        participation = np.zeros(n)
        for i in range(n):
            if total_str[i] > 0:
                comm_str = {}
                for j in range(n):
                    c = comm_labels[j]
                    comm_str[c] = comm_str.get(c, 0) + abs_net[i, j]
                participation[i] = 1 - sum((s/total_str[i])**2 for s in comm_str.values())
        feat['participation'] = participation
    except:
        feat['within_module_strength'] = np.zeros(n)
        feat['participation'] = np.zeros(n)
    # Hemisphere features
    left_str = np.zeros(n)
    right_str = np.zeros(n)
    inter_str = np.zeros(n)
    for i in range(n):
        left_conn = abs_net[i, :half]
        right_conn = abs_net[i, half:]
        left_str[i] = left_conn[left_conn > 0].mean() if (left_conn > 0).any() else 0
        right_str[i] = right_conn[right_conn > 0].mean() if (right_conn > 0).any() else 0
        inter_str[i] = right_str[i] if i < half else left_str[i]
    feat['left_hemi_strength'] = left_str
    feat['right_hemi_strength'] = right_str
    feat['inter_hemi_strength'] = inter_str
    return feat

# Extract nodal features for all subjects
print("Extracting nodal features...")
all_features = []
for s in range(n_subjects):
    feat = extract_nodal_features(mind_nets[s])
    for r in range(n_regions):
        row = {'subject_id': subject_ids[s], 'region': region_names[r]}
        for k, v in feat.items():
            row[k] = v[r]
        all_features.append(row)
node_df = pd.DataFrame(all_features)
print(f"Nodal feature matrix: {node_df.shape}")

feature_cols = [c for c in node_df.columns if c not in ['subject_id', 'region']]
print(f"Number of features: {len(feature_cols)}")

# Threshold sensitivity analysis
thresholds = [0.05, 0.10, 0.15, 0.20]
results = []

for threshold in thresholds:
    print(f"\n{'='*60}")
    print(f"Threshold: {threshold*100:.0f}%")
    print(f"{'='*60}")
    
    # Define resection labels
    resect_binary = (resect_cortex.values > threshold).astype(int)
    
    # Count resected regions
    n_resected_nodes = resect_binary.sum()
    n_total_nodes = n_subjects * n_regions
    print(f"Resected nodes: {n_resected_nodes}/{n_total_nodes} ({n_resected_nodes/n_total_nodes*100:.1f}%)")
    
    # Number of patients with detectable resection
    patients_with_resection = (resect_binary.sum(axis=1) > 0).sum()
    print(f"Patients with detectable resection: {patients_with_resection}/{n_subjects}")
    
    # Add resection labels
    node_df['resected'] = resect_binary.flatten()
    
    # Keep only patients with resected regions
    patients_with = [subject_ids[i] for i in range(n_subjects) if resect_binary[i].sum() > 0]
    df_analysis = node_df[node_df['subject_id'].isin(patients_with)].copy()
    
    # Within-subject paired tests
    paired_results = []
    for f in feature_cols:
        diffs = []
        for subj in patients_with:
            subj_data = df_analysis[df_analysis['subject_id'] == subj]
            r_vals = subj_data[subj_data['resected']==1][f].values
            nr_vals = subj_data[subj_data['resected']==0][f].values
            if len(r_vals) > 0 and len(nr_vals) > 0:
                diffs.append(r_vals.mean() - nr_vals.mean())
        diffs = np.array(diffs)
        if len(diffs) > 10:
            t, p = stats.ttest_1samp(diffs, 0)
            d = np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-10)
            paired_results.append({'feature': f, 'mean_diff': np.mean(diffs), 'cohens_d': d, 'p_value': p})
    paired_df = pd.DataFrame(paired_results).sort_values('p_value')
    # FDR correction
    from statsmodels.stats.multitest import multipletests
    _, paired_df['p_adj'], _, _ = multipletests(paired_df['p_value'].values, method='fdr_bh')
    print(f"\nPaired test Top 5 (sorted by |d|):")
    top_paired = paired_df.reindex(paired_df['cohens_d'].abs().sort_values(ascending=False).index).head(5)
    print(top_paired[['feature','cohens_d','p_adj']].to_string(index=False))
    
    # Random forest classification (GroupKFold)
    X = df_analysis[feature_cols].values
    y = df_analysis['resected'].values
    groups = df_analysis['subject_id'].values
    
    gkf = GroupKFold(n_splits=5)
    all_proba = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=2,
                                     class_weight='balanced', random_state=42, n_jobs=-1)
        rf.fit(Xtr, y[tr])
        all_proba[te] = rf.predict_proba(Xte)[:, 1]
    
    auc = roc_auc_score(y, all_proba)
    # 95% CI via bootstrap
    np.random.seed(42)
    auc_boots = []
    for _ in range(1000):
        idx = np.random.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) > 1:
            auc_boots.append(roc_auc_score(y[idx], all_proba[idx]))
    auc_ci = np.percentile(auc_boots, [2.5, 97.5])
    
    # Patient-level metrics
    patient_auc = []
    patient_correct = 0
    patient_total = 0
    for subj in patients_with:
        mask = df_analysis['subject_id'] == subj
        if mask.sum() > 0 and y[mask].sum() > 0 and y[mask].sum() < mask.sum():
            patient_total += 1
            if all_proba[mask][y[mask]==1].mean() > all_proba[mask][y[mask]==0].mean():
                patient_correct += 1
    patient_acc = patient_correct / patient_total if patient_total > 0 else 0
    
    print(f"\nClassification performance:")
    print(f"  AUC = {auc:.3f} (95% CI: {auc_ci[0]:.3f}-{auc_ci[1]:.3f})")
    print(f"  Patient-level accuracy = {patient_acc*100:.1f}% ({patient_correct}/{patient_total})")
    
    results.append({
        'threshold': threshold,
        'pct_resected': n_resected_nodes/n_total_nodes*100,
        'n_patients_with_resection': patients_with_resection,
        'auc': auc,
        'auc_ci_low': auc_ci[0],
        'auc_ci_high': auc_ci[1],
        'patient_accuracy': patient_acc,
        'patient_correct': patient_correct,
        'patient_total': patient_total,
        'top_feature': top_paired.iloc[0]['feature'],
        'top_feature_d': top_paired.iloc[0]['cohens_d'],
    })
    
    # Save paired test results
    paired_df.to_csv(os.path.join(result_dir, f"paired_stats_threshold_{int(threshold*100)}.csv"), index=False)

# Summary results
results_df = pd.DataFrame(results)
print(f"\n{'='*60}")
print("Threshold sensitivity analysis summary")
print(f"{'='*60}")
print(results_df[['threshold','pct_resected','auc','auc_ci_low','auc_ci_high','patient_accuracy']].to_string(index=False))
results_df.to_csv(os.path.join(result_dir, "threshold_sensitivity_summary.csv"), index=False)
print(f"\nResults saved to {result_dir}")
