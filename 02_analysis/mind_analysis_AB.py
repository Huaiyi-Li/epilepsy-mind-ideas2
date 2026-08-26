# -*- coding: utf-8 -*-
"""
IDEAS-II Epilepsy MIND Analysis - Step 2
Direction A: EZ localization (resected vs non-resected MIND nodal features)
Direction B: Outcome prediction (MIND global features + clinical variables -> ILAE outcome)
"""
import pandas as pd
import numpy as np
import os
import networkx as nx
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ========== Configuration ==========
base_dir = r"D:\epilepsy_MIND"
proc_dir = os.path.join(base_dir, "processed")
result_dir = os.path.join(base_dir, "results")
os.makedirs(result_dir, exist_ok=True)

# ========== Load Data ==========
print("=" * 60)
print("Loading data")
mind_nets = np.load(os.path.join(proc_dir, "mind_networks.npy"))
thick_aligned = pd.read_csv(os.path.join(proc_dir, "thickness_aligned.csv"), index_col=0)
clin = pd.read_csv(os.path.join(proc_dir, "clinical_aligned.csv"), index_col=0)
resect = pd.read_csv(os.path.join(proc_dir, "resection_aligned.csv"), index_col=0)
with open(os.path.join(proc_dir, "region_names.txt")) as f:
    region_names = [line.strip() for line in f]

# Use thickness_aligned index as standard subject IDs (consistent with mind_nets order)
subject_ids = thick_aligned.index.tolist()
# Clinical data has duplicate IDs, deduplicate first
clin = clin[~clin.index.duplicated(keep='first')]
clin = clin.loc[subject_ids]
resect = resect.loc[subject_ids]

print(f"  MIND networks: {mind_nets.shape}")
print(f"  Clinical data: {clin.shape}")
print(f"  Resection ratios: {resect.shape}")
print(f"  Number of regions: {len(region_names)}")

# ========== Region Name Mapping (resection table -> MIND) ==========
print("\n" + "=" * 60)
print("Region name mapping")

def map_region(name):
    """ctx-lh-bankssts -> lh_bankssts"""
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

print(f"  Resected cortical regions: {len(resect_map)} / {len(resect.columns)}")
resect_cortex = resect[list(resect_map.keys())].rename(columns=resect_map)
resect_cortex = resect_cortex[region_names]
print(f"  Aligned resection matrix: {resect_cortex.shape}")

# ========== Direction A: EZ Localization ==========
print("\n" + "=" * 60)
print("Direction A: Epileptogenic zone localization")
print("  Gold standard: resection ratio > 0 = EZ; compare MIND nodal features resected vs non-resected")

def compute_node_features(network):
    n = network.shape[0]
    pos_net = (network > 0).astype(float)
    neg_net = (network < 0).astype(float)
    feat = {}
    feat['pos_degree'] = pos_net.sum(axis=1)
    feat['neg_degree'] = neg_net.sum(axis=1)
    feat['total_degree'] = feat['pos_degree'] + feat['neg_degree']
    feat['pos_strength'] = np.where(network > 0, network, 0).sum(axis=1)
    feat['neg_strength'] = np.where(network < 0, -network, 0).sum(axis=1)
    feat['total_strength'] = feat['pos_strength'] + feat['neg_strength']
    # Centrality: betweenness/closeness use DISTANCE matrix (1 - |similarity|)
    # because NetworkX interprets weight as traversal cost.
    abs_sim = np.abs(network)
    distance = 1.0 - abs_sim
    np.fill_diagonal(distance, 0.0)
    G_dist = nx.from_numpy_array(distance)
    G_sim = nx.from_numpy_array(abs_sim)
    try:
        bc = nx.betweenness_centrality(G_dist, weight='weight')
        feat['betweenness'] = [bc[i] for i in range(n)]
    except Exception:
        feat['betweenness'] = [0]*n
    try:
        cc = nx.closeness_centrality(G_dist, distance='weight')
        feat['closeness'] = [cc[i] for i in range(n)]
    except Exception:
        feat['closeness'] = [0]*n
    try:
        ec = nx.eigenvector_centrality_numpy(G_sim, weight='weight')
        feat['eigenvector'] = [ec[i] for i in range(n)]
    except Exception:
        feat['eigenvector'] = [0]*n
    return pd.DataFrame(feat)

print("\n  Computing nodal features for all subjects...")
all_feats = []
for i in range(mind_nets.shape[0]):
    nf = compute_node_features(mind_nets[i])
    nf['subject_id'] = clin.index[i]
    nf['region'] = region_names
    subj_resect = resect_cortex.iloc[i].values
    nf['resected'] = (subj_resect > 0).astype(int)
    nf['resect_pct'] = subj_resect
    all_feats.append(nf)

node_df = pd.concat(all_feats, ignore_index=True)
feat_cols = ['pos_degree','neg_degree','total_degree','pos_strength','neg_strength',
             'total_strength','betweenness','closeness','eigenvector']
print(f"  Nodal feature matrix: {node_df.shape}")
print(f"  Resected: {(node_df['resected']==1).sum()}, Non-resected: {(node_df['resected']==0).sum()}")

# Statistical comparison
print("\n  Resected vs non-resected statistical comparison (sorted by p-value):")
comp = []
for f in feat_cols:
    r = node_df[node_df['resected']==1][f].dropna()
    nr = node_df[node_df['resected']==0][f].dropna()
    if len(r)>1 and len(nr)>1:
        t, p = stats.ttest_ind(r, nr)
        pooled = np.sqrt((r.std()**2 + nr.std()**2)/2)
        d = (r.mean()-nr.mean())/pooled if pooled>0 else 0
        comp.append({'feature':f,'resected_mean':r.mean(),'non_resected_mean':nr.mean(),
                     't':t,'p_value':p,'cohens_d':d})
comp_df = pd.DataFrame(comp).sort_values('p_value')
print(comp_df.to_string(index=False))
comp_df.to_csv(os.path.join(result_dir, "A_node_feature_comparison.csv"), index=False)

# Classifier (subject-grouped 5-fold cross-validation)
print("\n  Training classifier for EZ prediction (subject-grouped 5-fold CV)...")
X = node_df[feat_cols].values
y = node_df['resected'].values
groups = node_df['subject_id'].values
gkf = GroupKFold(n_splits=5)
y_prob = np.zeros(len(y))
for tr, te in gkf.split(X, y, groups):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(Xtr, y[tr])
    y_prob[te] = clf.predict_proba(Xte)[:,1]
y_pred = (y_prob > 0.5).astype(int)
auc_a = roc_auc_score(y, y_prob)
acc_a = accuracy_score(y, y_pred)
cm_a = confusion_matrix(y, y_pred)
sens_a = cm_a[1,1]/(cm_a[1,0]+cm_a[1,1]) if (cm_a[1,0]+cm_a[1,1])>0 else 0
spec_a = cm_a[0,0]/(cm_a[0,0]+cm_a[0,1]) if (cm_a[0,0]+cm_a[0,1])>0 else 0
print(f"  AUC={auc_a:.3f}  Acc={acc_a:.3f}  Sens={sens_a:.3f}  Spec={spec_a:.3f}")
print(f"  Confusion matrix: TN={cm_a[0,0]} FP={cm_a[0,1]} FN={cm_a[1,0]} TP={cm_a[1,1]}")
pd.DataFrame({'metric':['AUC','Accuracy','Sensitivity','Specificity'],
              'value':[auc_a,acc_a,sens_a,spec_a]}).to_csv(
    os.path.join(result_dir,"A_classifier_results.csv"), index=False)

# ========== Direction B: Outcome Prediction ==========
print("\n" + "=" * 60)
print("Direction B: Outcome prediction")
print("  Label: ILAE_Year1==1 -> seizure-free (good); otherwise -> seizure recurrence (poor)")
clin['outcome'] = (clin['ILAE_Year1'] == 1).astype(int)
print(f"  Good: {(clin['outcome']==1).sum()}, Poor: {(clin['outcome']==0).sum()}")

def compute_global_features(network):
    n = network.shape[0]
    f = {}
    nonzero = network[network != 0]
    f['mean_conn'] = nonzero.mean() if len(nonzero)>0 else 0
    f['std_conn'] = nonzero.std() if len(nonzero)>0 else 0
    f['pos_ratio'] = (network > 0).sum() / (n*(n-1))
    f['neg_ratio'] = (network < 0).sum() / (n*(n-1))
    abs_sim = np.abs(network)
    distance = 1.0 - abs_sim
    np.fill_diagonal(distance, 0.0)
    G_sim = nx.from_numpy_array(abs_sim)
    G_dist = nx.from_numpy_array(distance)
    try: f['global_efficiency'] = nx.global_efficiency(G_dist)
    except Exception: f['global_efficiency'] = 0
    try: f['avg_clustering'] = nx.average_clustering(G_sim, weight='weight')
    except Exception: f['avg_clustering'] = 0
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = greedy_modularity_communities(G_sim, weight='weight')
        f['modularity'] = nx.community.modularity(G_sim, comms, weight='weight')
        f['n_communities'] = len(comms)
    except Exception:
        f['modularity'] = 0; f['n_communities'] = 0
    try: f['char_path_length'] = nx.average_shortest_path_length(G_dist, weight='weight')
    except Exception: f['char_path_length'] = 0
    degs = np.array([d for _,d in G_sim.degree(weight='weight')])
    f['mean_degree'] = degs.mean(); f['std_degree'] = degs.std(); f['max_degree'] = degs.max()
    # Hemispheric features
    half = n//2
    for name, mat in [('left',network[:half,:half]),('right',network[half:,half:]),
                       ('inter',network[:half,half:])]:
        nz = mat[mat != 0]
        f[f'{name}_strength'] = np.abs(nz).mean() if len(nz)>0 else 0
    return f

print("\n  Extracting MIND global network features...")
gfeats = [compute_global_features(mind_nets[i]) for i in range(mind_nets.shape[0])]
global_df = pd.DataFrame(gfeats, index=clin.index)
print(f"  Global features: {global_df.shape}")

# One-hot encoding for clinical variables
clin_vars = ['Sex','Op_Side','Pathology','Binned_Age_at_Scan','Binned_Onset_Age','SE','FUS']
clin_dum = pd.get_dummies(clin[clin_vars], drop_first=True)
print(f"  Clinical variables: {clin_dum.shape}")

X_g = global_df.values
X_c = clin_dum.values
X_all = np.hstack([X_g, X_c])
y_b = clin['outcome'].values
feat_names = list(global_df.columns) + list(clin_dum.columns)

# 5-fold CV evaluation
print("\n  Outcome prediction (5-fold CV):")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def eval_model(X, y, label):
    yp = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(Xtr, y[tr])
        yp[te] = clf.predict_proba(Xte)[:,1]
    ypred = (yp > 0.5).astype(int)
    auc = roc_auc_score(y, yp)
    acc = accuracy_score(y, ypred)
    cm = confusion_matrix(y, ypred)
    sens = cm[1,1]/(cm[1,0]+cm[1,1]) if (cm[1,0]+cm[1,1])>0 else 0
    spec = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1])>0 else 0
    print(f"  [{label}] AUC={auc:.3f} Acc={acc:.3f} Sens={sens:.3f} Spec={spec:.3f}")
    return {'model':label,'auc':auc,'acc':acc,'sensitivity':sens,'specificity':spec}

res_b = []
res_b.append(eval_model(X_g, y_b, "MIND global only"))
res_b.append(eval_model(X_c, y_b, "Clinical only"))
res_b.append(eval_model(X_all, y_b, "MIND + Clinical combined"))
pd.DataFrame(res_b).to_csv(os.path.join(result_dir,"B_outcome_results.csv"), index=False)

# Combined model feature importance
print("\n  Combined model feature importance (top 10, sorted by |coef|):")
sc = StandardScaler(); Xs = sc.fit_transform(X_all)
clf_f = LogisticRegression(max_iter=1000, class_weight='balanced')
clf_f.fit(Xs, y_b)
imp = pd.DataFrame({'feature':feat_names,'coef':clf_f.coef_[0]})
imp['abs_coef'] = imp['coef'].abs()
imp = imp.sort_values('abs_coef', ascending=False).drop('abs_coef', axis=1)
print(imp.head(10).to_string(index=False))
imp.to_csv(os.path.join(result_dir,"B_feature_importance.csv"), index=False)

# ========== Summary ==========
print("\n" + "=" * 60)
print("Analysis complete! Results directory:", result_dir)
print("=" * 60)
print("\nGenerated files:")
for f in sorted(os.listdir(result_dir)):
    print(f"  {f}")
