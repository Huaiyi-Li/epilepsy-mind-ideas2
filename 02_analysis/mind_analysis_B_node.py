# -*- coding: utf-8 -*-
"""
Direction B：MINDnodesfeatureOutcome prediction
- 68nodes//Centralityfeature (68x8=544feature)
- feature selection (ANOVA F-value top-K)
- comparison: Global features vs nodesfeature vs +nodes
"""
import pandas as pd
import numpy as np
import os
import networkx as nx
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# ========== Configuration ==========
base_dir = r"D:\epilepsy_MIND"
proc_dir = os.path.join(base_dir, "processed")
result_dir = os.path.join(base_dir, "results")
fig_dir = os.path.join(result_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

# ========== Loading data ==========
print("Loading data...")
mind_nets = np.load(os.path.join(proc_dir, "mind_networks.npy"))
thick_aligned = pd.read_csv(os.path.join(proc_dir, "thickness_aligned.csv"), index_col=0)
clin = pd.read_csv(os.path.join(proc_dir, "clinical_aligned.csv"), index_col=0)
with open(os.path.join(proc_dir, "region_names.txt")) as f:
    region_names = [line.strip() for line in f]

subject_ids = thick_aligned.index.tolist()
clin = clin[~clin.index.duplicated(keep='first')].loc[subject_ids]
clin['outcome'] = (clin['ILAE_Year1'] == 1).astype(int)
print(f"subjects: {len(subject_ids)}, good: {(clin['outcome']==1).sum()}, poor: {(clin['outcome']==0).sum()}")

# ========== ComputingMINDGlobal features ==========
print("ComputingMINDGlobal features...")
def compute_global_features(network):
    n = network.shape[0]
    f = {}
    nz = network[network != 0]
    f['mind_mean_conn'] = nz.mean() if len(nz)>0 else 0
    f['mind_std_conn'] = nz.std() if len(nz)>0 else 0
    f['mind_pos_ratio'] = (network > 0).sum() / (n*(n-1))
    f['mind_neg_ratio'] = (network < 0).sum() / (n*(n-1))
    G = nx.from_numpy_array(np.abs(network))
    try: f['mind_avg_clustering'] = nx.average_clustering(G, weight='weight')
    except: f['mind_avg_clustering'] = 0
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = greedy_modularity_communities(G, weight='weight')
        f['mind_modularity'] = nx.community.modularity(G, comms, weight='weight')
    except: f['mind_modularity'] = 0
    degs = np.array([d for _,d in G.degree(weight='weight')])
    f['mind_mean_degree'] = degs.mean()
    f['mind_std_degree'] = degs.std()
    half = n//2
    f['mind_inter_hemi'] = np.abs(network[:half,half:][network[:half,half:]!=0]).mean() if (network[:half,half:]!=0).any() else 0
    return f

mind_global = pd.DataFrame([compute_global_features(mind_nets[i]) for i in range(len(subject_ids))], 
                            index=subject_ids)
global_feat_names = list(mind_global.columns)
print(f"Global features: {mind_global.shape[1]}")

# ========== ComputingMINDnodesfeature ==========
print("ComputingMINDnodesfeature...")
def compute_node_features(network):
    n = network.shape[0]
    half = n//2
    feat = {}
    pos_net = (network > 0).astype(float)
    neg_net = (network < 0).astype(float)
    feat['pos_degree'] = pos_net.sum(axis=1)
    feat['neg_degree'] = neg_net.sum(axis=1)
    feat['pos_strength'] = np.where(network > 0, network, 0).sum(axis=1)
    feat['neg_strength'] = np.where(network < 0, -network, 0).sum(axis=1)
    feat['total_strength'] = feat['pos_strength'] + feat['neg_strength']
    G = nx.from_numpy_array(np.abs(network))
    try:
        bc = nx.betweenness_centrality(G, weight='weight')
        feat['betweenness'] = [bc[i] for i in range(n)]
    except: feat['betweenness'] = [0]*n
    try:
        ec = nx.eigenvector_centrality_numpy(G, weight='weight')
        feat['eigenvector'] = [ec[i] for i in range(n)]
    except: feat['eigenvector'] = [0]*n
    # 
    inter = []
    for i in range(n):
        if i < half:
            inter.append(np.abs(network[i,half:][network[i,half:]!=0]).mean() if (network[i,half:]!=0).any() else 0)
        else:
            inter.append(np.abs(network[i,:half][network[i,:half]!=0]).mean() if (network[i,:half]!=0).any() else 0)
    feat['inter_hemi'] = inter
    return pd.DataFrame(feat)

node_feat_names = ['pos_degree','neg_degree','pos_strength','neg_strength',
                    'total_strength','betweenness','eigenvector','inter_hemi']

# nodesfeaturematrix: subjects → 68nodes × 8feature = 544feature
node_features_list = []
node_feature_names = []
for i in range(len(subject_ids)):
    nf = compute_node_features(mind_nets[i])
    row = {}
    for feat in node_feat_names:
        for j, reg in enumerate(region_names):
            row[f'{reg}_{feat}'] = nf.iloc[j][feat]
    node_features_list.append(row)
    if i == 0:
        node_feature_names = list(row.keys())

mind_node = pd.DataFrame(node_features_list, index=subject_ids)
print(f"nodesfeature: {mind_node.shape[1]} (68nodes × 8feature)")

# ========== Clinical variables ==========
clin_vars = ['Sex','Op_Side','Pathology','Binned_Age_at_Scan','Binned_Onset_Age','SE','FUS']
clin_dum = pd.get_dummies(clin[clin_vars], drop_first=True)
clin_feat_names = list(clin_dum.columns)
print(f"Clinical variables: {clin_dum.shape[1]}")

# ========== Outcome predictioncomparison ==========
print("\n" + "=" * 60)
print("Outcome predictioncomparison (5CV, Random Forest)")
y = clin['outcome'].values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_with_selection(X, y, label, k=50):
    """feature selectionevaluation"""
    y_prob = np.zeros(len(y))
    n_features = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        # feature selection
        k_actual = min(k, Xtr.shape[1])
        selector = SelectKBest(f_classif, k=k_actual)
        Xtr_sel = selector.fit_transform(Xtr, y[tr])
        Xte_sel = selector.transform(Xte)
        n_features.append(Xtr_sel.shape[1])
        clf = RandomForestClassifier(n_estimators=300, max_depth=8, 
                                      class_weight='balanced', random_state=42, n_jobs=-1)
        clf.fit(Xtr_sel, y[tr])
        y_prob[te] = clf.predict_proba(Xte_sel)[:,1]
    y_pred = (y_prob > 0.5).astype(int)
    auc = roc_auc_score(y, y_prob)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    sens = cm[1,1]/(cm[1,0]+cm[1,1]) if (cm[1,0]+cm[1,1])>0 else 0
    spec = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1])>0 else 0
    print(f"  [{label}] AUC={auc:.3f} Acc={acc:.3f} Sens={sens:.3f} Spec={spec:.3f} (avg features={np.mean(n_features):.0f})")
    return {'model':label,'auc':auc,'accuracy':acc,'sensitivity':sens,'specificity':spec}

X_global = mind_global.values
X_node = mind_node.values
X_clin = clin_dum.values
X_global_node = np.hstack([X_global, X_node])
X_global_clin = np.hstack([X_global, X_clin])
X_node_clin = np.hstack([X_node, X_clin])
X_all = np.hstack([X_global, X_node, X_clin])

results = []
results.append(evaluate_with_selection(X_global, y, "Global features(11)", k=10))
results.append(evaluate_with_selection(X_node, y, "nodesfeature(544→50)", k=50))
results.append(evaluate_with_selection(X_clin, y, "Clinical only", k=20))
results.append(evaluate_with_selection(X_global_node, y, "+nodes", k=50))
results.append(evaluate_with_selection(X_global_clin, y, "+", k=30))
results.append(evaluate_with_selection(X_node_clin, y, "nodes+", k=50))
results.append(evaluate_with_selection(X_all, y, "+nodes+(feature)", k=80))

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(result_dir, "B3_node_level_outcome.csv"), index=False)

# ========== nodesFeature importanceAnalysis ==========
print("\n" + "=" * 60)
print("nodesFeature importance (trained on full sample, SelectKBest)")
sc = StandardScaler()
X_node_scaled = sc.fit_transform(X_node)
selector = SelectKBest(f_classif, k=50)
X_node_sel = selector.fit_transform(X_node_scaled, y)
selected_indices = selector.get_support(indices=True)
f_scores = selector.scores_[selected_indices]
p_values = selector.pvalues_[selected_indices]

selected_features = [node_feature_names[i] for i in selected_indices]
feat_imp = pd.DataFrame({'feature':selected_features, 'f_score':f_scores, 'p_value':p_values})
feat_imp = feat_imp.sort_values('f_score', ascending=False)
print(feat_imp.head(20).to_string(index=False))
feat_imp.to_csv(os.path.join(result_dir, "B3_node_feature_importance.csv"), index=False)

# regions (regionsmeanf-score)
print("\nregionsFeature importance (top 15regions):")
region_fscores = {}
for _, row in feat_imp.iterrows():
    feat = row['feature']
    # regions (removing _feature suffix)
    for nf in node_feat_names:
        if feat.endswith('_' + nf):
            region = feat[:-(len(nf)+1)]
            if region not in region_fscores:
                region_fscores[region] = []
            region_fscores[region].append(row['f_score'])
            break

region_avg = {r: np.mean(fs) for r, fs in region_fscores.items()}
region_avg_df = pd.DataFrame([{'region':r, 'avg_f_score':v, 'n_features':len(region_fscores[r])} 
                                for r,v in region_avg.items()]).sort_values('avg_f_score', ascending=False)
print(region_avg_df.head(15).to_string(index=False))
region_avg_df.to_csv(os.path.join(result_dir, "B3_region_level_importance.csv"), index=False)

# ========== Visualization ==========
print("\nGeneratingVisualization...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150

# 6: Model comparison
fig, ax = plt.subplots(figsize=(12, 5))
labels = [r['model'] for r in results]
aucs = [r['auc'] for r in results]
colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
bars = ax.bar(range(len(labels)), aucs, color=colors)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('AUC')
ax.set_title('Outcome Prediction: Global vs Node-level MIND Features')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
ax.set_ylim(0, 0.8)
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{auc:.3f}', ha='center', fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig6_outcome_model_comparison.png'))
plt.close()

# 7: regionsHeatmap
fig, ax = plt.subplots(figsize=(14, 4))
top_regions = region_avg_df.head(30)
colors = ['#e74c3c' if r.startswith('lh_') else '#3498db' for r in top_regions['region']]
ax.bar(range(len(top_regions)), top_regions['avg_f_score'], color=colors)
ax.set_xticks(range(len(top_regions)))
ax.set_xticklabels([r.replace('lh_','L-').replace('rh_','R-') for r in top_regions['region']], 
                    rotation=90, fontsize=7)
ax.set_ylabel('Avg F-score')
ax.set_title('Top 30 Brain Regions for Outcome Prediction (Red=Left, Blue=Right)')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig7_outcome_important_regions.png'))
plt.close()

# ========== Summary ==========
print("\n" + "=" * 60)
print("Direction BnodesfeatureAnalysis complete！")
print("=" * 60)
best = res_df.loc[res_df['auc'].idxmax()]
print(f"\n: {best['model']} (AUC={best['auc']:.3f})")
print(f"\nresultfile:")
print(f"  B3_node_level_outcome.csv")
print(f"  B3_node_feature_importance.csv")
print(f"  B3_region_level_importance.csv")
print(f"\nfigure:")
print(f"  fig6_outcome_model_comparison.png")
print(f"  fig7_outcome_important_regions.png")
