# -*- coding: utf-8 -*-
"""
Direction A：Visualization + stratifiedAnalysis
- Pathology type(HS/FCD/other)stratifiedfeature
- GeneratingVisualizationfigure: regionsHeatmap、MIND、Boxplot
"""
import pandas as pd
import numpy as np
import os
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ========== Configuration ==========
base_dir = r"D:\epilepsy_MIND"
proc_dir = os.path.join(base_dir, "processed")
result_dir = os.path.join(base_dir, "results")
fig_dir = os.path.join(result_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

# plotting
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# ========== Loading data ==========
print("Loading data...")
mind_nets = np.load(os.path.join(proc_dir, "mind_networks.npy"))
thick_aligned = pd.read_csv(os.path.join(proc_dir, "thickness_aligned.csv"), index_col=0)
clin = pd.read_csv(os.path.join(proc_dir, "clinical_aligned.csv"), index_col=0)
resect = pd.read_csv(os.path.join(proc_dir, "resection_aligned.csv"), index_col=0)
with open(os.path.join(proc_dir, "region_names.txt")) as f:
    region_names = [line.strip() for line in f]

subject_ids = thick_aligned.index.tolist()
clin = clin[~clin.index.duplicated(keep='first')].loc[subject_ids]

# Region mapping
def map_region(name):
    if name.startswith('ctx-lh-'): return 'lh_' + name[7:]
    if name.startswith('ctx-rh-'): return 'rh_' + name[7:]
    return None
resect_map = {c: map_region(c) for c in resect.columns if map_region(c) and map_region(c) in region_names}
resect_cortex = resect[list(resect_map.keys())].rename(columns=resect_map)[region_names]

# ========== ComputingNodal features ==========
print("ComputingNodal features...")
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
    feat['inter_hemi_strength'] = inter
    return pd.DataFrame(feat)

all_feats = []
for i in range(len(subject_ids)):
    nf = compute_node_features(mind_nets[i])
    nf['subject_id'] = subject_ids[i]
    nf['region'] = region_names
    nf['hemisphere'] = ['L' if r.startswith('lh_') else 'R' for r in region_names]
    subj_resect = resect_cortex.iloc[i].values
    nf['resected'] = (subj_resect > 0).astype(int)
    nf['resect_pct'] = subj_resect
    nf['pathology'] = clin.iloc[i]['Pathology']
    nf['op_side'] = clin.iloc[i]['Op_Side']
    all_feats.append(nf)

node_df = pd.concat(all_feats, ignore_index=True)
feat_cols = ['pos_degree','neg_degree','pos_strength','neg_strength','total_strength',
             'betweenness','eigenvector','inter_hemi_strength']
print(f"Nodal feature matrix: {node_df.shape}")

# ========== 1: resection frequencyregionsHeatmap ==========
print("\nGenerating1: resection frequencyHeatmap...")
resect_freq = node_df.groupby('region')['resected'].mean().reindex(region_names)
fig, ax = plt.subplots(figsize=(16, 4))
colors = ['#e74c3c' if r.startswith('lh_') else '#3498db' for r in region_names]
bars = ax.bar(range(len(region_names)), resect_freq.values, color=colors, edgecolor='white', linewidth=0.5)
ax.set_xticks(range(len(region_names)))
ax.set_xticklabels([r.replace('lh_','L-').replace('rh_','R-') for r in region_names], 
                    rotation=90, fontsize=6)
ax.set_ylabel('Resection Frequency')
ax.set_title('Resection Frequency by Brain Region (Red=Left, Blue=Right)')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig1_resection_frequency.png'))
plt.close()

# ========== 2: ResectedvsNon-resectedfeatureBoxplot ==========
print("Generating2: featureBoxplot...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for idx, feat in enumerate(feat_cols[:8]):
    ax = axes[idx]
    data_nr = node_df[node_df['resected']==0][feat].values
    data_r = node_df[node_df['resected']==1][feat].values
    # plotting
    if len(data_nr) > 5000: data_nr = np.random.choice(data_nr, 5000, replace=False)
    if len(data_r) > 5000: data_r = np.random.choice(data_r, 5000, replace=False)
    bp = ax.boxplot([data_nr, data_r], tick_labels=['Non-resected','Resected'], 
                     patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor('#95a5a6')
    bp['boxes'][1].set_facecolor('#e74c3c')
    # t
    t, p = stats.ttest_ind(node_df[node_df['resected']==0][feat], node_df[node_df['resected']==1][feat])
    ax.set_title(f'{feat}\np={p:.2e}', fontsize=9)
    ax.set_ylabel(feat)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig2_feature_boxplots.png'))
plt.close()

# ========== 3: subjectsMIND ==========
print("Generating3: MIND...")
# subjects
resect_by_subj = node_df.groupby('subject_id')['resected'].sum()
typical_id = resect_by_subj.idxmax()
typical_idx = subject_ids.index(typical_id)
typical_net = mind_nets[typical_idx]
typical_resect = resect_cortex.iloc[typical_idx].values

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
# MIND network matrix
im1 = ax1.imshow(typical_net, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax1.set_title(f'MIND Network - Subject {typical_id}\n({int(typical_resect.sum())} regions resected)')
ax1.set_xlabel('Region')
ax1.set_ylabel('Region')
# Resected
resect_idx = np.where(typical_resect > 0)[0]
for ri in resect_idx[:5]:
    ax1.axhline(y=ri, color='yellow', linewidth=0.5, alpha=0.5)
plt.colorbar(im1, ax=ax1, label='Correlation')

# Resection ratiosedges
ax2.barh(range(len(region_names)), typical_resect, color='#e74c3c', height=0.8)
ax2.set_yticks(range(len(region_names)))
ax2.set_yticklabels([r.replace('lh_','L-').replace('rh_','R-') for r in region_names], fontsize=5)
ax2.set_xlabel('Resection Percentage')
ax2.set_title(f'Resection Mask - Subject {typical_id}')
ax2.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig3_typical_mind_network.png'))
plt.close()

# ========== 4: regionseffect sizeHeatmap ==========
print("Generating4: regionseffect size...")
region_effect = []
for reg in region_names:
    sub = node_df[node_df['region']==reg]
    r = sub[sub['resected']==1]['total_strength'].dropna()
    nr = sub[sub['resected']==0]['total_strength'].dropna()
    if len(r)>5 and len(nr)>5 and r.std()>0 and nr.std()>0:
        pooled = np.sqrt((r.std()**2 + nr.std()**2)/2)
        d = (r.mean()-nr.mean())/pooled
        region_effect.append(d)
    else:
        region_effect.append(0)

fig, ax = plt.subplots(figsize=(16, 4))
colors = ['#e74c3c' if d < 0 else '#2ecc71' for d in region_effect]
ax.bar(range(len(region_names)), region_effect, color=colors, edgecolor='white', linewidth=0.5)
ax.set_xticks(range(len(region_names)))
ax.set_xticklabels([r.replace('lh_','L-').replace('rh_','R-') for r in region_names], 
                    rotation=90, fontsize=6)
ax.set_ylabel("Cohen's d (total_strength)")
ax.set_title('Region-level Effect Size: Resected vs Non-resected\n(Red=resected weaker, Green=resected stronger)')
ax.axhline(y=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig4_region_effect_size.png'))
plt.close()

# ========== stratifiedAnalysis ==========
print("\n" + "=" * 60)
print("stratifiedAnalysis")
pathology_groups = {
    'HS': 'Hippocampal Sclerosis',
    'FCD': 'Focal Cortical Dysplasia',
    'DNT': 'Dysembryoplastic Neuroepithelial Tumor',
    'OTHER': 'Other pathologies'
}

strat_results = []
for patho, label in pathology_groups.items():
    sub_df = node_df[node_df['pathology'] == patho]
    if len(sub_df) == 0 or sub_df['resected'].nunique() < 2:
        print(f"  [{label}] insufficient data, skipping")
        continue
    n_subjects = sub_df['subject_id'].nunique()
    n_resected = (sub_df['resected']==1).sum()
    n_total = len(sub_df)
    
    # Statistical comparison
    feat_effects = {}
    for feat in feat_cols:
        r = sub_df[sub_df['resected']==1][feat].dropna()
        nr = sub_df[sub_df['resected']==0][feat].dropna()
        if len(r)>5 and len(nr)>5 and r.std()>0 and nr.std()>0:
            pooled = np.sqrt((r.std()**2 + nr.std()**2)/2)
            d = (r.mean()-nr.mean())/pooled
            feat_effects[feat] = d
    
    # classification
    X = sub_df[feat_cols].values
    y = sub_df['resected'].values
    groups = sub_df['subject_id'].values
    if len(np.unique(groups)) >= 5:
        y_prob = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
            clf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', 
                                          random_state=42, n_jobs=-1)
            clf.fit(Xtr, y[tr])
            y_prob[te] = clf.predict_proba(Xte)[:,1]
        auc = roc_auc_score(y, y_prob)
    else:
        auc = np.nan
    
    top_feat = max(feat_effects, key=lambda k: abs(feat_effects[k])) if feat_effects else 'N/A'
    print(f"  [{label}] n={n_subjects}subjects, Resected={n_resected}/{n_total}, AUC={auc:.3f}, "
          f"topfeature={top_feat}(d={feat_effects.get(top_feat,0):.3f})")
    strat_results.append({'pathology':label,'n_subjects':n_subjects,'n_resected_nodes':n_resected,
                          'auc':auc,'top_feature':top_feat,'top_effect_size':feat_effects.get(top_feat,0)})

pd.DataFrame(strat_results).to_csv(os.path.join(result_dir, "A3_pathology_stratified.csv"), index=False)

# ========== 5: stratifiedAUCcomparison ==========
print("\nGenerating5: stratifiedcomparison...")
if strat_results:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [r['pathology'] for r in strat_results]
    aucs = [r['auc'] for r in strat_results]
    ns = [r['n_subjects'] for r in strat_results]
    bars = ax.bar(range(len(labels)), aucs, color=['#3498db','#e74c3c','#2ecc71','#f39c12'][:len(labels)])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f'{l}\n(n={n})' for l,n in zip(labels,ns)], fontsize=9)
    ax.set_ylabel('AUC')
    ax.set_title('Epileptic Focus Localization by Pathology Type')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_ylim(0, 1)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{auc:.3f}', ha='center', fontsize=10)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig5_pathology_stratified_auc.png'))
    plt.close()

# ========== Summary ==========
print("\n" + "=" * 60)
print("Direction AVisualization+stratifiedcomplete！")
print("=" * 60)
print(f"\nfiguresaving: {fig_dir}")
for f in sorted(os.listdir(fig_dir)):
    print(f"  {f}")
print(f"\nstratifiedresult: A3_pathology_stratified.csv")
