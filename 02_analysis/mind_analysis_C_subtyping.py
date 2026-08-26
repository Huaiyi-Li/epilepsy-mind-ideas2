# -*- coding: utf-8 -*-
"""
Direction C：MIND (clustering)
- feature: MINDGlobal features + nodesfeature
- dimensionality reduction: PCA
- clustering: K-means (k)
- Analysis: feature、feature、differences
"""
import pandas as pd
import numpy as np
import os
import networkx as nx
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.stats import f_oneway, chi2_contingency
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# ========== Configuration ==========
base_dir = r"D:\epilepsy_MIND"
proc_dir = os.path.join(base_dir, "processed")
result_dir = os.path.join(base_dir, "results")
fig_dir = os.path.join(result_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

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
print(f"subjects: {len(subject_ids)}")

# ========== feature ==========
print("\nMINDfeature...")

def extract_all_features(network):
    """MIND+nodesfeature"""
    n = network.shape[0]
    half = n // 2
    feat = {}
    
    # === Global features ===
    nz = network[network != 0]
    feat['g_mean_conn'] = nz.mean() if len(nz)>0 else 0
    feat['g_std_conn'] = nz.std() if len(nz)>0 else 0
    feat['g_pos_ratio'] = (network > 0).sum() / (n*(n-1))
    feat['g_neg_ratio'] = (network < 0).sum() / (n*(n-1))
    feat['g_abs_mean'] = np.abs(network[network!=0]).mean() if (network!=0).any() else 0
    
    G = nx.from_numpy_array(np.abs(network))
    try: feat['g_clustering'] = nx.average_clustering(G, weight='weight')
    except: feat['g_clustering'] = 0
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = greedy_modularity_communities(G, weight='weight')
        feat['g_modularity'] = nx.community.modularity(G, comms, weight='weight')
        feat['g_n_communities'] = len(comms)
    except:
        feat['g_modularity'] = 0
        feat['g_n_communities'] = 0
    
    degs = np.array([d for _,d in G.degree(weight='weight')])
    feat['g_mean_degree'] = degs.mean()
    feat['g_std_degree'] = degs.std()
    feat['g_max_degree'] = degs.max()
    feat['g_degree_cv'] = degs.std() / (degs.mean()+1e-10)
    
    # Hemispheric features
    left_net = network[:half, :half]
    right_net = network[half:, half:]
    inter_net = network[:half, half:]
    feat['g_left_strength'] = np.abs(left_net[left_net!=0]).mean() if (left_net!=0).any() else 0
    feat['g_right_strength'] = np.abs(right_net[right_net!=0]).mean() if (right_net!=0).any() else 0
    feat['g_inter_strength'] = np.abs(inter_net[inter_net!=0]).mean() if (inter_net!=0).any() else 0
    feat['g_inter_ratio'] = feat['g_inter_strength'] / (feat['g_left_strength']+feat['g_right_strength']+1e-10)
    
    # === nodesfeature (mean/standard/largest/) ===
    pos_net = (network > 0).astype(float)
    neg_net = (network < 0).astype(float)
    pos_deg = pos_net.sum(axis=1)
    neg_deg = neg_net.sum(axis=1)
    pos_str = np.where(network > 0, network, 0).sum(axis=1)
    neg_str = np.where(network < 0, -network, 0).sum(axis=1)
    total_str = pos_str + neg_str
    
    # Centrality
    try:
        bc = nx.betweenness_centrality(G, weight='weight')
        betweenness = np.array([bc[i] for i in range(n)])
    except:
        betweenness = np.zeros(n)
    
    # featurevectorCentrality
    try:
        ec = nx.eigenvector_centrality_numpy(G, weight='weight')
        eigenvector = np.array([ec[i] for i in range(n)])
    except:
        eigenvector = np.zeros(n)
    
    # nodes
    for name, arr in [('pos_deg', pos_deg), ('neg_deg', neg_deg), 
                        ('pos_str', pos_str), ('neg_str', neg_str), 
                        ('total_str', total_str), ('betweenness', betweenness),
                        ('eigenvector', eigenvector)]:
        feat[f'n_{name}_mean'] = arr.mean()
        feat[f'n_{name}_std'] = arr.std()
        feat[f'n_{name}_max'] = arr.max()
        #  (left hemispheremean - mean)
        feat[f'n_{name}_asym'] = arr[:half].mean() - arr[half:].mean()
    
    return feat

features_list = [extract_all_features(mind_nets[i]) for i in range(len(subject_ids))]
feat_df = pd.DataFrame(features_list, index=subject_ids)
print(f"featurematrix: {feat_df.shape}")

# ========== standardization + PCAdimensionality reduction ==========
print("\nstandardization + PCAdimensionality reduction...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(feat_df.values)

# PCA95%
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA {X_pca.shape[1]}  (: {pca.explained_variance_ratio_.sum():.1%})")

# ========== Determining optimal cluster number ==========
print("\nDetermining optimal cluster number (k=2~8)...")
k_range = range(2, 9)
silhouette_scores = []
ch_scores = []
inertias = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_pca, labels))
    ch_scores.append(calinski_harabasz_score(X_pca, labels))

best_k_sil = k_range[np.argmax(silhouette_scores)]
best_k_ch = k_range[np.argmax(ch_scores)]
print(f"  Silhouette scorek: {best_k_sil} (score={max(silhouette_scores):.3f})")
print(f"  CHk: {best_k_ch}")

# Silhouette scorek
best_k = best_k_sil
print(f"  k={best_k}")

# ========== 8: clustering ==========
print("\nGenerating8: clustering...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(k_range, inertias, 'bo-')
ax1.set_xlabel('Number of clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')
ax1.axvline(x=best_k, color='red', linestyle='--', alpha=0.5)

ax2.plot(k_range, silhouette_scores, 'ro-', label='Silhouette')
ax2_twin = ax2.twinx()
ax2_twin.plot(k_range, ch_scores, 'g^-', label='CH index')
ax2.set_xlabel('Number of clusters (k)')
ax2.set_ylabel('Silhouette Score', color='red')
ax2_twin.set_ylabel('CH Index', color='green')
ax2.set_title('Cluster Validation')
ax2.axvline(x=best_k, color='red', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig8_cluster_selection.png'))
plt.close()

# ========== clustering ==========
print(f"\nclustering (k={best_k})...")
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = km_final.fit_predict(X_pca)
clin['cluster'] = cluster_labels
feat_df['cluster'] = cluster_labels

for c in range(best_k):
    n = (cluster_labels == c).sum()
    print(f"  {c+1}: {n} subjects ({n/len(cluster_labels):.1%})")

# ========== 9: PCAclusteringScatter plot ==========
print("Generating9: PCAclusteringScatter plot...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = plt.cm.Set1(np.linspace(0, 1, best_k))

# PC1 vs PC2
ax = axes[0]
for c in range(best_k):
    mask = cluster_labels == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[c]], alpha=0.6, s=30, label=f'Subtype {c+1}')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('MIND Network Subtypes (PC1 vs PC2)')
ax.legend()

# PC1 vs PC3
ax = axes[1]
for c in range(best_k):
    mask = cluster_labels == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 2], c=[colors[c]], alpha=0.6, s=30, label=f'Subtype {c+1}')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
ax.set_title('PC1 vs PC3')
ax.legend()

# 
ax = axes[2]
outcome_colors = ['#2ecc71' if o==1 else '#e74c3c' for o in clin['outcome'].values]
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=outcome_colors, alpha=0.6, s=30)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('Colored by Outcome (Green=Good, Red=Poor)')
legend_elements = [Patch(facecolor='#2ecc71', label='Good (ILAE=1)'),
                   Patch(facecolor='#e74c3c', label='Poor (ILAE>1)')]
ax.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig9_cluster_scatter.png'))
plt.close()

# ========== Subtype clinical characteristicsAnalysis ==========
print("\nSubtype clinical characteristicsAnalysis...")
clinical_comparison = []

# variables
continuous_vars = ['Binned_Age_at_Scan', 'Binned_Onset_Age', 'Number_ASMs']
for var in continuous_vars:
    if var in clin.columns:
        groups = [clin[clin['cluster']==c][var].dropna() for c in range(best_k)]
        if all(len(g)>0 for g in groups):
            try:
                f, p = f_oneway(*groups)
                clinical_comparison.append({'variable':var, 'test':'ANOVA', 'p_value':p})
            except: pass

# classificationvariables
categorical_vars = ['Sex', 'Op_Side', 'Pathology', 'SE', 'FUS', 'outcome']
for var in categorical_vars:
    if var in clin.columns:
        contingency = pd.crosstab(clin['cluster'], clin[var])
        if (contingency.sum(axis=1) > 0).all():
            try:
                chi2, p, dof, expected = chi2_contingency(contingency)
                clinical_comparison.append({'variable':var, 'test':'Chi-square', 'p_value':p})
            except: pass

clin_comp_df = pd.DataFrame(clinical_comparison).sort_values('p_value')
print(clin_comp_df.to_string(index=False))
clin_comp_df.to_csv(os.path.join(result_dir, "C1_clinical_comparison.csv"), index=False)

# 
print("\n:")
outcome_by_cluster = pd.crosstab(clin['cluster'], clin['outcome'], normalize='index')
outcome_by_cluster.columns = ['Poor(ILAE>1)', 'Good(ILAE=1)']
print(outcome_by_cluster.to_string())
outcome_by_cluster.to_csv(os.path.join(result_dir, "C1_outcome_by_subtype.csv"))

# ========== Subtype network featurescomparison ==========
print("\nSubtype network featurescomparison (topdifferencesfeature)...")
network_feat_cols = [c for c in feat_df.columns if c != 'cluster']
feat_comparison = []
for col in network_feat_cols:
    groups = [feat_df[feat_df['cluster']==c][col].dropna() for c in range(best_k)]
    if all(len(g)>5 for g in groups):
        try:
            f, p = f_oneway(*groups)
            means = [g.mean() for g in groups]
            feat_comparison.append({'feature':col, 'p_value':p, 'means':means})
        except: pass

feat_comp_df = pd.DataFrame(feat_comparison).sort_values('p_value')
print(f"total {len(feat_comp_df)} featuresignificantdifferences (p<0.05: {(feat_comp_df['p_value']<0.05).sum()})")
print("\nTop 15 differenceslargestfeature:")
for _, row in feat_comp_df.head(15).iterrows():
    means_str = ', '.join([f'S{i+1}={m:.3f}' for i,m in enumerate(row['means'])])
    print(f"  {row['feature']}: p={row['p_value']:.2e} ({means_str})")

feat_comp_df[['feature','p_value']].to_csv(os.path.join(result_dir, "C1_network_feature_comparison.csv"), index=False)

# ========== 10: Subtype clinical characteristicscomparison ==========
print("\nGenerating10: Subtype characteristicscomparison...")
n_top = min(8, len(feat_comp_df))
top_feats = feat_comp_df.head(n_top)['feature'].values

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for idx, feat in enumerate(top_feats):
    ax = axes[idx]
    data = [feat_df[feat_df['cluster']==c][feat].values for c in range(best_k)]
    bp = ax.boxplot(data, tick_labels=[f'S{i+1}' for i in range(best_k)], 
                    patch_artist=True, showfliers=False)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i])
    p_val = feat_comp_df[feat_comp_df['feature']==feat]['p_value'].values[0]
    ax.set_title(f'{feat}\np={p_val:.2e}', fontsize=8)
    ax.set_ylabel(feat, fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig10_subtype_feature_comparison.png'))
plt.close()

# ========== 11: meanMINDHeatmap ==========
print("Generating11: meanMIND...")
fig, axes = plt.subplots(1, best_k, figsize=(5*best_k, 5))
if best_k == 1: axes = [axes]
vmax = np.percentile(np.abs(mind_nets), 95)
for c in range(best_k):
    ax = axes[c]
    mask = cluster_labels == c
    avg_net = mind_nets[mask].mean(axis=0)
    im = ax.imshow(avg_net, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'Subtype {c+1} (n={mask.sum()})')
    ax.set_xlabel('Region')
    ax.set_ylabel('Region')
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig11_subtype_avg_networks.png'))
plt.close()

# ========== savingClustering results ==========
print("\nsavingClustering results...")
clin[['cluster','outcome','Sex','Op_Side','Pathology','ILAE_Year1']].to_csv(
    os.path.join(result_dir, "C1_subtype_labels.csv"))
feat_df.to_csv(os.path.join(result_dir, "C1_subtype_features.csv"))

# ========== Summary ==========
print("\n" + "=" * 60)
print("Direction Ccomplete！")
print("=" * 60)
print(f"\nclustering: k={best_k}")
print(f"Silhouette score: {max(silhouette_scores):.3f}")
print(f"\n:")
for c in range(best_k):
    n = (cluster_labels == c).sum()
    good = clin[(clin['cluster']==c) & (clin['outcome']==1)].shape[0]
    print(f"  {c+1}: {n}subjects, good={good/n:.1%}")

sig_clin = clin_comp_df[clin_comp_df['p_value']<0.05]
print(f"\nsignificantdifferencesClinical variables (p<0.05): {len(sig_clin)}")
if len(sig_clin)>0:
    print(sig_clin.to_string(index=False))

print(f"\nsignificantdifferencesfeature (p<0.05): {(feat_comp_df['p_value']<0.05).sum()}")
print(f"\nresultfile:")
print(f"  C1_subtype_labels.csv")
print(f"  C1_clinical_comparison.csv")
print(f"  C1_outcome_by_subtype.csv")
print(f"  C1_network_feature_comparison.csv")
print(f"\nfigure:")
print(f"  fig8_cluster_selection.png")
print(f"  fig9_cluster_scatter.png")
print(f"  fig10_subtype_feature_comparison.png")
print(f"  fig11_subtype_avg_networks.png")
