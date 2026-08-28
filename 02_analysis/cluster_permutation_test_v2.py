# -*- coding: utf-8 -*-
"""
Cluster permutation test - uses the same 44 features and PCA settings as the original clustering
"""
import pandas as pd
import numpy as np
import os
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

base_dir = r"D:\epilepsy_MIND"
proc_dir = os.path.join(base_dir, "processed")

# Load data
mind_nets = np.load(os.path.join(proc_dir, "mind_networks.npy"))
thick_aligned = pd.read_csv(os.path.join(proc_dir, "thickness_aligned.csv"), index_col=0)
subject_ids = thick_aligned.index.tolist()
n = 68
half = n // 2
print(f"Subjects: {len(subject_ids)}")

# Feature extraction function (identical to original clustering)
def extract_all_features(network):
    feat = {}
    # Global features
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
    left_net = network[:half, :half]
    right_net = network[half:, half:]
    inter_net = network[:half, half:]
    feat['g_left_strength'] = np.abs(left_net[left_net!=0]).mean() if (left_net!=0).any() else 0
    feat['g_right_strength'] = np.abs(right_net[right_net!=0]).mean() if (right_net!=0).any() else 0
    feat['g_inter_strength'] = np.abs(inter_net[inter_net!=0]).mean() if (inter_net!=0).any() else 0
    feat['g_inter_ratio'] = feat['g_inter_strength'] / (feat['g_left_strength']+feat['g_right_strength']+1e-10)
    # Nodal-level statistics
    pos_net = (network > 0).astype(float)
    neg_net = (network < 0).astype(float)
    pos_deg = pos_net.sum(axis=1)
    neg_deg = neg_net.sum(axis=1)
    pos_str = np.where(network > 0, network, 0).sum(axis=1)
    neg_str = np.where(network < 0, -network, 0).sum(axis=1)
    total_str = pos_str + neg_str
    try:
        bc = nx.betweenness_centrality(G, weight='weight')
        betweenness = np.array([bc[i] for i in range(n)])
    except:
        betweenness = np.zeros(n)
    try:
        ec = nx.eigenvector_centrality_numpy(G, weight='weight')
        eigenvector = np.array([ec[i] for i in range(n)])
    except:
        eigenvector = np.zeros(n)
    for name, arr in [('pos_deg', pos_deg), ('neg_deg', neg_deg),
                        ('pos_str', pos_str), ('neg_str', neg_str),
                        ('total_str', total_str), ('betweenness', betweenness),
                        ('eigenvector', eigenvector)]:
        feat[f'n_{name}_mean'] = arr.mean()
        feat[f'n_{name}_std'] = arr.std()
        feat[f'n_{name}_max'] = arr.max()
        feat[f'n_{name}_asym'] = arr[:half].mean() - arr[half:].mean()
    return feat

print("Extracting 44 features...")
features_list = [extract_all_features(mind_nets[i]) for i in range(len(subject_ids))]
feat_df = pd.DataFrame(features_list, index=subject_ids)
print(f"Feature matrix: {feat_df.shape}")

# Standardization + PCA (identical to original)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(feat_df.values)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA retained: {X_pca.shape[1]} dims (explained variance: {pca.explained_variance_ratio_.sum():.1%})")

# Real clustering
km = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_real = km.fit_predict(X_pca)
sil_real = silhouette_score(X_pca, labels_real)
print(f"Real clustering silhouette = {sil_real:.4f}")

# Permutation test
print("\nStarting permutation test (1000 iterations)...")
n_perm = 1000
sil_perm = []
np.random.seed(42)
for i in range(n_perm):
    if i % 200 == 0:
        print(f"  Permutation {i}/{n_perm}...")
    X_perm = X_pca.copy()
    for j in range(X_perm.shape[1]):
        np.random.shuffle(X_perm[:, j])
    km_perm = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_perm = km_perm.fit_predict(X_perm)
    sil_perm.append(silhouette_score(X_perm, labels_perm))

sil_perm = np.array(sil_perm)
p_value = (sil_perm >= sil_real).sum() / n_perm

print(f"\nPermutation test results:")
print(f"  Real silhouette = {sil_real:.4f}")
print(f"  Random silhouette mean = {sil_perm.mean():.4f}")
print(f"  Random silhouette std = {sil_perm.std():.4f}")
print(f"  Random silhouette max = {sil_perm.max():.4f}")
print(f"  p-value = {p_value:.4f}")

# Save results
os.makedirs(os.path.join(base_dir, "results_supplementary"), exist_ok=True)
with open(os.path.join(base_dir, "results_supplementary", "cluster_permutation_test_v2.txt"), 'w') as f:
    f.write(f"Features: 44 (same as original clustering)\n")
    f.write(f"PCA components: {X_pca.shape[1]}\n")
    f.write(f"Real silhouette = {sil_real:.4f}\n")
    f.write(f"Permutation mean = {sil_perm.mean():.4f}\n")
    f.write(f"Permutation std = {sil_perm.std():.4f}\n")
    f.write(f"Permutation max = {sil_perm.max():.4f}\n")
    f.write(f"p-value = {p_value:.4f}\n")
    f.write(f"n_permutations = {n_perm}\n")

print("\nResults saved.")
