"""
DWI Lausanne-36 (82regions) EZ localizationAnalysis
- readingsubjectsDWIConnectome matrix
- Computingnodesfeature
- ResectedvsNon-resectedStatistical comparison
- classification
- MINDresultcomparison
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# pathConfiguration
DWI_BASE = r'D:\epilepsy_MIND\fully_processed_dwi_connectomes\networks\deterministic\deterministic_tractography\Lausanne-36'
CLINICAL_FILE = r'D:\epilepsy_MIND\processed\clinical_aligned.csv'
RESECTION_FILE = r'D:\epilepsy_MIND\resection_percentage_table\table_resected.csv'
OUTPUT_DIR = r'D:\epilepsy_MIND\results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. readingClinical data
print("=== readingClinical data ===")
clinical = pd.read_csv(CLINICAL_FILE)
clinical['ID'] = clinical['ID'].astype(int)
print(f"subjects: {len(clinical)}")
print(f":\n{clinical['Op_Side'].value_counts()}")
print(f"Pathology type:\n{clinical['Pathology'].value_counts().head()}")

# 2. readingResection table
print("\n=== readingResection table ===")
resection = pd.read_csv(RESECTION_FILE, index_col=0)
print(f"Resection table: {resection.shape}")
print(f"regions: {len(resection.index)}")
# （removing）
resection.columns = [c.strip().strip("'") for c in resection.columns]
# 
for col in resection.columns:
    resection[col] = pd.to_numeric(resection[col], errors='coerce')

# regionslist
region_names = resection.index.tolist()
print(f"10regions: {region_names[:10]}")
print(f"10regions: {region_names[-10:]}")

# 3. readingsubjectsDWIConnectome matrix
print("\n=== readingDWIConnectome matrix ===")
dwi_matrices = {}
dwi_subjects = []

for sub_dir in sorted(os.listdir(DWI_BASE)):
    if not sub_dir.startswith('sub-'):
        continue
    sub_id = int(sub_dir.replace('sub-', ''))
    ses_dir = os.path.join(DWI_BASE, sub_dir, 'ses-1', 'dwi')
    if not os.path.exists(ses_dir):
        continue
    # Countmatrix
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
        # 
        n = min(mat.shape)
        mat = mat[:n, :n]
        # 
        mat = (mat + mat.T) / 2
        # removing
        np.fill_diagonal(mat, 0)
        dwi_matrices[sub_id] = mat
        dwi_subjects.append(sub_id)
    except Exception as e:
        print(f"  reading sub-{sub_id} : {e}")

print(f"readingDWIsubjects: {len(dwi_matrices)}")
print(f"matrix: {list(dwi_matrices.values())[0].shape}")

# 4. Clinical dataalignment
print("\n=== dataalignment ===")
common_subjects = [s for s in dwi_subjects if s in clinical['ID'].values]
print(f"DWI+Common subjects: {len(common_subjects)}")

# Resection tablealignment
common_with_resection = [s for s in common_subjects if str(s) in resection.columns]
print(f"DWI++dataCommon subjects: {len(common_with_resection)}")

# 5. Computingnodesfeature
print("\n=== Computingnodesfeature ===")
n_regions = list(dwi_matrices.values())[0].shape[0]
print(f"regions: {n_regions}")

# Feature list
feature_names = ['degree', 'strength', 'betweenness', 'eigenvector', 'clustering', 'local_efficiency']

# subjectsNodal features
all_node_features = []
all_node_labels = []  # 1=Resected, 0=Non-resected
all_subject_ids = []
all_region_indices = []

for sub_id in common_with_resection:
    mat = dwi_matrices[sub_id]
    n = mat.shape[0]
    
    # Binarization（median）
    threshold = np.median(mat[mat > 0]) if np.any(mat > 0) else 0
    binary_mat = (mat > threshold).astype(int)
    
    # NetworkX
    G = nx.from_numpy_array(binary_mat)
    
    # Computingfeature
    degree = np.array([G.degree(i) for i in range(n)])
    strength = mat.sum(axis=1)
    betweenness = np.array([nx.betweenness_centrality(G).get(i, 0) for i in range(n)])
    try:
        eigenvector = np.array([nx.eigenvector_centrality_numpy(G).get(i, 0) for i in range(n)])
    except:
        eigenvector = np.zeros(n)
    clustering = np.array([nx.clustering(G).get(i, 0) for i in range(n)])
    try:
        local_eff = np.array([nx.local_efficiency(G).get(i, 0) for i in range(n)])
    except:
        local_eff = np.zeros(n)
    
    # ResectedLabel（Resection table）
    # ：Resection table82regions，DWImatrix81regions，alignment
    resection_col = resection[str(sub_id)].values
    n_resection = len(resection_col)
    n_min = min(n, n_resection)
    
    # n_minregions
    is_resected = (resection_col[:n_min] > 0).astype(int)
    
    # 
    for i in range(n_min):
        all_node_features.append([
            degree[i], strength[i], betweenness[i], eigenvector[i],
            clustering[i], local_eff[i]
        ])
        all_node_labels.append(is_resected[i])
        all_subject_ids.append(sub_id)
        all_region_indices.append(i)

X_nodes = np.array(all_node_features)
y_nodes = np.array(all_node_labels)
print(f"nodes: {len(y_nodes)}")
print(f"Resectednodes: {np.sum(y_nodes)} ({np.mean(y_nodes)*100:.1f}%)")
print(f"Non-resectednodes: {np.sum(y_nodes==0)}")

# 6. ResectedvsNon-resectedStatistical comparison
print("\n=== ResectedvsNon-resectedStatistical comparison ===")
results_stats = []
for j, feat_name in enumerate(feature_names):
    resected_vals = X_nodes[y_nodes==1, j]
    non_resected_vals = X_nodes[y_nodes==0, j]
    
    # Cohen's d
    pooled_std = np.sqrt((np.std(resected_vals)**2 + np.std(non_resected_vals)**2) / 2)
    cohens_d = (np.mean(resected_vals) - np.mean(non_resected_vals)) / pooled_std if pooled_std > 0 else 0
    
    # Mann-Whitney U
    stat, pval = stats.mannwhitneyu(resected_vals, non_resected_vals, alternative='two-sided')
    
    results_stats.append({
        'feature': feat_name,
        'resected_mean': np.mean(resected_vals),
        'non_resected_mean': np.mean(non_resected_vals),
        'cohens_d': cohens_d,
        'p_value': pval
    })
    print(f"{feat_name:20s}: Resected={np.mean(resected_vals):.4f}, Non-resected={np.mean(non_resected_vals):.4f}, d={cohens_d:.3f}, p={pval:.2e}")

df_stats = pd.DataFrame(results_stats)
df_stats.to_csv(os.path.join(OUTPUT_DIR, 'F1_dwi_node_stats.csv'), index=False)

# 7. classification（nodes：ResectedvsNon-resected）
print("\n=== classification（nodes） ===")
# standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_nodes)

# 
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = cross_val_score(rf, X_scaled, y_nodes, cv=cv, scoring='roc_auc')
acc_scores = cross_val_score(rf, X_scaled, y_nodes, cv=cv, scoring='accuracy')

print(f" AUC: {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
print(f" accuracy: {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")

# Feature importance
rf.fit(X_scaled, y_nodes)
importances = rf.feature_importances_
print("\nFeature importance:")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"  {name:20s}: {imp:.4f}")

# savingclassificationresult
df_clf = pd.DataFrame({
    'model': ['RandomForest'],
    'auc_mean': [np.mean(auc_scores)],
    'auc_std': [np.std(auc_scores)],
    'acc_mean': [np.mean(acc_scores)],
    'acc_std': [np.std(acc_scores)]
})
df_clf.to_csv(os.path.join(OUTPUT_DIR, 'F2_dwi_classification.csv'), index=False)

# 8. Pathology typestratifiedAnalysis
print("\n=== Pathology typestratifiedAnalysis ===")
clinical_subset = clinical[clinical['ID'].isin(common_with_resection)]
pathology_counts = clinical_subset['Pathology'].value_counts()
print(f"Pathology type:\n{pathology_counts.head()}")

# Pathology type
main_pathologies = pathology_counts[pathology_counts >= 30].index.tolist()
print(f"sample>=30Pathology type: {main_pathologies}")

results_pathology = []
for path in main_pathologies:
    path_subjects = clinical_subset[clinical_subset['Pathology'] == path]['ID'].values
    path_mask = np.isin(all_subject_ids, path_subjects)
    
    if np.sum(path_mask) < 100:
        continue
    
    X_path = X_scaled[path_mask]
    y_path = y_nodes[path_mask]
    
    if len(np.unique(y_path)) < 2:
        continue
    
    try:
        auc_path = cross_val_score(rf, X_path, y_path, cv=3, scoring='roc_auc')
        results_pathology.append({
            'pathology': path,
            'n_subjects': len(path_subjects),
            'n_nodes': len(y_path),
            'auc_mean': np.mean(auc_path),
            'auc_std': np.std(auc_path)
        })
        print(f"{path:30s}: n={len(path_subjects):3d}subjects, AUC={np.mean(auc_path):.3f}")
    except Exception as e:
        print(f"{path}: Analysis - {e}")

df_path = pd.DataFrame(results_pathology)
df_path.to_csv(os.path.join(OUTPUT_DIR, 'F3_dwi_pathology_stratified.csv'), index=False)

# 9. MINDresultcomparison
print("\n=== MINDresultcomparison ===")
print("MINDresult（68regions）:")
print("   RandomForest AUC: 0.685")
print("  HSsubgroup AUC: 0.742")
print("  Resected d=-0.56")
print("  ResectedfeaturevectorCentrality d=-0.45")
print()
print("DWIresult（82regions）:")
print(f"   RandomForest AUC: {np.mean(auc_scores):.3f}")
print(f"  Resectedfeature: {df_stats.loc[df_stats['cohens_d'].abs().idxmax(), 'feature']} (d={df_stats['cohens_d'].abs().max():.3f})")

print("\n=== Analysis complete ===")
print(f"resultsaving: {OUTPUT_DIR}")
