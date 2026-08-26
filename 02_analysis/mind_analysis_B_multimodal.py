# -*- coding: utf-8 -*-
"""
Direction B：MIND + DWI +  → Outcome prediction
- MIND: 68regions (Global features)
- DWI: 233regionsgroup (Global features, Countmatrix)
- : ++
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
import warnings
warnings.filterwarnings('ignore')

# ========== Configuration ==========
base_dir = r"D:\epilepsy_MIND"
proc_dir = os.path.join(base_dir, "processed")
dwi_dir = os.path.join(base_dir, "fully_processed_dwi_connectomes", "networks", 
                         "deterministic", "deterministic_tractography", "Lausanne-125")
result_dir = os.path.join(base_dir, "results")
os.makedirs(result_dir, exist_ok=True)

# ========== loadingMINDdata ==========
print("=" * 60)
print("loadingMINDdata")
mind_nets = np.load(os.path.join(proc_dir, "mind_networks.npy"))
thick_aligned = pd.read_csv(os.path.join(proc_dir, "thickness_aligned.csv"), index_col=0)
clin = pd.read_csv(os.path.join(proc_dir, "clinical_aligned.csv"), index_col=0)
subject_ids = thick_aligned.index.tolist()
clin = clin[~clin.index.duplicated(keep='first')].loc[subject_ids]
print(f"  MINDsubjects: {len(subject_ids)}")

# Label
clin['outcome'] = (clin['ILAE_Year1'] == 1).astype(int)
print(f"  good(seizure-free): {(clin['outcome']==1).sum()}, poor: {(clin['outcome']==0).sum()}")

# ========== ComputingMINDGlobal features ==========
print("\nComputingMINDfeature...")
def compute_global_features(network):
    n = network.shape[0]
    f = {}
    nz = network[network != 0]
    f['mind_mean_conn'] = nz.mean() if len(nz)>0 else 0
    f['mind_std_conn'] = nz.std() if len(nz)>0 else 0
    f['mind_pos_ratio'] = (network > 0).sum() / (n*(n-1))
    f['mind_neg_ratio'] = (network < 0).sum() / (n*(n-1))
    G = nx.from_numpy_array(np.abs(network))
    try: f['mind_global_efficiency'] = nx.global_efficiency(G)
    except: f['mind_global_efficiency'] = 0
    try: f['mind_avg_clustering'] = nx.average_clustering(G, weight='weight')
    except: f['mind_avg_clustering'] = 0
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = greedy_modularity_communities(G, weight='weight')
        f['mind_modularity'] = nx.community.modularity(G, comms, weight='weight')
    except: f['mind_modularity'] = 0
    try: f['mind_char_path'] = nx.average_shortest_path_length(G, weight='weight')
    except: f['mind_char_path'] = 0
    degs = np.array([d for _,d in G.degree(weight='weight')])
    f['mind_mean_degree'] = degs.mean()
    f['mind_std_degree'] = degs.std()
    half = n//2
    f['mind_inter_hemi'] = np.abs(network[:half,half:][network[:half,half:]!=0]).mean() if (network[:half,half:]!=0).any() else 0
    return f

mind_global = pd.DataFrame([compute_global_features(mind_nets[i]) for i in range(len(subject_ids))], 
                            index=subject_ids)
print(f"  MINDGlobal features: {mind_global.shape}")

# ========== loadingDWIdata ==========
print("\n" + "=" * 60)
print("DWIgroupdata...")
dwi_files = {}
for subj_dir in os.listdir(dwi_dir):
    if not subj_dir.startswith('sub-'): continue
    subj_id = subj_dir.replace('sub-', '')
    ses_dirs = sorted([d for d in os.listdir(os.path.join(dwi_dir, subj_dir)) if d.startswith('ses-')])
    if not ses_dirs: continue
    # session
    ses = ses_dirs[0]
    dwi_subdir = os.path.join(dwi_dir, subj_dir, ses, 'dwi')
    if not os.path.exists(dwi_subdir): continue
    count_files = [f for f in os.listdir(dwi_subdir) if 'Count.csv' in f and 'CountScaled' not in f]
    if count_files:
        dwi_files[subj_id] = os.path.join(dwi_subdir, count_files[0])

print(f"  DWIdatasubjects: {len(dwi_files)}")

# ComputingDWIGlobal features
print("ComputingDWIfeature...")
dwi_global = {}
for subj_id, fpath in dwi_files.items():
    try:
        mat = np.loadtxt(fpath, delimiter=',')
        n = mat.shape[0]
        f = {}
        # 
        nz = mat[mat > 0]
        f['dwi_mean_conn'] = nz.mean() if len(nz)>0 else 0
        f['dwi_std_conn'] = nz.std() if len(nz)>0 else 0
        f['dwi_density'] = (mat > 0).sum() / (n*(n-1))
        f['dwi_total_strength'] = mat.sum() / 2
        #  (path)
        G = nx.from_numpy_array(mat)
        try: f['dwi_avg_clustering'] = nx.average_clustering(G, weight='weight')
        except: f['dwi_avg_clustering'] = 0
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            comms = greedy_modularity_communities(G, weight='weight')
            f['dwi_modularity'] = nx.community.modularity(G, comms, weight='weight')
            f['dwi_n_communities'] = len(comms)
        except:
            f['dwi_modularity'] = 0
            f['dwi_n_communities'] = 0
        # 
        degs = np.array([d for _,d in G.degree(weight='weight')])
        f['dwi_mean_degree'] = degs.mean()
        f['dwi_std_degree'] = degs.std()
        f['dwi_max_degree'] = degs.max()
        #  (coefficient)
        f['dwi_degree_cv'] = degs.std() / (degs.mean() + 1e-10)
        #  assortativity ()
        try: f['dwi_assortativity'] = nx.degree_assortativity_coefficient(G, weight='weight')
        except: f['dwi_assortativity'] = 0
        dwi_global[subj_id] = f
    except Exception as e:
        print(f"  warning: subjects{subj_id}loading: {e}")

dwi_df = pd.DataFrame(dwi_global).T
dwi_df.index.name = 'ID'
print(f"  DWIGlobal features: {dwi_df.shape}")

# ========== alignment ==========
print("\n" + "=" * 60)
print("alignmentsubjectsID")
# ID
mind_global.index = mind_global.index.astype(str)
clin.index = clin.index.astype(str)
dwi_df.index = dwi_df.index.astype(str)

common = set(mind_global.index) & set(clin.index) & set(dwi_df.index)
common = sorted(list(common), key=lambda x: int(x))
print(f"  totalsubjects: {len(common)}")

mind_g = mind_global.loc[common]
clin_g = clin.loc[common]
dwi_g = dwi_df.loc[common]

# One-hot encoding for clinical variables
clin_vars = ['Sex','Op_Side','Pathology','Binned_Age_at_Scan','Binned_Onset_Age','SE','FUS']
clin_dum = pd.get_dummies(clin_g[clin_vars], drop_first=True)

print(f"  MINDfeature: {mind_g.shape[1]}")
print(f"  DWIfeature: {dwi_g.shape[1]}")
print(f"  Clinical variables: {clin_dum.shape[1]}")
print(f"  : good={(clin_g['outcome']==1).sum()}, poor={(clin_g['outcome']==0).sum()}")

# ========== Outcome prediction ==========
print("\n" + "=" * 60)
print("Outcome prediction (5CV, Random Forest)")
y = clin_g['outcome'].values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate(X, y, label):
    y_prob = np.zeros(len(y))
    y_pred = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        clf = RandomForestClassifier(n_estimators=300, max_depth=8, 
                                      class_weight='balanced', random_state=42, n_jobs=-1)
        clf.fit(Xtr, y[tr])
        y_prob[te] = clf.predict_proba(Xte)[:,1]
        y_pred[te] = clf.predict(Xte)
    auc = roc_auc_score(y, y_prob)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    sens = cm[1,1]/(cm[1,0]+cm[1,1]) if (cm[1,0]+cm[1,1])>0 else 0
    spec = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1])>0 else 0
    print(f"  [{label}] AUC={auc:.3f} Acc={acc:.3f} Sens={sens:.3f} Spec={spec:.3f}")
    return {'model':label,'auc':auc,'accuracy':acc,'sensitivity':sens,'specificity':spec}

X_mind = mind_g.values
X_dwi = dwi_g.values
X_clin = clin_dum.values
X_mind_dwi = np.hstack([X_mind, X_dwi])
X_mind_clin = np.hstack([X_mind, X_clin])
X_dwi_clin = np.hstack([X_dwi, X_clin])
X_all = np.hstack([X_mind, X_dwi, X_clin])

results = []
results.append(evaluate(X_mind, y, "MIND"))
results.append(evaluate(X_dwi, y, "DWI"))
results.append(evaluate(X_clin, y, ""))
results.append(evaluate(X_mind_dwi, y, "MIND+DWI"))
results.append(evaluate(X_mind_clin, y, "MIND+"))
results.append(evaluate(X_dwi_clin, y, "DWI+"))
results.append(evaluate(X_all, y, "MIND+DWI+()"))

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(result_dir, "B2_multimodal_outcome.csv"), index=False)

# ========== Feature importance ==========
print("\n" + "=" * 60)
print("Combined model feature importance (top 15)")
all_feat_names = list(mind_g.columns) + list(dwi_g.columns) + list(clin_dum.columns)
sc = StandardScaler(); Xs = sc.fit_transform(X_all)
rf = RandomForestClassifier(n_estimators=500, max_depth=8, class_weight='balanced', 
                             random_state=42, n_jobs=-1)
rf.fit(Xs, y)
imp = pd.DataFrame({'feature':all_feat_names, 'importance':rf.feature_importances_})
imp['modality'] = imp['feature'].apply(lambda x: 'MIND' if x.startswith('mind_') else ('DWI' if x.startswith('dwi_') else ''))
imp = imp.sort_values('importance', ascending=False)
print(imp.head(15).to_string(index=False))
imp.to_csv(os.path.join(result_dir, "B2_multimodal_feature_importance.csv"), index=False)

# 
print("\n:")
mod_imp = imp.groupby('modality')['importance'].sum().sort_values(ascending=False)
print(mod_imp.to_string())

# ========== Summary ==========
print("\n" + "=" * 60)
print("Direction BAnalysis complete！")
print("=" * 60)
print(f"\ntotalsubjects: {len(common)}")
print(f"resultfile: B2_multimodal_outcome.csv, B2_multimodal_feature_importance.csv")
print(f"\n: {res_df.loc[res_df['auc'].idxmax(), 'model']} (AUC={res_df['auc'].max():.3f})")
