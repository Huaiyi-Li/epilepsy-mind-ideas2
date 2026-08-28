import os, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

PROC = r'D:\癫痫MIND\processed'
RESULT = r'D:\癫痫MIND\results_10pct'
clin = pd.read_csv(os.path.join(PROC, 'clinical_aligned.csv'), index_col=0)

DWI_BASE = r'D:\癫痫MIND\fully_processed_dwi_connectomes\networks\deterministic\deterministic_tractography\Lausanne-36'
dwi_subjects = []
for sub_dir in sorted(os.listdir(DWI_BASE)):
    if sub_dir.startswith('sub-'):
        dwi_subjects.append(int(sub_dir.replace('sub-', '')))

mind_ids = [int(x) for x in clin.index]
dti_in_mind = [s for s in dwi_subjects if s in mind_ids]
print(f'DTI subjects in MIND cohort: {len(dti_in_mind)}')

node_df = pd.read_csv(os.path.join(RESULT, 'node_features.csv'))
node_df['subject_id'] = node_df['subject_id'].astype(int)
dti_node = node_df[node_df['subject_id'].isin(dti_in_mind)]
print(f'DTI-subset MIND nodes: {len(dti_node)}')
print(f'Resected: {(dti_node["resected"]==1).sum()}')

feat_cols = ['pos_degree','neg_degree','pos_strength','neg_strength','total_strength',
             'betweenness','closeness','eigenvector','clustering','participation',
             'within_module_strength','left_hemi_strength','right_hemi_strength','inter_hemi_strength']

X = dti_node[feat_cols].values
y = dti_node['resected'].values
groups = dti_node['subject_id'].values

gkf = GroupKFold(n_splits=5)
y_prob = np.zeros(len(y))
for tr, te in gkf.split(X, y, groups):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=2,
                                  class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(Xtr, y[tr])
    y_prob[te] = clf.predict_proba(Xte)[:,1]

auc = roc_auc_score(y, y_prob)
print(f'MIND AUC in DTI-available subset ({len(dti_in_mind)} patients): {auc:.3f}')
