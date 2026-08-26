# -*- coding: utf-8 -*-
"""
IDEAS-II Epilepsy MIND Analysis Pipeline - Step 1: Data Cleaning + MIND Network Construction
Input: Raw data under D:\\epilepsy_MIND
Output: Aligned data and MIND networks under D:\\epilepsy_MIND\\processed
"""
import pandas as pd
import numpy as np
import os
from scipy.stats import zscore

# ========== Configuration ==========
base_dir = r"D:\epilepsy_MIND"
output_dir = os.path.join(base_dir, "processed")
os.makedirs(output_dir, exist_ok=True)

# ========== Utility Functions ==========
def get_base_region(col_name):
    """Extract base region name from column name, removing _thickness/_volume/_area suffix"""
    for suffix in ['_thickness', '_volume', '_area', '_thick']:
        if col_name.endswith(suffix):
            return col_name[:-len(suffix)]
    return col_name

def is_specific_region(col_name):
    """Check if column is a specific brain region (exclude summary/whole-brain measures)"""
    exclude_keywords = ['MeanThickness', 'BrainSegVol', 'eTIV',
                        'aparc.thickness', 'aparc.volume', 'aparc.area',
                        'aparc', 'BrainSeg', 'EstimatedTotalIntraCranial']
    for kw in exclude_keywords:
        if kw in col_name:
            return False
    # Must start with lh_ or rh_
    return col_name.startswith('lh_') or col_name.startswith('rh_')

# ========== 1. Load Clinical Data ==========
print("=" * 60)
print("1. Loading clinical data")
clin_path = os.path.join(base_dir, "clinical_metadata", "Metadata_Release_Anon.csv")
clin = pd.read_csv(clin_path)
clin['ID'] = clin['ID'].astype(str)
print(f"  Clinical data: {clin.shape[0]} subjects, {clin.shape[1]} variables")
print(f"  Outcome columns: ILAE_Year1~5 (1=seizure-free, higher=worse)")

# ========== 2. Load FreeSurfer Morphometric Features ==========
print("\n" + "=" * 60)
print("2. Loading FreeSurfer morphometric features (transposed to subjects x regions)")

def load_freesurfer(filepath, label):
    df = pd.read_csv(filepath, sep='\t', index_col=0)
    df_t = df.T
    df_t.index = df_t.index.astype(str)
    df_t.index.name = 'ID'
    print(f"  {label}: {df_t.shape[0]} subjects x {df_t.shape[1]} columns (incl. summary)")
    return df_t

thick = load_freesurfer(os.path.join(base_dir, "freesurfer_thickness_volume_area", "aparc_thick.txt"), "Cortical thickness")
vol   = load_freesurfer(os.path.join(base_dir, "freesurfer_thickness_volume_area", "aparc_vol.txt"),   "Gray matter volume")
area  = load_freesurfer(os.path.join(base_dir, "freesurfer_thickness_volume_area", "aparc_area.txt"),  "Surface area")
aseg  = load_freesurfer(os.path.join(base_dir, "freesurfer_thickness_volume_area", "aseg_vol.txt"),    "Subcortical volume")

# ========== 2b. Filter Specific Regions and Take Intersection ==========
print("\n  Filtering specific regions (excluding MeanThickness/eTIV etc.)...")
thick_specific = [c for c in thick.columns if is_specific_region(c)]
vol_specific   = [c for c in vol.columns   if is_specific_region(c)]
area_specific  = [c for c in area.columns  if is_specific_region(c)]
print(f"    Thickness: {len(thick_specific)} specific regions")
print(f"    Volume: {len(vol_specific)} specific regions")
print(f"    Area: {len(area_specific)} specific regions")

# Extract base name mapping
thick_map = {get_base_region(c): c for c in thick_specific}
vol_map   = {get_base_region(c): c for c in vol_specific}
area_map  = {get_base_region(c): c for c in area_specific}

# Take common base region names
common_bases = sorted(set(thick_map.keys()) & set(vol_map.keys()) & set(area_map.keys()))
print(f"    Common specific regions: {len(common_bases)}")

# Keep only common regions, unify column names to base names
thick = thick[[thick_map[b] for b in common_bases]]
vol   = vol[[vol_map[b]   for b in common_bases]]
area  = area[[area_map[b] for b in common_bases]]
thick.columns = common_bases
vol.columns   = common_bases
area.columns  = common_bases

print(f"\n  Aligned cortical regions ({len(common_bases)}):")
for i, b in enumerate(common_bases):
    print(f"    {i+1:2d}. {b}")

# ========== 3. Load Resection Percentage Table ==========
print("\n" + "=" * 60)
print("3. Loading resection percentage table (transposed to subjects x regions)")
resect_path = os.path.join(base_dir, "resection_percentage_table", "table_resected.csv")
resect = pd.read_csv(resect_path, index_col=0)
resect_t = resect.T
resect_t.index = resect_t.index.astype(str)
resect_t.index.name = 'ID'
resect_t.columns = [col.strip("'") for col in resect_t.columns]
print(f"  Resection table: {resect_t.shape[0]} subjects x {resect_t.shape[1]} regions")

# ========== 4. Align Subject IDs (Intersection) ==========
print("\n" + "=" * 60)
print("4. Aligning subject IDs")
common_ids = (set(clin['ID']) & set(thick.index) & set(vol.index) &
              set(area.index) & set(aseg.index) & set(resect_t.index))
common_ids = sorted(list(common_ids), key=lambda x: int(x))
print(f"  Common subjects: {len(common_ids)}")

clin     = clin[clin['ID'].isin(common_ids)].set_index('ID').loc[common_ids]
thick    = thick.loc[common_ids]
vol      = vol.loc[common_ids]
area     = area.loc[common_ids]
aseg     = aseg.loc[common_ids]
resect_t = resect_t.loc[common_ids]
print(f"  After alignment: {clin.shape[0]} subjects x {thick.shape[1]} cortical regions")

# ========== 5. Build MIND Networks ==========
print("\n" + "=" * 60)
print("5. Building individualized MIND networks")
print("  Method: per-region features=[thickness,volume,area] -> zscore -> Pearson correlation between regions")
print("  Note: Simplified region-level MIND; strict MIND requires vertex-level distribution")

def build_mind_networks(thick_df, vol_df, area_df):
    n_subj = len(thick_df)
    n_reg  = len(thick_df.columns)
    networks = np.zeros((n_subj, n_reg, n_reg))

    for i in range(n_subj):
        feat = np.column_stack([
            thick_df.iloc[i].values.astype(float),
            vol_df.iloc[i].values.astype(float),
            area_df.iloc[i].values.astype(float),
        ])
        feat = zscore(feat, axis=0, nan_policy='omit')
        # Handle potential NaN with nan_to_num
        feat = np.nan_to_num(feat, nan=0.0)
        sim = np.corrcoef(feat)
        sim = np.nan_to_num(sim, nan=0.0)
        np.fill_diagonal(sim, 0)
        networks[i] = sim

    return networks

mind_nets = build_mind_networks(thick, vol, area)
print(f"  MIND network matrix: {mind_nets.shape} (subjects x regions x regions)")
print(f"  Network value range: [{mind_nets.min():.3f}, {mind_nets.max():.3f}]")
nonzero = mind_nets[mind_nets != 0]
print(f"  Non-zero connection mean strength: {nonzero.mean():.3f} (total {len(nonzero)} edges)")

# ========== 6. Save Results ==========
print("\n" + "=" * 60)
print("6. Saving results ->", output_dir)

clin.to_csv(os.path.join(output_dir, "clinical_aligned.csv"))
thick.to_csv(os.path.join(output_dir, "thickness_aligned.csv"))
vol.to_csv(os.path.join(output_dir, "volume_aligned.csv"))
area.to_csv(os.path.join(output_dir, "area_aligned.csv"))
aseg.to_csv(os.path.join(output_dir, "aseg_aligned.csv"))
resect_t.to_csv(os.path.join(output_dir, "resection_aligned.csv"))
np.save(os.path.join(output_dir, "mind_networks.npy"), mind_nets)

with open(os.path.join(output_dir, "region_names.txt"), 'w') as f:
    f.write('\n'.join(common_bases))

print(f"  clinical_aligned.csv      ({clin.shape[0]} subjects)")
print(f"  thickness/volume/area/aseg_aligned.csv")
print(f"  resection_aligned.csv     ({resect_t.shape[1]} regions)")
print(f"  mind_networks.npy         {mind_nets.shape}")
print(f"  region_names.txt          ({len(common_bases)} regions)")

# ========== 7. Data Summary ==========
print("\n" + "=" * 60)
print("7. Data summary")
print(f"  Total subjects: {len(common_ids)}")
print(f"  Surgery side:")
print(clin['Op_Side'].value_counts().to_string())
print(f"\n  Pathology type (top 5):")
print(clin['Pathology'].value_counts().head(5).to_string())
print(f"\n  ILAE_Year1:")
print(clin['ILAE_Year1'].value_counts().sort_index().to_string())

resect_vals = resect_t.values.astype(float)
resect_vals = np.nan_to_num(resect_vals, nan=0.0)
total_resect = resect_vals.sum(axis=1)
print(f"\n  Total resection ratio per subject: mean={total_resect.mean():.3f}, median={np.median(total_resect):.3f}")
print(f"  Subjects with resection (>0): {(total_resect > 0).sum()} / {len(total_resect)}")

print("\n" + "=" * 60)
print("Step 1 complete!")
print("Next: Direction A (EZ localization) + Direction B (outcome prediction)")
print("=" * 60)
