# Morphometric Individual Network (MIND) and DTI Connectomics for Epileptogenic Zone Localization in Temporal Lobe Epilepsy

This repository contains the analysis code for the study: **"Morphometric Individual Network and Diffusion Tensor Connectomics for Preoperative Localization of the Epileptogenic Zone in Temporal Lobe Epilepsy: A Multimodal Imaging and Transcriptomic Association Study"**.

## Overview

Drug-resistant temporal lobe epilepsy (TLE) requires accurate preoperative localization of the epileptogenic network for successful surgical resection. This study leverages the large, multicenter IDEAS-II public dataset to:

1. Construct individualized **Morphometric Individual Networks (MIND)** from routine T1-weighted MRI
2. Validate epileptogenic zone localization using surgical resection masks as the reference standard
3. Independently validate findings using **DTI white matter connectomics** (Lausanne-36 parcellation)
4. Integrate imaging phenotypes with **AHBA transcriptomic data** to reveal molecular pathways associated with epilepsy vulnerability
5. Explore patient subtypes based on MIND network features using unsupervised clustering

## Data Source

All imaging and clinical data are from the publicly available **IDEAS-II dataset**:

- **OpenNeuro accession**: [ds007401](https://openneuro.org/datasets/ds007401/versions/1.0.0)
- **Clinical data & resection masks**: [cnnp-lab.com/ideas-data](https://www.cnnp-lab.com/ideas-data)
- **AHBA transcriptomic data**: [Allen Human Brain Atlas](https://human.brain-map.org), preprocessed expression matrix from [French et al. (2015)](https://doi.org/10.6084/m9.figshare.1439749)

## Project Structure

```
epilepsy-mind-ideas2/
├── 01_preprocessing/
│   └── mind_pipeline.py              # Data cleaning, alignment, MIND network construction (68 regions)
├── 02_analysis/
│   ├── mind_analysis_AB.py           # Direction A (localization) + Direction B (outcome prediction)
│   ├── mind_analysis_A_deep.py       # Random Forest classification with GroupKFold, pathology stratification
│   ├── mind_analysis_B_multimodal.py # Multimodal outcome prediction (MIND + clinical)
│   ├── mind_analysis_B_node.py       # Node-level outcome analysis
│   └── mind_analysis_C_subtyping.py  # Unsupervised clustering (k-means, PCA)
├── 03_transcriptome/
│   └── mind_analysis_E_transcriptome_v2.py  # AHBA spatial association + GO/KEGG enrichment
├── 04_dwi/
│   ├── dwi_analysis_localization.py  # DTI connectome localization (Lausanne-36, 82 regions)
│   └── dwi_visualization.py          # DTI figures (node features, MIND vs DTI comparison)
└── 05_visualization/
    ├── mind_analysis_A_viz.py        # MIND figures (network matrix, boxplots, pathology AUC)
    └── regenerate_transcriptome_figs_en.py  # English-labeled transcriptome figures (300 DPI)
```

## Environment

- **Python**: 3.10+
- **Key dependencies**:
  - `numpy`, `pandas`, `scipy` — data processing and statistics
  - `scikit-learn` — machine learning (Random Forest, Logistic Regression, cross-validation)
  - `networkx` — graph theory metrics (betweenness, eigenvector centrality, clustering)
  - `matplotlib` — visualization (Agg backend for headless rendering)
  - `gseapy` — gene set enrichment analysis (GO/KEGG)
  - `python-docx` — manuscript generation (not required for analysis)

See `requirements.txt` for the full list.

## Usage

### 1. Data Preparation

Download the IDEAS-II dataset from OpenNeuro and the accompanying clinical data/resection masks from the project website. Organize the data as follows:

```
D:\癫痫MIND\
├── clinical_metadata\          # Patient metadata (CSV)
├── freesurfer_thickness_volume_area\  # FreeSurfer outputs (aparc_thick/vol/area, aseg_vol)
├── resection_percentage_table\ # Resection proportion table (table_resected.csv)
├── fully_processed_dwi_connectomes\  # DWI connectome matrices
└── ahba_data\                  # AHBA expression matrix (AllenHBA_DK_ExpressionMatrix.tsv)
```

### 2. Run the Pipeline

Execute scripts in numerical order:

```bash
# Step 1: Preprocessing and MIND network construction
python 01_preprocessing/mind_pipeline.py

# Step 2: Main analyses (localization, outcome, subtyping)
python 02_analysis/mind_analysis_AB.py
python 02_analysis/mind_analysis_A_deep.py
python 02_analysis/mind_analysis_C_subtyping.py

# Step 3: Transcriptomic analysis
python 03_transcriptome/mind_analysis_E_transcriptome_v2.py

# Step 4: DTI analysis
python 04_dwi/dwi_analysis_localization.py

# Step 5: Visualization
python 05_visualization/mind_analysis_A_viz.py
python 04_dwi/dwi_visualization.py
python 05_visualization/regenerate_transcriptome_figs_en.py
```

## Key Results

| Analysis | Metric | Value |
|---|---|---|
| MIND localization (overall) | AUC | 0.702 (95% CI: 0.685-0.717) |
| MIND localization (HS subgroup) | AUC | 0.766 |
| MIND patient-level accuracy | % | 85.1% |
| MIND top feature (neg. connection strength) | Cohen's d | -0.64 |
| MIND in DTI subsample (n=191) | AUC | 0.680 |
| DTI localization (overall, n=87) | AUC | 0.627 |
| DTI localization (HS subgroup) | AUC | 0.700 |
| Transcriptomic genes (FDR < 0.05) | Count | 1,571 |
| Clustering stability (bootstrap ARI) | - | 0.909 |
| Top GO enrichment | Chemical synaptic transmission | FDR = 1.17×10⁻⁴ |
| Top KEGG enrichment | GABAergic synapse | FDR = 2.85×10⁻⁴ |

## Methodological Note

This study employs a **simplified region-level MIND** approach (Pearson correlation of region-level aggregated features: cortical thickness, volume, surface area) rather than the strict vertex-level morphometric distribution divergence (MIND) method proposed by Kong et al. (2018). The simplified approach was chosen for computational efficiency and reproducibility across multicenter retrospective data. This limitation is acknowledged in the manuscript.

## Code Revisions (v2)

The following methodological corrections were applied after independent code review:

1. **Betweenness/closeness centrality weight direction** (critical bug fix): NetworkX's `betweenness_centrality(weight=)` and `closeness_centrality(distance=)` interpret edge weights as traversal cost (lower = shorter path). The original code passed raw similarity weights, inverting the direction of these measures. Fixed by constructing a distance matrix `distance = 1 - |similarity|` for these measures. Eigenvector centrality correctly uses similarity weights.

2. **Removed `local_efficiency` feature**: `nx.local_efficiency()` returns a graph-level scalar, not a node-level dictionary. The original code's `try/except` silently set this feature to all zeros. Its feature importance was 0 in all analyses, so it was removed rather than reimplemented.

3. **Corrected Cohen's d formula**: The original formula `sqrt((s1² + s2²)/2)` only applies to equal sample sizes. Replaced with sample-size-weighted pooled SD: `sqrt(((n1-1)s1² + (n2-1)s2²) / (n1+n2-2))`.

4. **Added within-subject paired tests**: To address non-independence of regional nodes within patients, added patient-level paired tests (mean resected - mean non-resected per patient, one-sample t-test). All key findings confirmed at N=430 patient level.

5. **Added patient-level classification evaluation**: Report proportion of patients with higher mean predicted probability in resected than non-resected regions (85.1%).

6. **Added data consistency assertions**: Hemisphere ordering checks (first 34 = lh_, last 34 = rh_), matrix dimension checks.

7. **Unified random_state=42** across all classifiers (LogisticRegression, SVC, RandomForest).

8. **Narrowed exception handling**: Replaced bare `except:` with `except Exception:` and warning messages.

9. **All comments and docstrings converted to English**.

## Citation

If you use this code, please cite:

> [Author names]. Morphometric Individual Network and Diffusion Tensor Connectomics for Preoperative Localization of the Epileptogenic Zone in Temporal Lobe Epilepsy: A Multimodal Imaging and Transcriptomic Association Study. *Human Brain Mapping*, 2026. (in review)

And the original dataset:

> Wang Y, Vos SB, Winston GP, et al. Open diffusion MRI and connectivity data for epilepsy and surgery: The IDEAS II release. *Epilepsia*, 2026.

## License

This code is released under the MIT License. Please adhere to the data use agreements of the IDEAS-II dataset and the Allen Human Brain Atlas.
