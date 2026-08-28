# Region-Level Morphometric Similarity Networks and DTI Connectomics for Preoperative Epileptogenic Zone Localization in Temporal Lobe Epilepsy

This repository contains the analysis code for the study: **"Region-Level Morphometric Similarity Networks and DTI Connectomics for Preoperative Epileptogenic Zone Localization in Temporal Lobe Epilepsy"**.

## Overview

Drug-resistant temporal lobe epilepsy (TLE) requires accurate preoperative localization of the epileptogenic network for successful surgical resection. This study leverages the large, multicenter IDEAS-II public dataset to:

1. Construct individualized **region-level morphometric similarity networks (simplified MIND)** from routine T1-weighted MRI
2. Validate epileptogenic zone localization using surgical resection masks as the reference standard
3. Independently validate findings using **DTI white matter connectomics** (Lausanne-36 parcellation)
4. Integrate imaging phenotypes with **AHBA transcriptomic data** to reveal molecular pathways associated with epilepsy vulnerability
5. Explore patient subtypes based on network features using unsupervised clustering
6. Conduct sensitivity analyses (resection threshold, HS-subgroup transcriptomics, cluster permutation test)

## Data Source

All imaging and clinical data are from the publicly available **IDEAS-II dataset**:

- **OpenNeuro accession**: [ds007401](https://openneuro.org/datasets/ds007401/versions/1.0.0)
- **Clinical data & resection masks**: [cnnp-lab.com/ideas-data](https://www.cnnp-lab.com/ideas-data)
- **AHBA transcriptomic data**: [Allen Human Brain Atlas](https://human.brain-map.org), preprocessed expression matrix from [French et al. (2015)](https://doi.org/10.6084/m9.figshare.1439749)

## Project Structure

```
epilepsy-mind-ideas2/
├── 01_preprocessing/
│   └── mind_pipeline.py              # Data cleaning, alignment, simplified MIND network construction (68 regions)
├── 02_analysis/
│   ├── mind_analysis_AB.py           # Direction A (localization) + Direction B (outcome prediction)
│   ├── mind_analysis_A_deep.py       # Random Forest classification with GroupKFold, pathology stratification
│   ├── mind_analysis_B_multimodal.py # Multimodal outcome prediction (MIND + clinical)
│   ├── mind_analysis_B_node.py       # Node-level outcome analysis
│   ├── mind_analysis_C_subtyping.py  # Unsupervised clustering (k-means, PCA)
│   ├── paired_analysis.py            # Within-subject paired tests (resected vs non-resected)
│   ├── threshold_sensitivity_analysis.py  # Resection threshold sensitivity (5%/10%/15%/20%)
│   ├── cluster_permutation_test_v2.py     # Permutation test for clustering significance
│   └── mind_in_dti_subset.py         # MIND re-analysis in DTI subsample for fair comparison
├── 03_transcriptome/
│   ├── mind_analysis_E_transcriptome_v2.py  # AHBA spatial association + GO/KEGG enrichment
│   └── transcriptome_HS_subgroup.py  # HS-subgroup sensitivity analysis (surgical access control)
├── 04_dwi/
│   ├── dwi_analysis_localization.py  # DTI connectome localization (Lausanne-36, 82 regions)
│   ├── dwi_visualization.py          # DTI figures (node features, MIND vs DTI comparison)
│   └── dti_baseline_comparison.py    # DTI-included vs excluded baseline comparison
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
  - `statsmodels` — multiple comparison correction (FDR)

See `requirements.txt` for the full list.

## Usage

### 1. Data Preparation

Download the IDEAS-II dataset from OpenNeuro and the accompanying clinical data/resection masks from the project website.

### 2. Run the Pipeline

Execute scripts in numerical order:

```bash
# Step 1: Preprocessing and simplified MIND network construction
python 01_preprocessing/mind_pipeline.py

# Step 2: Main analyses (localization, outcome, subtyping)
python 02_analysis/mind_analysis_AB.py
python 02_analysis/mind_analysis_A_deep.py
python 02_analysis/mind_analysis_C_subtyping.py
python 02_analysis/paired_analysis.py

# Step 3: Sensitivity analyses
python 02_analysis/threshold_sensitivity_analysis.py
python 02_analysis/cluster_permutation_test_v2.py

# Step 4: Transcriptomic analysis
python 03_transcriptome/mind_analysis_E_transcriptome_v2.py
python 03_transcriptome/transcriptome_HS_subgroup.py

# Step 5: DTI analysis
python 04_dwi/dwi_analysis_localization.py
python 04_dwi/dti_baseline_comparison.py
python 02_analysis/mind_in_dti_subset.py

# Step 6: Visualization
python 05_visualization/mind_analysis_A_viz.py
python 04_dwi/dwi_visualization.py
python 05_visualization/regenerate_transcriptome_figs_en.py
```

## Key Results

| Analysis | Metric | Value |
|---|---|---|
| Simplified MIND localization (overall, n=442) | AUC | 0.750 (95% CI: 0.739-0.761) |
| Simplified MIND localization (HS subgroup, n=215) | AUC | 0.765 |
| Simplified MIND localization (FCD subgroup, n=40) | AUC | 0.523 (near-chance) |
| Simplified MIND patient-level accuracy | % | 86.8% (362/417) |
| Simplified MIND top feature (closeness centrality) | Cohen's d (within-subject) | -1.05 |
| Simplified MIND in DTI subsample (n=191) | AUC | 0.733 |
| DTI localization (overall, n=87) | AUC | 0.707 (95% CI: 0.684-0.730) |
| DTI patient-level accuracy | % | 81.9% (68/83) |
| DTI top feature (eigenvector centrality) | Cohen's d (within-subject) | -0.526 |
| Transcriptomic genes (FDR < 0.05) | Count | 1,571 |
| Clustering stability (bootstrap ARI) | - | 0.909 |
| Clustering permutation test | p-value | 0.159 (not significant) |
| Top GO enrichment | Chemical synaptic transmission | FDR = 1.17×10⁻⁴ |
| Top KEGG enrichment | GABAergic synapse | FDR = 2.85×10⁻⁴ |

## Methodological Note

This study employs a **simplified region-level morphometric similarity network** approach (Pearson correlation of region-level aggregated features: cortical thickness, volume, surface area) rather than the strict vertex-level morphometric distribution divergence (MIND) method proposed by Kong et al. The simplified approach was chosen for computational efficiency and reproducibility across multicenter retrospective data. This limitation is acknowledged in the manuscript.

**Resection threshold**: Regions with >10% overlap with the surgical resection mask are defined as "resected" (epileptogenic zone proxy), following established conventions in epilepsy surgery imaging research. Threshold sensitivity analysis (5%, 10%, 15%, 20%) confirms robust results across thresholds.

## Code Revisions

### v3 (10% threshold + DTI corrections + sensitivity analyses)

1. **Resection threshold changed from >0 to >10%**: The original >0% definition included regions with only incidental overlap with the surgical access corridor, introducing noise. Changed to >10% following epilepsy surgery imaging conventions.
2. **Added threshold sensitivity analysis** (5%, 10%, 15%, 20%) confirming robust results.
3. **Added HS-subgroup transcriptomic analysis** to control for surgical access heterogeneity.
4. **Added cluster permutation test** confirming clustering does not exceed chance level.
5. **Added DTI baseline comparison** (included vs excluded subjects).
6. **DTI analysis corrections**: weighted networks, distance transformation, node-level local efficiency, GroupKFold, corrected Cohen's d.
7. **Added bootstrap 95% CI** for all AUC estimates and **FDR correction** for multiple comparisons.

### v2 (methodological corrections after independent code review)

1. **Betweenness/closeness centrality weight direction** (critical bug fix): Fixed by constructing a distance matrix `distance = 1 - |similarity|`.
2. **Corrected Cohen's d formula** with sample-size-weighted pooled SD.
3. **Added within-subject paired tests** to address non-independence of regional nodes.
4. **Added patient-level classification evaluation**.
5. **Unified random_state=42** across all classifiers.
6. **All comments and docstrings converted to English**.

## Citation

If you use this code, please cite:

> Li H, Lan S, He F, Zhang D. Region-Level Morphometric Similarity Networks and DTI Connectomics for Preoperative Epileptogenic Zone Localization in Temporal Lobe Epilepsy. *Human Brain Mapping*, 2026. (under review)

And the original dataset:

> Wang Y, Vos SB, Winston GP, et al. Open diffusion MRI and connectivity data for epilepsy and surgery: The IDEAS II release. *Epilepsia*, 2026.

## License

This code is released under the MIT License. Please adhere to the data use agreements of the IDEAS-II dataset and the Allen Human Brain Atlas.
