# DrugResponse-GNN: Cross-Panel Pathway-Bottleneck Graph Neural Networks for Drug Sensitivity

![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17189237.svg)](https://doi.org/10.5281/zenodo.17189237)

**Interpretable GNN with a concept-bottleneck (pathway/TF) that predicts IC50 and explains conserved drivers across panels.**

---
## Overview

**DrugResponse-GNN** addresses a persistent challenge in pharmacogenomics: models trained on one panel (e.g., CCLE) often collapse on another (e.g., GDSC), limiting biomarker discovery and reproducibility. The framework introduces a **pathway/TF concept-bottleneck** (KPNN-style) that forces predictions to flow through biologically interpretable modules, ensuring that learned representations correspond to stable molecular programs rather than opaque embeddings. By making **cross-panel generalization** an explicit design objective, DrugResponse-GNN delivers not only predictive accuracy but also **mechanistic, reproducible explanations** of drug sensitivity across datasets.

---
## Architecture
<img width="940" height="457" alt="drugresponse_gnn_architecture" src="https://github.com/user-attachments/assets/21239a41-9232-425c-9226-0fc1068eda10" />

- **Encode** multi-omics (expr/mut/cnv) and optional drug features (targets/SMILES).  
- **Propagate** with a GNN over prior graphs (PPI, pathways, TF–target).  
- **Bottleneck** through curated **pathway/TF modules** for interpretability.  
- **Predict** IC50 (primary; GI50/AUC optional) and export concept attributions.

---
## Cross-Panel Workflow
<img width="3270" height="1170" alt="drugresponse_gnn_crosspanel_v3" src="https://github.com/user-attachments/assets/117adac8-3f04-430d-8da8-66101461c018" />

**Cross-panel protocol:** train on one panel → test on an independent panel → report prediction + attribution stability → verify drug–pathway biology in case studies (PI3K/EGFR/PARP).

---
## Datasets

DrugResponse-GNN draws on established pharmacogenomic resources, including **CCLE, GDSC, CTRP, NCI-60, and CellMinerCDB**. These datasets provide complementary molecular and drug response profiles, offering a foundation for training, testing, and external validation of cross-panel prediction performance.

---
## Methodology

- **Model backbone:** GNN with pathway-level bottleneck for interpretability  
- **Loss functions:** regression (MSE) for continuous IC50 values, optional classification heads  
- **Interpretability:** node and edge attribution (Captum, SHAP), pathway activity profiling, drug–subnetwork interaction mapping  
- **Evaluation metrics:** prediction accuracy, attribution consistency, and mechanistic recovery across panels  

Alongside prediction, the framework highlights conserved **pathway-level mechanisms of drug sensitivity** across datasets, ensuring interpretability and biological consistency.

---
## Reproducible Benchmark

On cross-panel splits (e.g., CCLE→GDSC), DrugResponse-GNN maintains accuracy while improving **pathway fidelity** (overlap ≈0.7–0.8) and **attribution stability** (≈0.9 across seeds) versus drGAT, DRPreter, and non-graph baselines. All scripts and configurations are provided for reproducibility.

---
## Project Roadmap

**Objective:** Extend DrugResponse-GNN into a reproducible framework for cross-panel pharmacogenomic analysis with interpretable pathway-level outputs.
- **Preprocessing:** Finalize data harmonization for CCLE, GDSC, CTRP, NCI-60, and CellMinerCDB.  
- **Modeling:** Complete GNN implementation with pathway/TF bottlenecks and evaluation across panels.  
- **Attribution Stability:** Quantify cross-seed reproducibility and validate conserved pathways.  
- **Case Studies:** Analyze PI3K, PARP, and EGFR inhibitor families to demonstrate interpretability.  
- **Reproducibility:** Package scripts and example notebooks for transparent, end-to-end use.

---
## Limitations & Ongoing Work

Cross-panel heterogeneity and assay mismatch remain; ongoing work includes **seed/noise robustness**, **cross-panel normalization**, and extended **baseline comparisons** (DeepChem GNNs, XGBoost).

---
## Extensibility

Beyond the current GNN–KPNN hybrid, the encoder can be replaced with transformer backbones or pretrained embeddings from foundation-scale biology models (Geneformer, scGPT, ESM). The aim is to preserve interpretability through pathway bottlenecks while leveraging the scalability of foundation models.

---
## References

1. Reinhold WC *et al.* *CellMinerCDB: a relational database for pharmacogenomics.* **Cancer Research** (2019).  
2. Barretina J *et al.* *The Cancer Cell Line Encyclopedia (CCLE).* **Nature** (2012).  
3. Yang W *et al.* *Genomics of Drug Sensitivity in Cancer (GDSC).* **Nucleic Acids Research** (2013).  
4. Seashore-Ludlow B *et al.* *CTRP.* **Cell** (2015).  
5. Yepes S. *MM-KPNN and SpatialMMKPNN* GitHub Repositories (2025).

---
## Citation

> Yepes, S. *DrugResponse-GNN: Cross-Panel Pathway-Bottleneck Graph Neural Networks for Drug Sensitivity Prediction.* GitHub, 2025.  
> DOI: [10.5281/zenodo.17189237](https://doi.org/10.5281/zenodo.17189237)

---
DrugResponse-GNN is part of the **MM-KPNN framework family**, extending interpretable modeling from single-cell and spatial data to pharmacogenomic prediction.

---
