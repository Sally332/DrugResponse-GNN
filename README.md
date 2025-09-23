# DrugResponse-GNN
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

**Cross-Panel Pathway-Bottleneck Graph Neural Networks for Drug Sensitivity Prediction**

---

## Overview
DrugResponse-GNN is an interpretable **graph neural network (GNN)** designed to predict drug sensitivity in cancer cell lines.  
It integrates **multi-omic molecular features**, **drug descriptors**, and **prior knowledge graphs** (protein–protein interactions, pathways, regulatory networks) through a **pathway bottleneck layer**, ensuring that predictions are mechanistically grounded.  

The framework emphasizes **cross-panel generalization** (e.g., CCLE → GDSC), addressing a persistent gap in pharmacogenomics where models often fail to transfer across datasets.

---

## Architecture
<img width="940" height="457" alt="drugresponse_gnn_architecture" src="https://github.com/user-attachments/assets/21239a41-9232-425c-9226-0fc1068eda10" />

The architecture includes:
- **Inputs**:  
  - Cell line features: gene expression, mutation profiles, copy number variation  
  - Drug features: molecular descriptors, known targets  
  - Prior knowledge: curated graphs (PPI, pathways, TF–target networks)  
- **GNN Encoder**: propagates information across nodes, integrating drug–cell line relationships  
- **Pathway bottleneck layer**: forces predictions to flow through interpretable biological modules  
- **Prediction head**: outputs drug sensitivity (IC50, GI50)  

---

## Cross-Panel Workflow
<img width="3270" height="1170" alt="drugresponse_gnn_crosspanel_v3" src="https://github.com/user-attachments/assets/117adac8-3f04-430d-8da8-66101461c018" />

1. **Training panel**: one pharmacogenomic dataset (e.g., CCLE, GDSC, CTRP, NCI-60)  
2. **Model fit**: GNN with pathway bottleneck layer  
3. **Test panel**: independent dataset for cross-dataset evaluation  
4. **Evaluation**: predictive accuracy, attribution stability, recovery of known drug–pathway interactions  

---

## Datasets
This repo is designed for integration across **multiple pharmacogenomic resources**:  
- **CellMinerCDB**: harmonized access to NCI-60, GDSC, CTRP, CCLE  
- **GDSC (Genomics of Drug Sensitivity in Cancer)**  
- **CCLE (Cancer Cell Line Encyclopedia)**  
- **CTRP (Cancer Therapeutics Response Portal)**  
- **NCI-60**  

Each panel provides drug sensitivity measures linked to molecular profiles. Cross-panel benchmarking ensures generalization and robustness.

---

## Methodology
- **Model backbone**: GNN with pathway-level bottleneck for interpretability  
- **Loss functions**: regression (MSE) for continuous IC50/GI50 values, with optional classification heads  
- **Interpretability**:  
  - Node and edge attribution (Captum, SHAP)  
  - Pathway activity profiling  
  - Drug–subnetwork interaction mapping  
- **Evaluation metrics**:  
  - Prediction accuracy within and across panels  
  - Attribution consistency across datasets  
  - Identification of mechanistically consistent subnetworks  

---

## Roadmap
- [ ] Upload preprocessing scripts for CellMinerCDB  
- [ ] Implement GNN with pathway bottleneck layer  
- [ ] Integrate interpretability tools (Captum, SHAP, integrated gradients)  
- [ ] Benchmark across CCLE, GDSC, CTRP, NCI-60  
- [ ] Test generalization to unseen drugs and cell lines  
- [ ] Release reproducible analysis notebooks  

---

## References
1. Reinhold WC et al. *CellMinerCDB: a relational database for pharmacogenomics* (Cancer Research, 2019).  
2. Barretina J et al. *CCLE* (Nature, 2012).  
3. Yang W et al. *GDSC* (Nucleic Acids Research, 2013).  
4. Seashore-Ludlow B et al. *CTRP* (Cell, 2015).  
5. Yepes S. *MM-KPNN and SpatialMMKPNN* GitHub Repositories (2025).  

---

## Citation
If you use or adapt this framework, please cite:

> Yepes S. *DrugResponse-GNN: Cross-Panel Pathway-Bottleneck Graph Neural Networks for Drug Sensitivity Prediction*. GitHub, 2025.
