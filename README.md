# Physics-informed and interpretable state of health estimation framework for lithium-ion batteries

> **Authors:**
Lei Liu, Jiahui Huang, Bo Peng, Ying Li, Peng Wang, Xianhao Wang, Hongwei Zhao, Bin Li.

This repo contains the code and data from our paper published in Journal of Energy Storage.

Website: .

## Table of contents

- [Physics-informed and interpretable state of health estimation framework for lithium-ion batteries](#physics-informed-and-interpretable-state-of-health-estimation-framework-for-lithium-ion-batteries)
  - [Table of contents](#table-of-contents)
  - [1. Abstract](#1-abstract)
  - [2. PIKAN's architecture](#2-pikans-architecture)
  - [3. Environment configuration](#3-environment-configuration)
    - [3.1. Create environment](#31-create-environment)
    - [3.2. Activate environment](#32-activate-environment)
  - [4. Datasets](#4-datasets)
  - [5. Usage](#5-usage)
    - [5.1. Model training and validation](#51-model-training-and-validation)
      - [5.1.1. Regular and small-sample scenarios](#511-regular-and-small-sample-scenarios)
      - [5.1.2. Zero-shot and few-shot transfer scenarios](#512-zero-shot-and-few-shot-transfer-scenarios)
    - [5.2. Model performance evaluation](#52-model-performance-evaluation)
      - [5.2.1. Regular and small-sample scenarios](#521-regular-and-small-sample-scenarios)
      - [5.2.2. Zero-shot and few-shot transfer scenarios](#522-zero-shot-and-few-shot-transfer-scenarios)
    - [5.3. Model interpretability analysis](#53-model-interpretability-analysis)
    - [5.4. Visualization](#54-visualization)
  - [6. Experimental procedures](#6-experimental-procedures)
  - [7. Acknowledgments](#7-acknowledgments)
  - [8. Citation](#8-citation)

## 1. Abstract

Accurate state-of-health (SOH) estimation is critical for the safety of lithium-ion batteries. Current data-driven methods face three key challenges: limited accuracy under data scarcity, insufficient physical consistency, and poor interpretability. This study proposes the Physics-Informed Kolmogorov-Arnold Network (PIKAN), establishing a novel interpretable symbiotic co-evolution mechanism that combines and jointly optimizes SOH estimation and degradation mechanism discovery. PIKAN integrates three innovations: (1) the State of Health Estimation Module (SHEM) utilizes the functional decomposition and symbolic representation capabilities of KAN to achieve accurate and interpretable estimation; (2) the Degradation Dynamics Modeling Module (DDMM) uses KAN as a general approximator to adaptively discover degradation dynamical equations without predefined forms; (3) these modules interact bidirectionally, where SHEM provides real-time battery state to guide the equation discovery of DDMM, while DDMM regularizes the output of SHEM through physical constraints, achieving self-regularization and synergistic learning. Comprehensive experiments across four datasets demonstrate that PIKAN outperforms state-of-the-art methods in conventional and transfer scenarios, with maximum 63.4% RMSE reduction in small-sample scenarios. Crucially, symbolic regression transforms the learned model into explicit analytical expressions, revealing the intrinsic decision-making logic of SOH mapping and degradation dynamics. Under mechanism guidance, discovered degradation equations exhibit structural similarity to existing electrochemical models, bridging data-driven discovery with physical theory. The analytical expressions obtained from the PIKAN framework provide actionable insights for practical battery management systems (BMS) and exhibit a certain degree of cross-battery transferability. This study offers a physics-informed interpretable paradigm for intelligent battery management.

## 2. PIKAN's architecture

![PIKAN's architecture](Charts/PIKAN_architecture.jpg "PIKAN's architecture")

The overall architecture of PIKAN. (a) The structure of SHEM and DDMM, and the design of the composite loss function. (b) The schematic diagram of symbolic regression technology based on KAN.

## 3. Environment configuration

### 3.1. Create environment

```bash
conda env create -f pikan.yaml
```
### 3.2. Activate environment

```bash
conda activate pikan
```

## 4. Datasets

![datasets](Charts/Datasets.jpg "datasets")

The capacity degradation curves of the battery datasets. (a) XJTU dataset, (b) TJU dataset, (c) MIT dataset, (d) HUST dataset. The scales on the color bars of
panel (c) and (d) indicate the total number of cycles for each battery cell.

Four publicly available lithium-ion battery datasets have been placed in the `Data/` folder.

The XJTU dataset is available at: https://doi.org/10.5281/zenodo.10963339.

The TJU dataset is available at: https://zenodo.org/record/6405084.

The HUST dataset is available at: https://data.mendeley.com/datasets/nsc7hnsg4s/2.

The MIT dataset is available at: https://data.matr.io/1/projects/5c48dd2bc625d700019f3204.

## 5. Usage

### 5.1. Model training and validation

#### 5.1.1. Regular and small-sample scenarios

Taking the XJTU dataset as an example, the model training and evaluation process is as follows:

```bash
# Use hyperparameter search
python main_our_models_optimize_XJTU.py
# Hyperparameter search is not used
python main_comparision_XJTU.py
```

#### 5.1.2. Zero-shot and few-shot transfer scenarios

All pretrained model parameter files are saved in the `pretrained_models/` folder.

```bash
python main_our_models_fine_tuning.py
```

### 5.2. Model performance evaluation

#### 5.2.1. Regular and small-sample scenarios

Taking the XJTU dataset as an example, the result analysis process is as follows:

```bash
python results_analysis_code/Model_optimize_XJTU_results.py
python results_analysis_code/Model_optimize_XJTU_results_group_mean.py
python results_analysis_code/Comparision_results_XJTU.py
```

#### 5.2.2. Zero-shot and few-shot transfer scenarios

```bash
python results_analysis_code/FineTune_results.py
```

### 5.3. Model interpretability analysis

All interpretability analysis code is saved in the `Notebooks/` folder. Taking the XJTU dataset as an example, the interpretability analysis process is as follows:

Step 1. Dataset construction
   - `Notebooks/XJTU_Dataloader.ipynb`

Step 2. Direct symbolizaton
   - `Notebooks/PIKAN_XJTU_Var2.ipynb`
  
Step 3. Prior-guided symbolizatioin
   - `Notebooks/PIKAN_XJTU_Var2_PDEAutoDiscovery.ipynb`

All images and videos generated by interpretable analysis are saved in the `Images/` and `Videos/` folders, respectively.

### 5.4. Visualization

All visualizations are saved in the `Figures/` folder. Taking Fig. 1(a) in the paper as an example, the visualization process is as follows:

```bash
python plotter/Figure_1a.py
```

## 6. Experimental procedures

![experimental framework](Charts/framework.jpg "experimental framework")

The overall experimental procedures for the proposed PIKAN method.

## 7. Acknowledgments

Work & Code is inspired by https://github.com/wang-fujin/PINN4SOH.

## 8. Citation

If you find our work useful in your research, please consider citing:

```latex

```

If you have any problems, contact me via liulei13@ustc.edu.cn.        