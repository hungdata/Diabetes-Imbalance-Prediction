# Analyzing Resampling Effects on Class Decision Boundaries for Diabetes Prediction

This repository contains the source code for our study on the impact of various resampling strategies for diabetes prediction under severe class imbalance.

## Dataset
We utilize the BRFSS (Behavioral Risk Factor Surveillance System) Diabetes Health Indicators dataset. In this dataset, the positive (diabetic) class represents only **13.9%** of the observations.

## Methodology
To thoroughly evaluate the mechanisms of different imbalance-handling techniques, we implemented a robust pipeline that prevents data leakage:
1. **Stratified Splitting:** 80/20 train/test split.
2. **Preprocessing:** BMI outlier capping using the IQR method and `StandardScaler` feature scaling.
3. **Resampling Techniques:** 
   - Oversampling: SMOTE, ADASYN
   - Undersampling: ENN, Tomek Links, One-Sided Selection (OSS)
   - Hybrid: SMOTE+ENN, ADASYN+ENN, ADASYN+Tomek

## Repository Structure
We have modularized our implementations to allow researchers to easily run and compare specific methods:
- `utils.py`: Contains the `load_and_preprocess_data` pipeline.
- `method_smote.py`: Implementation of SMOTE.
- `method_adasyn.py`: Implementation of ADASYN.
- `method_enn.py`: Implementation of Edited Nearest Neighbours (ENN).
- `method_tomek.py`: Implementation of Tomek Links.
- `method_oss.py`: Implementation of One-Sided Selection.
- `method_smote_enn.py`: Implementation of SMOTE+ENN.
- `method_adasyn_enn.py`: Implementation of ADASYN+ENN.
- `method_adasyn_tomek.py`: Implementation of ADASYN+Tomek.

## Getting Started
Ensure you have the required libraries installed:
```bash
pip install pandas scikit-learn imbalanced-learn numpy
```

To evaluate a specific resampling method, simply run the corresponding script. For example, to run the ENN method (which achieved the best F1-score in our study):
```bash
python method_enn.py
```

*Note: The original paper (`APWEB_2026_Diabetes_Prediction.pdf`) will be uploaded to this repository soon.*
