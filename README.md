# 🩺 Imbalanced Diabetes Prediction: Resampling Effects on Decision Boundaries

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E.svg)
![Imbalanced-Learn](https://img.shields.io/badge/imbalanced--learn-0.9%2B-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

> **"Understanding Resampling Effects on Decision Boundaries for Imbalanced Diabetes Prediction"**  
> *A systematic study on how different class imbalance handling techniques influence predictive performance in medical screening.*

## 📌 The Medical Challenge: Class Imbalance & Overlap
In medical machine learning, specifically for diabetes screening, datasets are often heavily skewed. Non-diabetic cases dominate the data, while actual diabetic cases represent a small minority (e.g., 13.9% in our dataset). 

**The consequence:** A standard learning algorithm will become biased toward the majority class. It might achieve **deceptively high overall accuracy** (>86%) by simply predicting "Non-diabetic" most of the time, yet fail critically by missing a substantial number of true diabetic cases (**low recall**). In a clinical screening setting, false negatives (missing a disease) are far more costly than a modest reduction in overall accuracy.

This project empirically investigates whether predictive improvement is driven by **increasing minority-class samples** (Oversampling) or by **reducing boundary ambiguity and class overlap** (Undersampling/Boundary Cleaning).

## 📊 Dataset
We use the **BRFSS (Behavioral Risk Factor Surveillance System) Diabetes Health Indicators dataset**.
*   **Target:** `Diabetes_binary`
*   **Imbalance Ratio:** ~86.1% Non-diabetic vs ~13.9% Diabetic.
*   **Key Features:** BMI, High Blood Pressure, High Cholesterol, Smoking behavior, Physical Activity, Age, General Health, etc.

## 🚀 Implemented Resampling Strategies
To isolate the effect of imbalance handling from model tuning, we implement and compare several dedicated resampling strategies under a rigorous, leakage-aware pipeline:

1. **Oversampling (Minority Expansion):**
   * `SMOTE`: Synthetic Minority Over-sampling Technique.
   * `ADASYN`: Adaptive Synthetic Sampling (focuses specifically on difficult-to-learn minority regions).
2. **Undersampling (Boundary Cleaning):**
   * `ENN`: Edited Nearest Neighbors (strong boundary cleaning mechanism).
   * `Tomek Links`: Reduces direct class overlap along the decision boundary.
   * `One-Sided Selection (OSS)`: Combines Tomek Links & condensed nearest neighbors.
3. **Hybrid Methods (Expansion + Cleaning):**
   * `SMOTEENN`: Applies SMOTE followed by ENN.
   * `ADASYN+ENN`: Applies ADASYN followed by ENN.
   * `ADASYN+Tomek`: Applies ADASYN followed by Tomek Links.

## 🏆 Highlight Results
Below is a comparison of how different methods impact the performance of a **Gradient Boosting Classifier** on the held-out test set. Notice how relying on accuracy alone is dangerously misleading.

| Resampling Method | Accuracy | Precision | Recall (Sensitivity) | F1-Score | Mechanism Highlight |
|:---|:---:|:---:|:---:|:---:|:---|
| **Original (Baseline)** | 86.5% | 54.8% | **16.9%** | 0.258 | Heavily biased toward majority class |
| **SMOTE** | 82.3% | 39.8% | 52.2% | 0.451 | Expands minority coverage |
| **ENN (Best F1)** | 79.6% | 36.7% | 64.0% | **0.466** | Cleans ambiguous boundary samples |
| **ADASYN+ENN (Max Recall)**| 75.3% | 32.7% | **72.7%** | 0.451 | Aggressively targets difficult minority cases |

*(Note: Applying Logistic Regression with ADASYN+ENN pushes Recall even higher to an astonishing **83.0%**, up from the baseline 17.1%)*

### 💡 Key Findings
*   **The Baseline Trap:** Without resampling, models reach high accuracy (86.5%) but catch only **~17%** of diabetic patients. They fail their medical purpose.
*   **Quality over Quantity:** **ENN** achieves the best overall balance (F1-Score: 0.466) simply by removing ambiguous majority samples near the boundary, proving that *class overlap* is as critical an issue as class imbalance.
*   **Maximum Sensitivity:** If minimizing false negatives is the absolute priority, adaptive oversampling followed by cleaning (**ADASYN+ENN**) provides the most aggressive minority-case detection, though at the cost of more false positives (lower precision).

## 📂 Repository Structure
Our pipeline is designed to be **leakage-aware**. All preprocessing (BMI Outlier Capping via IQR, Standard Scaling, and Resampling itself) is strictly fitted on the training split only.

```bash
.
├── README.md                  # This comprehensive documentation
├── utils.py                   # Core pipeline utilities (data loading, preprocessing, cross-validation)
├── method_smote.py            # Execution script for SMOTE
├── method_adasyn.py           # Execution script for ADASYN
├── method_enn.py              # Execution script for ENN
├── method_tomek.py            # Execution script for Tomek Links
├── method_oss.py              # Execution script for One-Sided Selection
├── method_smote_enn.py        # Execution script for SMOTEENN
├── method_adasyn_enn.py       # Execution script for ADASYN+ENN
└── method_adasyn_tomek.py     # Execution script for ADASYN+Tomek
```

## 💻 How to Run

**1. Clone the repository and install dependencies:**
Ensure you have `scikit-learn`, `imbalanced-learn`, and `pandas` installed.

**2. Execute a specific pipeline:**
To reproduce the results for a specific resampling strategy, simply execute the corresponding Python script. For example, to test the best-performing **ENN** method:
```bash
python method_enn.py
```
This will automatically:
1. Load and stratify-split the BRFSS dataset.
2. Apply IQR capping and scaling strictly on the training set.
3. Apply ENN to clean the boundary.
4. Train baseline classifiers and output detailed evaluation metrics (Precision, Recall, F1).
