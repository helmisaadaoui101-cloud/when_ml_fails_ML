# When Machine Learning Fails
### Diagnosing and Repairing Failure Modes in a Non-Linear Model

*Mini-project — Ecole Centrale Casablanca · Introduction to AI & Machine Learning*

---

## Overview

This project investigates a well-documented yet underestimated failure mode in supervised learning: a model that achieves high global accuracy while completely failing to detect the minority class.

Using the **Online Shoppers Purchasing Intention Dataset**, we study how a baseline Random Forest behaves under strong class imbalance, and evaluate targeted correction strategies to recover minority-class detection.

The project focuses on:

- Failure diagnosis through structured analysis
- Class imbalance quantification and its effects on model behaviour
- Model evaluation beyond accuracy — using recall, macro-F1, and ROC-AUC
- Minority-class recall optimization via class weighting
- Controlled experimentation and temporal drift analysis

---

## Research Question

> Can a baseline Random Forest trained without any imbalance correction achieve high overall accuracy while producing near-zero recall on the minority class — and can class weighting significantly improve that recall without substantially degrading macro-F1?

---

## Dataset

**Online Shoppers Purchasing Intention Dataset**
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)

### Dataset Characteristics

| Property           | Value               |
|--------------------|---------------------|
| Total sessions     | 12,330              |
| Buyers (True)      | 1,908 — 15.5%       |
| Non-buyers (False) | 10,422 — 84.5%      |
| Imbalance ratio    | 5.5 : 1             |
| Missing values     | 0                   |
| Features           | 17                  |

### Feature Groups

**Behavioural**
Administrative, informational, and product pages visited; bounce rates; exit rates; page value score.

**Temporal**
Month; weekend indicator; special day proximity.

**Technical**
Browser; operating system; region; traffic type; visitor type.

---

## Baseline Model

A Random Forest Classifier trained with no imbalance correction serves as the reference point.

### Hyperparameters

| Parameter          | Value                |
|--------------------|----------------------|
| n_estimators       | 100                  |
| class_weight       | None                 |
| random_state       | 42                   |
| Train / test split | 80 / 20, stratified  |

### Main Failure Observed

| Metric         | Value  | Interpretation                                                |
|----------------|--------|---------------------------------------------------------------|
| Accuracy       | ~87%   | Misleadingly high — the model defaults to the majority class  |
| Buyer recall   | ~0.55  | Minority-class detection nearly collapses                     |
| ROC-AUC        | 0.916  | Discriminability is preserved, but threshold calibration is skewed |

This confirms that **accuracy alone is a deceptive metric under class imbalance**: the model effectively ignores the minority class while appearing to perform well on a surface level.

---

## Objectives

1. **Diagnose** the root cause of the failure mode
2. **Quantify** the impact of class imbalance on model behaviour
3. **Improve** minority-class recall through class weighting
4. **Compare** baseline and corrected models across multiple metrics
5. **Analyze** potential temporal drift effects on predictive performance

---

## Project Structure

```
when-ml-fails/
├── README.md
├── requirements.txt
└── when_ml_fails.ipynb
```

---

## Key Concepts

- **Class imbalance** — when one class significantly outnumbers another, standard classifiers are biased toward the majority class.
- **Recall** — the fraction of actual positive cases correctly identified; the critical metric for business-sensitive minority-class detection.
- **Macro-F1** — the unweighted average of F1 scores across classes; a balanced measure of overall performance.
- **ROC-AUC** — measures the model's ability to discriminate between classes across all thresholds, independent of class distribution.
- **Class weighting** — a correction strategy that penalizes misclassification of the minority class during training.

---

*Keywords: class imbalance · Random Forest · recall optimization · macro-F1 · binary classification · e-commerce · failure mode diagnosis*
