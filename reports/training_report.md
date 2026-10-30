# Training report

Generated 2026-10-30T22:30:18.571411+00:00 with scikit-learn 1.9.1 (Python 3.13.16). Educational benchmark only, not for medical use.

Data: 569 rows (212 malignant, 357 benign); stratified split 455 train / 114 test. Target encoding: 0 = malignant, 1 = benign (positive class = benign).

## Model selection (highest mean 5-fold CV roc_auc on the training split)

| Candidate | CV ROC-AUC (mean ± sd) | Train ROC-AUC | Grid size | Best params |
|---|---|---|---|---|
| logistic_regression **(selected)** | 0.9965 ± 0.0049 | 0.9989 | 6 | `{'model__C': 10.0, 'model__l1_ratio': 1.0}` |
| random_forest | 0.9917 ± 0.0043 | 1.0000 | 36 | `{'model__max_depth': None, 'model__max_features': 'sqrt', 'model__min_samples_leaf': 2, 'model__n_estimators': 200}` |

## Held-out test set (evaluated once, threshold 0.5)

| Metric | Value |
|---|---|
| roc_auc | 0.9954 |
| average_precision | 0.9971 |
| accuracy | 0.9386 |
| balanced_accuracy | 0.9464 |
| precision | 0.9851 |
| recall | 0.9167 |
| f1 | 0.9496 |
| brier_score | 0.0340 |
| recall, malignant class | 0.9762 |
| precision, malignant class | 0.8723 |

Confusion matrix (rows = true, columns = predicted; order malignant, benign):

| | pred malignant | pred benign |
|---|---|---|
| true malignant | 41 | 1 |
| true benign | 6 | 66 |

## Uncertainty on the test set

Probabilities are P(benign); a case is called malignant when P(benign) < threshold. Test set: 114 rows (42 malignant, 72 benign). Intervals are 95%: Wilson score for count proportions, and a stratified bootstrap (2000 resamples of the test set, model held fixed) for the rest. They reflect test-set sampling only, not variation from re-training, and with 42 malignant cases one error moves malignant recall by about 2.4 points.

| Metric (threshold 0.5) | Value | 95% CI | Method |
|---|---|---|---|
| ROC-AUC | 0.995 | 0.983 – 1.000 | stratified bootstrap |
| Average precision, malignant | 0.994 | 0.979 – 1.000 | stratified bootstrap |
| Brier score (P(benign)) | 0.034 | 0.014 – 0.061 | stratified bootstrap |
| Accuracy | 93.9% | 88.6% – 97.4% | stratified bootstrap |
| Malignant recall (sensitivity) | 97.6% (41/42) | 87.7% – 99.6% | Wilson |
| Malignant precision | 87.2% (41/47) | 74.8% – 94.0% | Wilson |
| Benign recall (specificity for malignant) | 91.7% (66/72) | 83.0% – 96.1% | Wilson |

## Threshold trade-off (training split, out-of-fold)

Computed from out-of-fold predictions on the training split, never from the test set.

| Threshold on P(benign) | Malignant caught | Malignant missed | Benign flagged | Malignant recall | Malignant precision |
|---|---|---|---|---|---|
| 0.3 | 161 | 9 | 2 | 0.947 | 0.988 |
| 0.4 | 165 | 5 | 2 | 0.971 | 0.988 |
| 0.5 | 167 | 3 | 6 | 0.982 | 0.965 |
| 0.6 | 167 | 3 | 11 | 0.982 | 0.938 |
| 0.7 | 167 | 3 | 14 | 0.982 | 0.923 |
| 0.8 | 168 | 2 | 19 | 0.988 | 0.898 |
| 0.9 | 169 | 1 | 26 | 0.994 | 0.867 |

Operating-point rule: lowest threshold with out-of-fold malignant recall >= 0.98 on the training split. It gives threshold **0.490** (out-of-fold: recall 0.982, precision 0.971). Evaluated once on the test set:

| Threshold | Malignant recall | 95% CI | | Malignant precision | 95% CI | |
|---|---|---|---|---|---|---|
| 0.5 (default) | 97.6% (41/42) | 87.7% – 99.6% | Wilson | 87.2% (41/47) | 74.8% – 94.0% | Wilson |
| 0.490 (CV rule) | 97.6% (41/42) | 87.7% – 99.6% | Wilson | 87.2% (41/47) | 74.8% – 94.0% | Wilson |

The saved model and the API use the default threshold 0.5.

## Calibration

No calibrator is fitted; calibration is checked, not adjusted. Mean predicted P(benign) vs observed benign rate in 5 quantile bins:

| Split | Bins (predicted → observed, n) | Brier |
|---|---|---|
| training, out-of-fold | 0.00 → 0.00 (91); 0.13 → 0.16 (91); 0.94 → 0.98 (91); 1.00 → 0.99 (91); 1.00 → 1.00 (91) | 0.018 |
| test | 0.00 → 0.00 (23); 0.10 → 0.22 (23); 0.86 → 0.95 (22); 0.99 → 1.00 (23); 1.00 → 1.00 (23) | 0.034 |

Figures: `reports/figures/confusion_matrix.png`, `reports/figures/roc_pr_curves.png`, `reports/figures/calibration.png`, `reports/figures/threshold_tradeoff.png`
