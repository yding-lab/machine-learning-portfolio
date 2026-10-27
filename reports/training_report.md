# Training report

Generated 2026-10-27T23:01:10.655221+00:00 with scikit-learn 1.9.1 (Python 3.13.16). Educational benchmark only, not for medical use.

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
