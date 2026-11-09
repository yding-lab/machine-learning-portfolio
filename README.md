# Machine Learning Portfolio

Supervised-learning case studies on public US government data (data.gov). Each case is self-contained, with its own data snapshot, code, tests and results, and is evaluated the same way: a held-out test set touched once, simple baselines first, uncertainty on the main metrics, and an honest list of limitations.

| Case | Question | Data | Result |
|---|---|---|---|
| [mortgage-approval](mortgage-approval/) | Will a home-purchase mortgage application be denied? PyTorch MLP vs logistic regression and gradient boosting, plus a fairness audit | HMDA public loan data (CFPB / FFIEC), New York; train 2023, test 2024 | Gradient boosting AP 0.519 (PyTorch MLP 0.516, tied; baseline 0.103). Approved Black applicants are falsely flagged at 7.0% vs 3.1% for White applicants |
