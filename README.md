# Machine Learning Portfolio

Supervised-learning case studies on public US government data (data.gov). Each case is self-contained, with its own data snapshot, code, tests and results, and is evaluated the same way: a held-out test set touched once, simple baselines first, uncertainty on the main metrics, and an honest list of limitations.

| Case | Question | Data | Result |
|---|---|---|---|
| [mortgage-approval](mortgage-approval/) | Will a home-purchase mortgage application be denied? PyTorch MLP vs logistic regression and gradient boosting, plus a fairness audit | HMDA public loan data (CFPB / FFIEC), New York; train 2023, test 2024 | Gradient boosting AP 0.519 (PyTorch MLP 0.516, tied; baseline 0.103). Approved Black applicants are falsely flagged at 7.0% vs 3.1% for White applicants |
| [mars-image-classification](mars-image-classification/) | Which of 24 classes does a Curiosity rover image show, and does a model trained on the first 180 sols still work 400-900 sols later? Small CNN vs ResNet-18 linear probe vs fine-tuning (PyTorch) | NASA JPL Mars surface image labeled data set (Curiosity), official chronological split | Fine-tuned ResNet-18 test macro-F1 0.688 (95% CI 0.651-0.713), tied with the linear probe; validation overstated it by 0.19 |
