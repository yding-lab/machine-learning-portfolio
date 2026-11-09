"""Models. All expose predict_proba(num, cat) -> P(denied), on preprocessed arrays.

- PriorModel: the training denial rate for everyone (the floor any model must beat)
- LogisticModel / BoostingModel: scikit-learn baselines on one-hot features
- MLPModel: PyTorch network with an embedding per categorical field, early-stopped on the
  validation set
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch import nn


def one_hot(num: np.ndarray, cat: np.ndarray, cardinalities: list[int]) -> np.ndarray:
    blocks = [num]
    for j, k in enumerate(cardinalities):
        oh = np.zeros((len(cat), k), dtype=np.float32)
        oh[np.arange(len(cat)), cat[:, j]] = 1
        blocks.append(oh)
    return np.hstack(blocks)


class PriorModel:
    name = "prior"

    def fit(self, num, cat, y, **_):
        self.rate = float(np.mean(y))
        return self

    def predict_proba(self, num, cat):
        return np.full(len(num), self.rate)


class LogisticModel:
    name = "logistic_regression"

    def __init__(self, cardinalities, C=1.0, seed=0):
        self.card, self.clf = cardinalities, LogisticRegression(C=C, max_iter=3000, random_state=seed)

    def fit(self, num, cat, y, **_):
        self.clf.fit(one_hot(num, cat, self.card), y)
        return self

    def predict_proba(self, num, cat):
        return self.clf.predict_proba(one_hot(num, cat, self.card))[:, 1]


class BoostingModel:
    name = "gradient_boosting"

    def __init__(self, cardinalities, learning_rate=0.1, max_leaf_nodes=31, seed=0):
        self.card = cardinalities
        # Early stopping uses an internal split of the TRAINING data only.
        self.clf = HistGradientBoostingClassifier(learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                                                  max_iter=500, early_stopping=True, validation_fraction=0.1,
                                                  n_iter_no_change=20, random_state=seed)

    def fit(self, num, cat, y, **_):
        self.clf.fit(one_hot(num, cat, self.card), y)
        return self

    def predict_proba(self, num, cat):
        return self.clf.predict_proba(one_hot(num, cat, self.card))[:, 1]


class _Net(nn.Module):
    def __init__(self, n_num, cardinalities, hidden, dropout):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(k, min(8, (k + 1) // 2)) for k in cardinalities])
        width = n_num + sum(e.embedding_dim for e in self.emb)
        layers = []
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, num, cat):
        parts = [num] + [e(cat[:, j]) for j, e in enumerate(self.emb)]
        return self.mlp(torch.cat(parts, dim=1)).squeeze(1)


class MLPModel:
    name = "pytorch_mlp"

    def __init__(self, cardinalities, hidden=(64, 32), dropout=0.1, lr=1e-3, batch_size=512, max_epochs=40,
                 patience=5, seed=0):
        self.card, self.hidden, self.dropout = cardinalities, tuple(hidden), dropout
        self.lr, self.batch_size, self.max_epochs, self.patience, self.seed = lr, batch_size, max_epochs, patience, seed
        self.history: list[dict] = []

    def fit(self, num, cat, y, val=None):
        """Train with Adam on binary cross-entropy; keep the epoch with the best validation
        average precision (early stopping). `val` = (num, cat, y) from the validation split."""
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.net = _Net(num.shape[1], self.card, self.hidden, self.dropout)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        X_num, X_cat, Y = torch.from_numpy(num), torch.from_numpy(cat), torch.from_numpy(y.astype(np.float32))
        best, best_state, waited = -1.0, None, 0
        for epoch in range(1, self.max_epochs + 1):
            self.net.train()
            order = rng.permutation(len(Y))
            total = 0.0
            for i in range(0, len(order), self.batch_size):
                b = torch.from_numpy(order[i:i + self.batch_size])
                opt.zero_grad()
                loss = loss_fn(self.net(X_num[b], X_cat[b]), Y[b])
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(b)
            row = {"epoch": epoch, "train_loss": total / len(Y)}
            if val is not None:
                ap = average_precision_score(val[2], self.predict_proba(val[0], val[1]))
                row["val_average_precision"] = float(ap)
                if ap > best + 1e-4:
                    best, best_state, waited = ap, copy.deepcopy(self.net.state_dict()), 0
                else:
                    waited += 1
            self.history.append(row)
            if val is not None and waited >= self.patience:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_epoch = max(self.history, key=lambda r: r.get("val_average_precision", -1))["epoch"]
        return self

    @torch.no_grad()
    def predict_proba(self, num, cat):
        self.net.eval()
        out = []
        for i in range(0, len(num), 8192):
            out.append(torch.sigmoid(self.net(torch.from_numpy(num[i:i + 8192]), torch.from_numpy(cat[i:i + 8192]))))
        return torch.cat(out).numpy().astype(float)
