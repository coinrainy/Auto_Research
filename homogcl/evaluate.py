from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .utils import accuracy, set_seed


def _standardize_from_train(
    x: torch.Tensor,
    train_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x[train_mask].mean(dim=0, keepdim=True)
    std = x[train_mask].std(dim=0, keepdim=True).clamp_min(1e-6)
    return (x - mean) / std, mean, std


def ridge_regression_probe(
    emb: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    num_classes: int,
    alphas: Sequence[float],
) -> dict[str, float | int]:
    x = emb.detach().double()
    x, _, _ = _standardize_from_train(x, train_mask)
    ones = torch.ones(x.size(0), 1, dtype=x.dtype, device=x.device)
    x = torch.cat([x, ones], dim=1)
    y_onehot = F.one_hot(y, num_classes=num_classes).double()

    best = {
        "train_acc": 0.0,
        "val_acc": -1.0,
        "test_acc": 0.0,
        "best_epoch": 0,
        "ridge_alpha": float(alphas[0]),
    }
    xt = x[train_mask]
    yt = y_onehot[train_mask]
    eye = torch.eye(xt.size(1), dtype=x.dtype, device=x.device)
    eye[-1, -1] = 0.0
    for alpha in alphas:
        lhs = xt.t() @ xt + float(alpha) * eye
        rhs = xt.t() @ yt
        weight = torch.linalg.solve(lhs, rhs)
        logits = x @ weight
        train_acc = accuracy(logits[train_mask], y[train_mask])
        val_acc = accuracy(logits[val_mask], y[val_mask])
        test_acc = accuracy(logits[test_mask], y[test_mask])
        if val_acc > best["val_acc"]:
            best = {
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "best_epoch": 0,
                "ridge_alpha": float(alpha),
            }
    return best


def sklearn_logistic_regression_probe(
    emb: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    c_grid: Sequence[float],
) -> dict[str, float | int]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import normalize

    x = normalize(emb.detach().cpu().numpy(), norm="l2")
    labels = y.detach().cpu().numpy()
    train = train_mask.detach().cpu().numpy().astype(bool)
    val = val_mask.detach().cpu().numpy().astype(bool)
    test = test_mask.detach().cpu().numpy().astype(bool)

    best = {
        "train_acc": 0.0,
        "val_acc": -1.0,
        "test_acc": 0.0,
        "best_epoch": 0,
        "logreg_C": float(c_grid[0]),
    }
    for c in c_grid:
        clf = OneVsRestClassifier(
            LogisticRegression(solver="liblinear", C=float(c), max_iter=1000)
        )
        clf.fit(x[train], labels[train])
        train_pred = np.argmax(clf.predict_proba(x[train]), axis=1)
        val_pred = np.argmax(clf.predict_proba(x[val]), axis=1)
        test_pred = np.argmax(clf.predict_proba(x[test]), axis=1)
        train_acc = float((train_pred == labels[train]).mean())
        val_acc = float((val_pred == labels[val]).mean())
        test_acc = float((test_pred == labels[test]).mean())
        if val_acc > best["val_acc"]:
            best = {
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "best_epoch": 0,
                "logreg_C": float(c),
            }
    return best


def torch_logistic_regression_probe(
    emb: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    num_classes: int,
    seed: int,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    epochs: int = 1000,
    patience: int = 100,
) -> dict[str, float | int]:
    set_seed(seed)
    device = emb.device
    x = emb.detach()
    model = nn.Linear(x.size(1), num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0
    best_epoch = 0
    stale = 0
    for epoch in range(epochs):
        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(x)
            val_acc = accuracy(logits[val_mask], y[val_mask])
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        return {
            "train_acc": accuracy(logits[train_mask], y[train_mask]),
            "val_acc": accuracy(logits[val_mask], y[val_mask]),
            "test_acc": accuracy(logits[test_mask], y[test_mask]),
            "best_epoch": int(best_epoch),
        }


def linear_probe(
    emb: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    num_classes: int,
    seed: int,
    probe: str = "sklogreg",
    alphas: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
    c_grid: Sequence[float] = tuple(2.0**i for i in range(-10, 11)),
    lr: float = 0.01,
    weight_decay: float = 0.0,
    epochs: int = 1000,
    patience: int = 100,
) -> dict[str, float | int]:
    if probe == "sklogreg":
        return sklearn_logistic_regression_probe(
            emb=emb,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            c_grid=c_grid,
        )
    if probe == "ridge":
        return ridge_regression_probe(
            emb=emb,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_classes=num_classes,
            alphas=alphas,
        )
    if probe in {"torchlogreg", "logreg"}:
        return torch_logistic_regression_probe(
            emb=emb,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_classes=num_classes,
            seed=seed,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            patience=patience,
        )
    raise ValueError(f"Unsupported linear probe: {probe}")
