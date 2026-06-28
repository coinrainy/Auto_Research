import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize


def _to_numpy_mask(mask):
    if mask is None:
        return None
    return mask.detach().cpu().numpy().astype(bool)


def _fit_logreg(x_train, y_train, c_value):
    clf = LogisticRegression(solver="lbfgs", C=float(c_value), max_iter=1000)
    clf.fit(x_train, y_train)
    return clf


def linear_probe_with_masks(embeddings, labels, train_mask, val_mask, test_mask):
    x = normalize(embeddings.detach().cpu().numpy(), norm="l2")
    y = labels.detach().cpu().numpy()
    if y.ndim > 1:
        y = y.reshape(-1)
    train_mask = _to_numpy_mask(train_mask)
    val_mask = _to_numpy_mask(val_mask)
    test_mask = _to_numpy_mask(test_mask)

    c_values = 2.0 ** np.arange(-10, 10)
    best_c = c_values[0]
    best_score = -1.0
    if val_mask is not None and val_mask.any():
        for c_value in c_values:
            clf = _fit_logreg(x[train_mask], y[train_mask], c_value)
            pred = clf.predict(x[val_mask])
            score = f1_score(y[val_mask], pred, average="micro", zero_division=0)
            if score > best_score:
                best_score = score
                best_c = c_value
    else:
        clf = GridSearchCV(
            LogisticRegression(solver="lbfgs", max_iter=1000),
            param_grid={"C": c_values},
            n_jobs=4,
            cv=5,
            verbose=0,
        )
        clf.fit(x[train_mask], y[train_mask])
        best_c = clf.best_params_["C"]

    clf = _fit_logreg(x[train_mask], y[train_mask], best_c)
    pred = clf.predict(x[test_mask])
    return {
        "accuracy": float(accuracy_score(y[test_mask], pred)),
        "F1Mi": float(f1_score(y[test_mask], pred, average="micro", zero_division=0)),
        "F1Ma": float(f1_score(y[test_mask], pred, average="macro", zero_division=0)),
        "best_c": float(best_c),
    }


def linear_probe_random(embeddings, labels, ratio, seed, repeats=3):
    x = normalize(embeddings.detach().cpu().numpy(), norm="l2")
    y = labels.detach().cpu().numpy().reshape(-1, 1)
    encoder = OneHotEncoder(categories="auto").fit(y)
    y = encoder.transform(y).toarray().astype(bool)
    metrics = []
    for idx in range(repeats):
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=1.0 - ratio,
            random_state=seed + idx,
        )
        base = LogisticRegression(solver="liblinear")
        clf = GridSearchCV(
            estimator=OneVsRestClassifier(base),
            param_grid={"estimator__C": 2.0 ** np.arange(-10, 10)},
            n_jobs=4,
            cv=5,
            verbose=0,
        )
        clf.fit(x_train, y_train)
        prob = clf.predict_proba(x_test)
        pred = np.zeros(prob.shape, dtype=bool)
        pred[np.arange(prob.shape[0]), np.argmax(prob, axis=1)] = True
        metrics.append({
            "accuracy": float((pred == y_test).all(axis=1).mean()),
            "F1Mi": float(f1_score(y_test, pred, average="micro", zero_division=0)),
            "F1Ma": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        })
    keys = metrics[0].keys()
    return {key: float(np.mean([item[key] for item in metrics])) for key in keys}

